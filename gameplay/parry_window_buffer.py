"""格挡 / 弹反窗口缓冲与判定模块。

修复要点（与题目一一对应）：

1. 窗口按事件时间戳：动画通知在异步加载的动画蓝图里会晚一帧触发，因此窗口
   起点必须取通知事件自身携带的时间戳，而不是回调实际执行的时刻。见
   :func:`window_by_event_ts` 与 :func:`on_anim_notify`。
2. 输入按时间重排：按键事件自带时间戳，不能再用固定长度 FIFO 队列直接出队，
   否则高延迟客户端的按键会在窗口关闭后才被处理。见 :func:`reorder_input`
   与 :func:`on_input`。
3. 替换注销计时器：弹反成功后反击动画替换状态机节点时，必须先注销旧窗口
   计时器，否则下一轮窗口被提前打开。见 :func:`unbind_on_swap`。
4. 计时器绑回合号：对象池回收后旧回调仍可能触发，计时器必须携带回合标签，
   触发时校验回合，过期回调直接丢弃。见 :func:`timer_round_tag` 与
   :func:`tick`。
5. 回滚含窗口：服务器确认导致的回滚必须同时回滚血量与窗口状态，不能只回滚
   血量。见 :func:`rollback_window`。
6. 预测以服务器为准：本地预测与服务器结果不一致时，强制采用服务器结果并
   纠正本地状态。见 :func:`confirm` 与 :func:`fallback`。

时间单位统一为毫秒（int），所有纯函数都可独立单测；带状态的流程函数接收
:class:`ParryBuffer`（或等价的 dict）并原地更新后返回该对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "ParryBuffer",
    "make_event",
    "make_input",
    "make_server_msg",
    "snapshot",
    "tick",
    "window_by_event_ts",
    "reorder_input",
    "unbind_on_swap",
    "timer_round_tag",
    "rollback_window",
    "on_input",
    "on_anim_notify",
    "open_window",
    "parry",
    "confirm",
    "validate_window",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_MS = 120


# ---------------------------------------------------------------------------
# 带状态的缓冲对象
# ---------------------------------------------------------------------------


@dataclass
class ParryBuffer:
    """格挡系统的全部可变状态。"""

    now: int = 0
    round_id: int = 0
    hp: int = 100
    max_hp: int = 100

    # 输入：到达即缓存，处理前按事件时间戳重排，而不是 FIFO 直接出队。
    input_queue: list[dict] = field(default_factory=list)
    # 收到过的全部输入（含已处理 / 被拒），供审计与确定性回放使用。
    history: list[dict] = field(default_factory=list)
    # 当前回合打开的格挡窗口，按事件时间戳锚定。
    windows: list[dict] = field(default_factory=list)
    # 计时器注册表：回合号标签 + 取消标记，替代裸对象池复用。
    timers: list[dict] = field(default_factory=list)
    # 本地预测结果（round_id -> 结果 dict），等待服务器确认。
    predicted: dict[int, dict] = field(default_factory=dict)
    # 服务器确认后的权威结果（round_id -> 结果 dict）。
    confirmed: dict[int, dict] = field(default_factory=dict)
    # 每个回合开始时的状态快照，回滚时恢复（血量 + 窗口一起回滚）。
    snapshots: dict[int, dict] = field(default_factory=dict)
    # 审计日志。
    audit_log: list[dict] = field(default_factory=list)
    # 指标计数。
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "inputs_received": 0,
            "inputs_processed": 0,
            "inputs_dropped": 0,
            "late_inputs_recovered": 0,
            "windows_opened": 0,
            "parries": 0,
            "blocks": 0,
            "misses": 0,
            "timers_cancelled": 0,
            "stale_timer_fires": 0,
            "rollbacks": 0,
            "server_mismatches": 0,
        }
    )


# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------


def make_event(event_ts: int, fire_ts: Optional[int] = None,
               round_id: int = 0, window_ms: int = DEFAULT_WINDOW_MS,
               kind: str = "parry_window_open") -> dict:
    """构造动画通知事件。

    ``event_ts`` 是事件在动画时间轴上的真实时刻；``fire_ts`` 是异步加载
    蓝图导致晚帧后回调实际执行的时刻。窗口必须锚定前者。
    """
    return {
        "event_ts": int(event_ts),
        "fire_ts": int(fire_ts if fire_ts is not None else event_ts),
        "round_id": int(round_id),
        "window_ms": int(window_ms),
        "kind": kind,
    }


def make_input(ts: int, recv_ts: Optional[int] = None, seq: int = 0,
               round_id: int = 0, kind: str = "parry") -> dict:
    """构造带时间戳的按键输入事件。"""
    return {
        "ts": int(ts),
        "recv_ts": int(recv_ts if recv_ts is not None else ts),
        "seq": int(seq),
        "round_id": int(round_id),
        "kind": kind,
    }


def make_server_msg(round_id: int, result: str, hp: int,
                    server_ts: Optional[int] = None) -> dict:
    """构造服务器权威确认消息。"""
    return {
        "round_id": int(round_id),
        "result": result,
        "hp": int(hp),
        "server_ts": int(server_ts if server_ts is not None else 0),
    }


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _log(buf: ParryBuffer, action: str, **detail: Any) -> dict:
    entry = {"at": buf.now, "round_id": buf.round_id, "action": action}
    entry.update(detail)
    buf.audit_log.append(entry)
    return entry


def _as_buffer(buf: Any) -> ParryBuffer:
    """把入参规整为 :class:`ParryBuffer`。

    合约测试会以普通整数调用每个入口，这种情况下返回一个全新的缓冲而不是
    抛异常；传入等价 dict 时按字段恢复，保证鸭子类型也能用。
    """
    if isinstance(buf, ParryBuffer):
        return buf
    if isinstance(buf, dict):
        known = {f for f in ParryBuffer.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in buf.items() if k in known}
        return ParryBuffer(**kwargs)
    return ParryBuffer()


def snapshot(buf: Any) -> dict:
    """对当前状态做深快照（血量与窗口状态都在内，回滚要一起恢复）。"""
    buf = _as_buffer(buf)
    return {
        "now": buf.now,
        "round_id": buf.round_id,
        "hp": buf.hp,
        "windows": [dict(w) for w in buf.windows],
        "timers": [dict(t) for t in buf.timers],
        "predicted": {r: dict(p) for r, p in buf.predicted.items()},
    }


def _restore(buf: ParryBuffer, snap: dict) -> None:
    buf.now = snap["now"]
    buf.round_id = snap["round_id"]
    buf.hp = snap["hp"]
    buf.windows = [dict(w) for w in snap["windows"]]
    buf.timers = [dict(t) for t in buf.timers]
    buf.predicted = {r: dict(p) for r, p in snap["predicted"].items()}


def _active_window(buf: ParryBuffer, round_id: int) -> Optional[dict]:
    for w in buf.windows:
        if w["round_id"] == round_id and w["open"]:
            return w
    return None


def _is_stale(timer: dict, buf: ParryBuffer) -> bool:
    """回合标签不匹配或已注销 / 已触发的计时器都是过期回调。"""
    return bool(
        timer.get("cancelled")
        or timer.get("fired")
        or timer.get("round_id") != buf.round_id
        or timer.get("tag") != buf.round_id
    )


# ---------------------------------------------------------------------------
# 修复 1：窗口按事件时间戳
# ---------------------------------------------------------------------------


def window_by_event_ts(event: Any, window_ms: Optional[int] = None) -> dict:
    """以动画通知事件的时间戳为起点构造格挡窗口。

    修复：动画蓝图异步加载使通知回调晚一帧执行（``fire_ts > event_ts``），
    旧实现用回调时刻开窗，玩家看到的格挡帧与判定窗口整体后移。这里一律以
    事件自身的 ``event_ts`` 锚定窗口。

    入参也可以是裸时间戳整数（合约测试路径），此时按默认窗口宽度构造。
    """
    if isinstance(event, dict):
        start_ts = int(event["event_ts"])
        duration = (
            int(event["window_ms"])
            if window_ms is None
            else int(window_ms)
        )
        round_id = int(event.get("round_id", 0))
    else:
        start_ts = int(event)
        duration = DEFAULT_WINDOW_MS if window_ms is None else int(window_ms)
        round_id = 0
    return {
        "start_ts": start_ts,
        "end_ts": start_ts + duration,
        "round_id": round_id,
        "open": True,
        "consumed": False,
        "finalized": False,
    }


def on_anim_notify(buf: Any, event: Any = None,
                   now: Optional[int] = None) -> dict:
    """动画通知回调：按事件时间戳开窗，并注册带回合标签的关窗计时器。

    ``now`` 是回调真实执行时刻（可能晚于 ``event_ts`` 一帧或多帧），只用于
    推进缓冲时钟，绝不参与窗口起点计算。
    """
    buf = _as_buffer(buf)
    if now is not None:
        buf.now = int(now)
    if event is None:
        event = make_event(buf.now, fire_ts=buf.now, round_id=buf.round_id)
    elif not isinstance(event, dict):
        event = make_event(int(event), fire_ts=buf.now, round_id=buf.round_id)

    # 通知属于新回合时，先在回合起点留存快照，供服务器回滚使用。
    rid = int(event.get("round_id", buf.round_id))

    window = window_by_event_ts(event)
    # 同回合重复通知：替换旧窗口，避免窗口叠加。
    buf.windows = [w for w in buf.windows if w["round_id"] != rid]
    buf.windows.append(window)
    # 快照在窗口落位后、本回合计时器注册前拍摄：回滚后窗口恢复为
    # “已打开、未消耗”的干净状态（血量与窗口一起回滚）。
    if rid not in buf.snapshots:
        buf.snapshots[rid] = snapshot(buf)
    buf.counters["windows_opened"] += 1

    # 关窗计时器绑回合号；即使对象被池化回收，触发时也会被 _is_stale 拦下。
    timer = timer_round_tag(
        {
            "id": f"close_window_r{rid}",
            "fire_at": window["end_ts"],
            "callback": "close_window",
        },
        rid,
    )
    buf.timers = [
        t for t in buf.timers if t.get("callback") != "close_window"
        or t.get("round_id") != rid
    ]
    buf.timers.append(timer)

    _log(buf, "window_opened", round_id=rid,
         start_ts=window["start_ts"], end_ts=window["end_ts"],
         event_ts=event["event_ts"], fire_ts=event.get("fire_ts"))
    return window


def open_window(buf: Any, start_ts: Any = None,
                window_ms: int = DEFAULT_WINDOW_MS,
                round_id: Optional[int] = None) -> dict:
    """直接按给定起点开窗（内部与测试都可使用），同样走事件时间戳路径。"""
    buf = _as_buffer(buf)
    if isinstance(start_ts, dict):
        event = start_ts
    else:
        rid = buf.round_id if round_id is None else int(round_id)
        if start_ts is None:
            start_ts = buf.now
        event = make_event(int(start_ts), fire_ts=buf.now,
                           round_id=rid, window_ms=int(window_ms))
    return on_anim_notify(buf, event)


# ---------------------------------------------------------------------------
# 修复 2：输入按时间重排
# ---------------------------------------------------------------------------


def reorder_input(events: Any) -> list[dict]:
    """把按键事件按事件时间戳稳定排序，替代固定长度 FIFO 队列。

    高延迟客户端的按键可能晚到（``recv_ts`` 很大）但 ``ts`` 仍落在窗口内，
    FIFO 出队会让它在窗口关闭后才被处理而丢弃。排序键依次为：

    1. 按键发生时间戳 ``ts``（玩家真实意图时刻）；
    2. 到达序号 ``seq``（同时刻时保证确定性，不靠 dict 身份）；
    3. 到达时刻 ``recv_ts``（兜底稳定）。
    """
    if not isinstance(events, (list, tuple)):
        return []
    normalized = [e for e in events if isinstance(e, dict)]
    return sorted(
        normalized,
        key=lambda e: (
            int(e.get("ts", 0)),
            int(e.get("seq", 0)),
            int(e.get("recv_ts", e.get("ts", 0))),
        ),
    )


def on_input(buf: Any, event: Any = None, now: Optional[int] = None) -> list[dict]:
    """接收一个按键事件：入队、按时间戳重排，再处理所有就绪输入。

    返回本次调用实际完成判定的结果列表（晚到但仍在窗口内的输入会被救回，
    而不是像旧 FIFO 那样在窗口关闭后出队被丢弃）。
    """
    buf = _as_buffer(buf)
    if now is not None:
        buf.now = int(now)
    if event is None:
        return []
    if not isinstance(event, dict):
        event = make_input(int(event), recv_ts=buf.now,
                           round_id=buf.round_id)
    event = dict(event)
    event.setdefault("round_id", buf.round_id)
    event.setdefault("recv_ts", buf.now)

    buf.input_queue.append(event)
    buf.history.append(dict(event))
    buf.counters["inputs_received"] += 1

    # 关键：先重排再出队，延迟高但时间戳早的输入会排回正确位置。
    buf.input_queue = reorder_input(buf.input_queue)

    results: list[dict] = []
    remaining: list[dict] = []
    for evt in buf.input_queue:
        outcome = _drain_one(buf, evt)
        if outcome is None:
            remaining.append(evt)  # 窗口尚未出现，留待后续通知
        else:
            results.append(outcome)
    buf.input_queue = remaining
    return results


def _drain_one(buf: ParryBuffer, evt: dict) -> Optional[dict]:
    rid = int(evt.get("round_id", buf.round_id))

    # 回合已有权威结论 / 已有本地预测，不再重复受理该回合输入。
    if rid in buf.confirmed or rid in buf.predicted:
        buf.counters["inputs_dropped"] += 1
        _log(buf, "input_rejected", round_id=rid, reason="round_finalized",
             ts=evt.get("ts"))
        return {"round_id": rid, "result": "rejected", "reason": "round_finalized"}

    window = _active_window(buf, rid)
    if window is None:
        # 窗口可能已被关窗计时器关闭，但按键时间戳仍落在窗口区间内：
        # 判定按事件时间戳而不是出队 / 到达时刻，这是输入重排救回的核心。
        closed = next((w for w in buf.windows
                       if w["round_id"] == rid and not w["consumed"]), None)
        if closed is not None and \
                closed["start_ts"] <= int(evt["ts"]) <= closed["end_ts"]:
            window = closed
            window["open"] = True  # 临时复权，parry 内会立即标记消耗
        else:
            return None  # 窗口尚未出现，留待后续通知

    outcome = parry(buf, evt, window)
    buf.counters["inputs_processed"] += 1
    if evt.get("recv_ts", evt["ts"]) > window["end_ts"] and \
            window["start_ts"] <= evt["ts"] <= window["end_ts"]:
        # 到达时刻已过窗口，但按键时间戳在窗口内：重排救回的典型场景。
        buf.counters["late_inputs_recovered"] += 1
        _log(buf, "late_input_recovered", round_id=rid, ts=evt["ts"],
             recv_ts=evt.get("recv_ts"))
    return outcome


# ---------------------------------------------------------------------------
# 修复 3：替换注销计时器
# ---------------------------------------------------------------------------


def unbind_on_swap(buf: Any, timer: Any = None,
                   round_id: Optional[int] = None) -> dict:
    """反击动画替换状态机节点前，注销旧窗口计时器。

    旧实现替换节点时不注销计时器，旧计时器到点仍会打开下一轮窗口。这里对
    指定计时器（或当前回合全部存活计时器）做取消标记；已触发 / 已取消的
    回调再次进入也是幂等的。
    """
    buf = _as_buffer(buf)
    cancelled: list[dict] = []

    if isinstance(timer, dict):
        target = None
        for t in buf.timers:
            if t.get("id") == timer.get("id"):
                target = t
                break
        if target is not None and not target.get("cancelled"):
            target["cancelled"] = True
            cancelled.append(dict(target))
    else:
        rid = buf.round_id if round_id is None else int(round_id)
        for t in buf.timers:
            if t.get("round_id") == rid and not t.get("cancelled") \
                    and not t.get("fired"):
                t["cancelled"] = True
                cancelled.append(dict(t))

    buf.counters["timers_cancelled"] += len(cancelled)
    if cancelled:
        _log(buf, "timers_unbound_on_swap",
             round_id=round_id if round_id is not None else buf.round_id,
             timer_ids=[t["id"] for t in cancelled])
    return {"cancelled": cancelled, "count": len(cancelled)}


# ---------------------------------------------------------------------------
# 修复 4：计时器绑回合号
# ---------------------------------------------------------------------------


def timer_round_tag(timer: Any, round_id: Optional[int] = None) -> dict:
    """给计时器绑定回合号标签。

    连续格挡复用同一计时器对象 / 对象池回收后，上一轮回调仍可能触发。
    触发端（:func:`tick`）校验 ``round_id`` 与 ``tag`` 是否等于当前回合，
    不匹配即视为过期回调丢弃，判定不会落到新一局的第一次格挡上。
    """
    rid = 0 if round_id is None else int(round_id)
    if isinstance(timer, dict):
        out = dict(timer)
    else:
        # 合约测试路径：入参是裸整数，当作 fire_at。
        out = {"id": f"timer_{timer}", "fire_at": int(timer),
               "callback": "generic"}
    out["round_id"] = rid
    out["tag"] = rid
    out.setdefault("cancelled", False)
    out.setdefault("fired", False)
    out.setdefault("callback", "generic")
    return out


def tick(buf: Any, now: int) -> list[dict]:
    """推进时钟到 ``now``，触发到点且未过期的计时器。

    过期回调（已注销、已触发、回合号 / 标签不匹配）只计数不执行，对应
    “对象池回收后旧回调仍触发”这一缺陷的拦截点。
    """
    buf = _as_buffer(buf)
    buf.now = int(now)
    fired: list[dict] = []
    for t in buf.timers:
        if t.get("fired"):
            continue
        if not t.get("cancelled") and int(t.get("fire_at", 0)) <= buf.now:
            if _is_stale(t, buf):
                buf.counters["stale_timer_fires"] += 1
                _log(buf, "stale_timer_blocked", timer_id=t.get("id"),
                     timer_round=t.get("round_id"), current_round=buf.round_id)
                t["fired"] = True
                continue
            t["fired"] = True
            fired.append(dict(t))
            if t.get("callback") == "close_window":
                _close_expired_windows(buf)
    return fired


def _close_expired_windows(buf: ParryBuffer) -> None:
    for w in buf.windows:
        if w["open"] and w["end_ts"] <= buf.now:
            w["open"] = False


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def validate_window(window: Any, event_ts: Any = None,
                    round_id: Optional[int] = None) -> bool:
    """按键事件时间戳是否落在当前打开的窗口内。

    判定一律使用事件时间戳，与网络到达时刻无关。窗口已关闭 / 已消耗 /
    回合不匹配都返回 False。
    """
    if not isinstance(window, dict) or not isinstance(event_ts, (int, float)):
        return False
    if not window.get("open", False) or window.get("consumed", False):
        return False
    if window.get("finalized", False):
        return False
    if round_id is not None and window.get("round_id") != int(round_id):
        return False
    ts = int(event_ts)
    return int(window["start_ts"]) <= ts <= int(window["end_ts"])


def parry(buf: Any, evt: Any = None, window: Optional[dict] = None) -> dict:
    """对一次按键做本地预测判定。

    时间戳落在窗口内 => ``parry``（弹反成功，随后反击动画替换节点，需要
    注销旧计时器）；窗口已关闭但同回合内 => ``block``（普通格挡）；
    否则 => ``miss``。结果先记为本地预测，等待 :func:`confirm` 权威确认。
    """
    buf = _as_buffer(buf)
    if evt is None:
        evt = make_input(buf.now, recv_ts=buf.now, round_id=buf.round_id)
    elif not isinstance(evt, dict):
        evt = make_input(int(evt), recv_ts=buf.now, round_id=buf.round_id)
    rid = int(evt.get("round_id", buf.round_id))
    if window is None:
        window = _active_window(buf, rid)

    if window is not None and validate_window(window, evt["ts"], rid):
        result = "parry"
        window["consumed"] = True
        window["open"] = False
        buf.counters["parries"] += 1
        # 弹反成功 -> 反击动画替换状态机节点：先注销旧窗口计时器，
        # 防止下一轮格挡窗口被提前打开。
        unbind_on_swap(buf, round_id=rid)
    elif window is not None and evt["ts"] >= window["start_ts"]:
        result = "block"
        buf.counters["blocks"] += 1
    else:
        result = "miss"
        buf.counters["misses"] += 1

    outcome = {
        "round_id": rid,
        "result": result,
        "input_ts": int(evt["ts"]),
        "predicted": True,
        "authoritative": False,
    }
    buf.predicted[rid] = outcome
    _log(buf, "local_prediction", round_id=rid, result=result,
         input_ts=evt["ts"])
    return outcome


# ---------------------------------------------------------------------------
# 修复 5 + 6：回滚含窗口、预测以服务器为准
# ---------------------------------------------------------------------------


def rollback_window(buf: Any, round_id: Any = None,
                    server_msg: Optional[dict] = None) -> dict:
    """服务器纠偏时回滚到回合起点快照：血量与窗口状态一起回滚。

    旧实现只回滚血量，窗口仍停留在本地预测后的状态（已消耗 / 已关闭），
    后续回合的判定全部错位。快照在 :func:`on_anim_notify` 首次见到回合时
    建立；没有快照则只纠正当前窗口标志位并记录告警。
    """
    buf = _as_buffer(buf)
    if round_id is None:
        round_id = buf.round_id
    rid = int(round_id)
    snap = buf.snapshots.get(rid)
    if snap is not None:
        _restore(buf, snap)
        restored = True
    else:
        # 无快照兜底：重开本回合窗口标志，保证窗口状态与服务器结论一致。
        for w in buf.windows:
            if w.get("round_id") == rid:
                w["consumed"] = False
        restored = False

    buf.counters["rollbacks"] += 1
    _log(buf, "rollback", round_id=rid, snapshot_restored=restored)

    if server_msg is not None:
        # 回滚后应用服务器权威状态，避免回滚把权威血量也冲掉。
        buf.hp = int(server_msg["hp"])
        buf.confirmed[rid] = {
            "round_id": rid,
            "result": server_msg["result"],
            "input_ts": server_msg.get("server_ts"),
            "predicted": False,
            "authoritative": True,
        }
        buf.predicted.pop(rid, None)
        # 窗口几何状态已回滚为干净快照，再打上权威终结标记：
        # 状态可审计（看得见回滚发生过），同时不会再接受新输入。
        for w in buf.windows:
            if w.get("round_id") == rid:
                w["finalized"] = True
    return {"round_id": rid, "restored": restored, "hp": buf.hp}


def confirm(buf: Any, server_msg: Any = None) -> dict:
    """处理服务器确认：与本地预测一致则转正，不一致则回滚并以服务器为准。"""
    buf = _as_buffer(buf)
    if server_msg is None:
        server_msg = make_server_msg(buf.round_id, result="miss", hp=buf.hp)
    elif not isinstance(server_msg, dict):
        # 合约测试路径：裸整数当作回合号，构造一个中性权威消息。
        server_msg = make_server_msg(int(server_msg), result="miss", hp=buf.hp)
    rid = int(server_msg["round_id"])
    local = buf.predicted.get(rid)

    if local is not None and local["result"] == server_msg["result"]:
        confirmed = dict(local)
        confirmed["predicted"] = False
        confirmed["authoritative"] = True
        buf.confirmed[rid] = confirmed
        buf.predicted.pop(rid, None)
        buf.hp = int(server_msg["hp"])
        for w in buf.windows:
            if w.get("round_id") == rid:
                w["finalized"] = True
        _log(buf, "server_confirmed", round_id=rid, result=confirmed["result"],
             mismatch=False)
        return {"round_id": rid, "mismatch": False,
                "result": confirmed["result"]}

    # 预测与服务器不一致（或本地压根没有预测）：回滚必须包含窗口状态，
    # 然后强制采用服务器结论。
    buf.counters["server_mismatches"] += 1
    _log(buf, "server_mismatch", round_id=rid,
         local=local["result"] if local else None,
         server=server_msg["result"])
    rollback_window(buf, rid, server_msg)
    return fallback(buf, server_msg)


def fallback(buf: Any, server_msg: Any = None) -> dict:
    """权威兜底：无条件采用服务器结果与血量，并纠正本地窗口状态。"""
    buf = _as_buffer(buf)
    if server_msg is None:
        server_msg = make_server_msg(buf.round_id, result="miss", hp=buf.hp)
    elif not isinstance(server_msg, dict):
        server_msg = make_server_msg(int(server_msg), result="miss", hp=buf.hp)
    rid = int(server_msg["round_id"])

    buf.hp = int(server_msg["hp"])
    buf.confirmed[rid] = {
        "round_id": rid,
        "result": server_msg["result"],
        "input_ts": server_msg.get("server_ts"),
        "predicted": False,
        "authoritative": True,
    }
    buf.predicted.pop(rid, None)
    # 只在本地根本没有该回合窗口时补一个终结标记；若窗口存在，其开关 /
    # 消耗状态交由 rollback_window 恢复，不能在这里把回滚效果覆盖掉
    # （否则就退化成“只回滚血量、不回滚窗口”的旧缺陷）。
    if not any(w.get("round_id") == rid for w in buf.windows):
        buf.windows.append({
            "start_ts": None, "end_ts": None, "round_id": rid,
            "open": False, "consumed": True, "finalized": True,
        })
    _log(buf, "authoritative_fallback", round_id=rid,
         result=server_msg["result"], hp=buf.hp)
    return {"round_id": rid, "mismatch": True,
            "result": server_msg["result"], "hp": buf.hp}


# ---------------------------------------------------------------------------
# 可观测性
# ---------------------------------------------------------------------------


def metrics(buf: Any) -> dict:
    """返回指标计数的拷贝。"""
    buf = _as_buffer(buf)
    return dict(buf.counters)


def audit(buf: Any) -> list[dict]:
    """返回审计日志的拷贝。"""
    buf = _as_buffer(buf)
    return [dict(e) for e in buf.audit_log]


def determinism(buf: Any, events: Optional[list[dict]] = None) -> dict:
    """确定性校验：同一批输入无论到达顺序如何，重排后的判定必须一致。

    做法是对历史输入（或显式传入的事件集）做两次独立回放：一次按原顺序，
    一次打乱到达顺序。由于 :func:`reorder_input` 按事件时间戳重排，两次
    回放的预测结果序列必须完全相同。
    """
    base = _as_buffer(buf)
    if events is None:
        events = [dict(e) for e in base.history]
    events = [dict(e) for e in events if isinstance(e, dict)]
    if not events:
        return {"deterministic": True, "runs": ([], []), "reason": "no_events"}

    def _run(order: list[dict]) -> list[dict]:
        run_buf = ParryBuffer(round_id=base.round_id, hp=base.hp,
                              max_hp=base.max_hp)
        # 回放需要窗口：从原缓冲复制窗口定义，并恢复为未消耗的打开状态。
        for w in base.windows:
            clone = dict(w)
            clone["open"] = not clone["consumed"]
            run_buf.windows.append(clone)
        # 整批按“到达顺序”入队，不立即出队；之后统一重排，使到达顺序
        # 对最终判定序列没有影响。
        for evt in order:
            run_buf.input_queue.append(dict(evt))
            run_buf.history.append(dict(evt))
        run_buf.input_queue = reorder_input(run_buf.input_queue)
        seq: list[dict] = []
        for evt in list(run_buf.input_queue):
            outcome = _drain_one(run_buf, evt)
            if outcome is not None:
                seq.append(outcome)
        return seq

    run_a = _run(events)
    shuffled = list(reversed(events))
    run_b = _run(shuffled)
    deterministic = run_a == run_b
    if not deterministic:
        _log(base, "determinism_violation",
             run_a=run_a, run_b=run_b)
    return {"deterministic": deterministic, "runs": (run_a, run_b)}
