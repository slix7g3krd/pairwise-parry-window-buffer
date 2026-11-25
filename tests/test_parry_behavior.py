"""六个修复点的行为级测试。

每个测试对应题目中的一个缺陷，命名中的 fixN 与模块 docstring 中的编号一致：

1. 窗口按事件时间戳（动画通知晚帧不移动判定窗口）
2. 输入按时间重排（高延迟按键不被 FIFO 队列丢弃）
3. 替换注销计时器（反击动画替换节点不提前打开下一轮窗口）
4. 计时器绑回合号（对象池回收后的旧回调落空）
5. 回滚含窗口（服务器回滚同时恢复血量与窗口状态）
6. 预测以服务器为准（不一致时强制采用权威结论）
"""

from __future__ import annotations

import gameplay.parry_window_buffer as mod
from gameplay.parry_window_buffer import (
    ParryBuffer,
    make_event,
    make_input,
    make_server_msg,
)


# ---------------------------------------------------------------------------
# 修复 1：窗口按事件时间戳
# ---------------------------------------------------------------------------


def test_fix1_window_anchored_on_event_timestamp_not_callback_time():
    # 动画时间轴上通知发生在 100ms，但异步蓝图让回调晚一帧在 116ms 才执行。
    event = make_event(event_ts=100, fire_ts=116, round_id=0, window_ms=120)
    window = mod.window_by_event_ts(event)

    assert window["start_ts"] == 100
    assert window["end_ts"] == 220
    assert window["round_id"] == 0

    buf = ParryBuffer()
    mod.on_anim_notify(buf, event, now=116)
    only = buf.windows[0]
    assert only["start_ts"] == 100 and only["end_ts"] == 220

    # 110ms 的按键在玩家看到的格挡帧内：按事件时间戳应判弹反成功，
    # 若错误地按回调时刻 116ms 开窗，该按键会落到窗口外变成 miss。
    result = mod.on_input(buf, make_input(ts=110, recv_ts=118, round_id=0))
    assert [r["result"] for r in result] == ["parry"]


def test_fix1_late_notify_does_not_shift_window():
    event = make_event(event_ts=100, fire_ts=200, round_id=1, window_ms=50)
    buf = ParryBuffer(round_id=1)
    mod.on_anim_notify(buf, event, now=200)
    window = buf.windows[0]

    # 窗口 [100, 150]，与 200ms 的回调时刻完全无关。
    assert mod.validate_window(window, 150, round_id=1) is True
    assert mod.validate_window(window, 151, round_id=1) is False
    assert mod.validate_window(window, 99, round_id=1) is False


# ---------------------------------------------------------------------------
# 修复 2：输入按时间重排
# ---------------------------------------------------------------------------


def test_fix2_reorder_sorts_by_event_timestamp_with_stable_tiebreak():
    # 到达顺序是乱序的（高延迟客户端先按的键后到）。
    events = [
        make_input(ts=120, recv_ts=300, seq=1),
        make_input(ts=105, recv_ts=110, seq=0),
        make_input(ts=110, recv_ts=112, seq=2),
    ]
    ordered = mod.reorder_input(events)
    assert [e["ts"] for e in ordered] == [105, 110, 120]

    # 同时间戳时按到达序号 seq 打破平局，保证确定性。
    tied = [make_input(ts=100, seq=9), make_input(ts=100, seq=3)]
    assert [e["seq"] for e in mod.reorder_input(tied)] == [3, 9]


def test_fix2_late_packet_inside_window_is_recovered_not_dropped():
    buf = ParryBuffer()
    mod.on_anim_notify(
        buf, make_event(event_ts=100, fire_ts=100, round_id=0, window_ms=100),
        now=100,
    )
    # 关窗计时器到点，窗口已关闭；按键发生在 150ms（窗口内），
    # 但网络延迟让它在 250ms 才到达。旧 FIFO 实现此刻出队只会丢弃。
    mod.tick(buf, 210)
    assert buf.windows[0]["open"] is False

    results = mod.on_input(
        buf, make_input(ts=150, recv_ts=250, round_id=0), now=250,
    )
    assert len(results) == 1
    assert results[0]["result"] == "parry"
    assert mod.metrics(buf)["late_inputs_recovered"] == 1
    assert mod.metrics(buf)["inputs_dropped"] == 0


def test_fix2_out_of_order_arrivals_produce_timestamp_order_judgement():
    buf = ParryBuffer()
    mod.on_anim_notify(
        buf, make_event(event_ts=100, fire_ts=100, round_id=0, window_ms=100),
        now=100,
    )
    # 两个输入都已到达并缓存在队列中（到达顺序与按下顺序相反）：
    # 高延迟的 110ms 按键比 180ms 按键更晚到达。
    buf.input_queue.append(make_input(ts=180, recv_ts=181, seq=1))
    buf.input_queue.append(make_input(ts=110, recv_ts=182, seq=0))

    # 出队前统一按事件时间戳重排：110ms 的按键先判定，弹反成功并消耗窗口；
    # 旧 FIFO 实现会让 180ms 先出队，110ms 的真实格挡意图反而落空。
    buf.input_queue = mod.reorder_input(buf.input_queue)
    first = mod.parry(buf, buf.input_queue[0])
    second = mod.parry(buf, buf.input_queue[1])

    assert first["result"] == "parry"
    assert first["input_ts"] == 110
    assert second["result"] != "parry"  # 窗口已被更早的弹反消耗
    assert mod.metrics(buf)["parries"] == 1


# ---------------------------------------------------------------------------
# 修复 3：替换注销计时器
# ---------------------------------------------------------------------------


def test_fix3_swap_unbinds_old_window_timer():
    buf = ParryBuffer()
    mod.on_anim_notify(
        buf, make_event(event_ts=100, fire_ts=100, round_id=0, window_ms=100),
        now=100,
    )
    # 弹反成功触发反击动画替换状态机节点，内部必须注销旧关窗计时器。
    mod.on_input(buf, make_input(ts=120, recv_ts=121, round_id=0), now=121)
    assert mod.metrics(buf)["timers_cancelled"] >= 1
    assert all(t["cancelled"] for t in buf.timers)

    # 旧计时器到点不会再触发任何回调，下一轮窗口不会被它提前打开。
    fired = mod.tick(buf, 1000)
    assert fired == []
    assert buf.windows[0]["open"] is False  # 窗口是被弹反消耗关闭的


def test_fix3_manual_unbind_is_idempotent_and_scoped():
    buf = ParryBuffer()
    mod.on_anim_notify(
        buf, make_event(event_ts=0, fire_ts=0, round_id=0, window_ms=100),
        now=0,
    )
    timer = buf.timers[0]
    first = mod.unbind_on_swap(buf, timer)
    second = mod.unbind_on_swap(buf, timer)
    assert first["count"] == 1
    assert second["count"] == 0  # 重复注销不重复计数


# ---------------------------------------------------------------------------
# 修复 4：计时器绑回合号
# ---------------------------------------------------------------------------


def test_fix4_stale_timer_from_recycled_object_is_blocked():
    buf = ParryBuffer()
    mod.on_anim_notify(
        buf, make_event(event_ts=0, fire_ts=0, round_id=0, window_ms=50),
        now=0,
    )
    old_timer = buf.timers[0]

    # 模拟对象池复用：同一个计时器对象被新一轮“捡起来”，
    # 但新逻辑要求计时器必须带当前回合标签，旧标签（round 0）已过期。
    buf.round_id = 1
    mod.tick(buf, 100)  # 旧计时器到点

    assert mod.metrics(buf)["stale_timer_fires"] == 1
    assert old_timer["fired"] is True
    # 过期回调没有产生任何窗口动作：新回合尚未开窗。
    assert all(w["round_id"] != 1 for w in buf.windows)


def test_fix4_tagged_timer_fires_only_in_its_own_round():
    tagged = mod.timer_round_tag(
        {"id": "t", "fire_at": 50, "callback": "close_window"}, round_id=3,
    )
    assert tagged["round_id"] == 3 and tagged["tag"] == 3

    buf = ParryBuffer(round_id=3)
    buf.timers.append(dict(tagged))
    assert len(mod.tick(buf, 50)) == 1  # 同回合正常触发

    # 再次推进不会重复触发。
    assert mod.tick(buf, 60) == []


# ---------------------------------------------------------------------------
# 修复 5：回滚含窗口
# ---------------------------------------------------------------------------


def test_fix5_rollback_restores_hp_and_window_state():
    buf = ParryBuffer(hp=100)
    mod.on_anim_notify(
        buf, make_event(event_ts=100, fire_ts=100, round_id=0, window_ms=100),
        now=100,
    )
    # 本地预测：弹反成功，窗口被消耗关闭，血量先按本地逻辑变化。
    mod.on_input(buf, make_input(ts=120, recv_ts=121, round_id=0), now=121)
    buf.hp = 90
    assert buf.windows[0]["consumed"] is True

    # 服务器判定与本地不一致，回滚必须同时恢复血量与窗口。
    mod.rollback_window(buf, 0, make_server_msg(0, "block", hp=80, server_ts=120))

    assert buf.hp == 80  # 服务器权威血量
    assert mod.metrics(buf)["rollbacks"] == 1
    window = next(w for w in buf.windows if w["round_id"] == 0)
    assert window["consumed"] is False  # 窗口几何状态也被回滚，而非只回滚血量
    assert window["open"] is True
    assert window["finalized"] is True  # 但权威结论已终结该回合，不再吃输入
    assert mod.validate_window(window, 150, round_id=0) is False
    assert 0 in buf.confirmed and 0 not in buf.predicted


# ---------------------------------------------------------------------------
# 修复 6：预测以服务器为准
# ---------------------------------------------------------------------------


def test_fix6_confirm_agreement_promotes_local_prediction():
    buf = ParryBuffer(hp=100)
    mod.on_anim_notify(
        buf, make_event(event_ts=100, fire_ts=100, round_id=0, window_ms=100),
        now=100,
    )
    mod.on_input(buf, make_input(ts=120, recv_ts=121, round_id=0), now=121)
    assert buf.predicted[0]["result"] == "parry"

    out = mod.confirm(buf, make_server_msg(0, "parry", hp=100))
    assert out["mismatch"] is False
    assert buf.confirmed[0]["authoritative"] is True
    assert 0 not in buf.predicted
    assert mod.metrics(buf)["server_mismatches"] == 0


def test_fix6_mismatch_forces_server_authority():
    buf = ParryBuffer(hp=100)
    mod.on_anim_notify(
        buf, make_event(event_ts=100, fire_ts=100, round_id=0, window_ms=100),
        now=100,
    )
    mod.on_input(buf, make_input(ts=120, recv_ts=121, round_id=0), now=121)
    assert buf.predicted[0]["result"] == "parry"

    # 服务器说这一刀其实没弹反掉。
    out = mod.confirm(buf, make_server_msg(0, "miss", hp=70))
    assert out["mismatch"] is True
    assert out["result"] == "miss"
    assert buf.hp == 70
    assert buf.confirmed[0]["result"] == "miss"
    assert buf.confirmed[0]["authoritative"] is True
    assert 0 not in buf.predicted
    assert mod.metrics(buf)["server_mismatches"] == 1
    # 回滚恢复了窗口几何状态，同时权威终结标记阻止后续输入再进该窗口。
    window = next(w for w in buf.windows if w["round_id"] == 0)
    assert window["consumed"] is False and window["finalized"] is True
    # 审计链路上能同时看到分歧与权威兜底。
    actions = [e["action"] for e in mod.audit(buf)]
    assert "server_mismatch" in actions
    assert "authoritative_fallback" in actions


def test_fix6_fallback_overwrites_local_state_unconditionally():
    buf = ParryBuffer(hp=100)
    out = mod.fallback(buf, make_server_msg(2, "block", hp=55))
    assert out["result"] == "block" and buf.hp == 55
    assert buf.confirmed[2]["authoritative"] is True


# ---------------------------------------------------------------------------
# 可观测性与确定性
# ---------------------------------------------------------------------------


def test_metrics_and_audit_are_copies():
    buf = ParryBuffer()
    m1 = mod.metrics(buf)
    m1["parries"] = 999
    assert mod.metrics(buf)["parries"] == 0

    log1 = mod.audit(buf)
    log1.append({"fake": True})
    assert mod.audit(buf) == []


def test_determinism_holds_under_shuffled_arrival_order():
    buf = ParryBuffer()
    mod.on_anim_notify(
        buf, make_event(event_ts=100, fire_ts=100, round_id=0, window_ms=100),
        now=100,
    )
    events = [
        make_input(ts=150, recv_ts=151, seq=2),
        make_input(ts=110, recv_ts=111, seq=0),
        make_input(ts=130, recv_ts=131, seq=1),
    ]
    report = mod.determinism(buf, events)
    assert report["deterministic"] is True
    # 两种到达顺序下，第一个判定都是时间戳最早的 110ms 弹反。
    assert report["runs"][0][0]["result"] == "parry"
    assert report["runs"][1][0]["input_ts"] == 110
