# pairwise-parry-window-buffer

## 领域

游戏、图形与多媒体

## 主题

格挡与弹反窗口

## 说明

事件时间戳、输入重排与回合绑定。

初始状态下 `gameplay/parry_window_buffer.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
