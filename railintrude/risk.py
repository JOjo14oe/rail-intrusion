"""四级报警与降级滞回。内部枚举英文，中文映射只在这里定义一次。"""

from __future__ import annotations

LEVELS = ("notice", "alert", "warn", "emerg")
LEVEL_ZH = {"notice": "注意", "alert": "提示", "warn": "警告", "emerg": "紧急"}
_SEVERITY = {name: i for i, name in enumerate(LEVELS)}

# TTC 阈值（秒），从宽到严
T_EMERG = 3.0
T_WARN = 5.0
T_ALERT = 8.0


def level_from_ttc(ttc_s: float) -> str:
    if ttc_s <= T_EMERG:
        return "emerg"
    if ttc_s <= T_WARN:
        return "warn"
    if ttc_s <= T_ALERT:
        return "alert"
    return "notice"


def severity(level: str) -> int:
    return _SEVERITY[level]


class Hysteresis:
    """升级立即生效；降级需连续 down_frames 帧低等级。"""

    def __init__(self, down_frames: int = 5):
        if down_frames < 1:
            raise ValueError("down_frames 至少为 1")
        self.down_frames = down_frames
        self._current = "notice"
        self._below = 0

    @property
    def current(self) -> str:
        return self._current

    def update(self, new_level: str) -> str:
        if severity(new_level) >= severity(self._current):
            self._current = new_level
            self._below = 0
            return self._current
        self._below += 1
        if self._below >= self.down_frames:
            self._current = new_level
            self._below = 0
        return self._current
