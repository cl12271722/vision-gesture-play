from __future__ import annotations

import math
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence


class LandmarkLike(Protocol):
    """手部关键点需要提供的最小坐标接口。"""

    x: float
    y: float


class Direction(Enum):
    """手指可以指向的四个方向。"""

    UP = ("上", 0, -1, 90)
    DOWN = ("下", 0, 1, -90)
    LEFT = ("左", -1, 0, 180)
    RIGHT = ("右", 1, 0, 0)

    def __init__(
        self,
        label: str,
        dx: int,
        dy: int,
        rotation: int,
    ) -> None:
        self.label = label
        self.dx = dx
        self.dy = dy
        self.rotation = rotation


def _distance(first: LandmarkLike, second: LandmarkLike) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def classify_index_direction(
    landmarks: Sequence[LandmarkLike],
    frame_width: int,
    frame_height: int,
) -> Direction | None:
    """根据食指从掌指关节到指尖的向量判断指向。"""

    if len(landmarks) < 21:
        return None

    wrist = landmarks[0]
    index_mcp = landmarks[5]
    index_pip = landmarks[6]
    index_tip = landmarks[8]

    # 食指未明显伸直时不产生答案，避免握拳或 OK 手势被误判
    if _distance(index_tip, wrist) <= _distance(index_pip, wrist) * 1.08:
        return None

    dx = (index_tip.x - index_mcp.x) * frame_width
    dy = (index_tip.y - index_mcp.y) * frame_height
    length = math.hypot(dx, dy)
    if length < min(frame_width, frame_height) * 0.08:
        return None

    # 将向量角度划分为右、下、左、上四个 90 度区间
    angle = math.degrees(math.atan2(dy, dx))
    if -45 <= angle < 45:
        return Direction.RIGHT
    if 45 <= angle < 135:
        return Direction.DOWN
    if angle >= 135 or angle < -135:
        return Direction.LEFT
    return Direction.UP


def detect_ok_gesture(
    landmarks: Sequence[LandmarkLike],
    frame_width: int,
    frame_height: int,
) -> bool:
    """识别拇指与食指相接、其余三指伸开的 OK 手势。"""

    if len(landmarks) < 21:
        return False

    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = max(_distance(wrist, middle_mcp), 1e-6)

    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    thumb_index_ratio = _distance(thumb_tip, index_tip) / palm_size
    if thumb_index_ratio > 0.42:
        return False

    extended_pairs = ((12, 10), (16, 14), (20, 18))
    for tip_index, pip_index in extended_pairs:
        tip = landmarks[tip_index]
        pip = landmarks[pip_index]
        if _distance(tip, wrist) <= _distance(pip, wrist) * 1.08:
            return False

    # 使用帧尺寸参数保留接口一致性，同时避免不同宽高比导致无意误判
    if frame_width <= 0 or frame_height <= 0:
        return False
    return True


class GestureStabilizer:
    """用连续多帧投票抑制方向识别抖动。"""

    def __init__(
        self,
        window_size: int = 6,
        min_votes: int = 4,
        release_frames: int = 8,
    ) -> None:
        self._history: deque[Direction] = deque(maxlen=window_size)
        self._min_votes = min_votes
        self._release_frames = release_frames
        self._none_count = 0
        self.current: Direction | None = None

    def reset(self) -> None:
        self._history.clear()
        self._none_count = 0
        self.current = None

    def update(self, direction: Direction | None) -> Direction | None:
        if direction is None:
            self._none_count += 1
            if self._none_count >= self._release_frames:
                self.reset()
            return self.current

        self._none_count = 0
        self._history.append(direction)
        direction_name, votes = Counter(self._history).most_common(1)[0]
        if votes >= self._min_votes:
            self.current = direction_name
        return self.current


class BooleanStabilizer:
    """用连续多帧投票稳定 OK 手势状态。"""

    def __init__(
        self,
        window_size: int = 7,
        on_votes: int = 5,
        off_votes: int = 2,
    ) -> None:
        self._history: deque[bool] = deque(maxlen=window_size)
        self._on_votes = on_votes
        self._off_votes = off_votes
        self.current = False

    def update(self, value: bool) -> bool:
        self._history.append(value)
        true_count = sum(self._history)
        if true_count >= self._on_votes:
            self.current = True
        elif true_count <= self._off_votes:
            self.current = False
        return self.current


class RoundStatus(Enum):
    FALLING = "下落中"
    JUDGING = "判定中"
    RESOLVED = "已判定"


class RoundOutcome(Enum):
    CORRECT = "正确"
    WRONG = "错误"
    MISS = "未作答"


@dataclass(frozen=True)
class QuestionConfig:
    font_size: int
    speed: float
    judge_window: float


def question_config(question_number: int) -> QuestionConfig:
    """让每题的字号、下落速度和判定时间逐步变化。"""

    index = max(0, question_number - 1)
    return QuestionConfig(
        font_size=max(50, 94 - index * 2),
        speed=145.0 + index * 5.0,
        judge_window=max(0.72, 1.18 - index * 0.02),
    )


@dataclass
class FallingRound:
    question_number: int
    direction: Direction
    font_size: int
    speed: float
    judge_window: float
    x_ratio: float
    y: float = 175.0
    status: RoundStatus = RoundStatus.FALLING
    judge_elapsed: float = 0.0
    user_answer: Direction | None = None
    outcome: RoundOutcome | None = None

    def update(self, dt: float, judge_y: float) -> None:
        if self.status is RoundStatus.FALLING:
            self.y += self.speed * dt
            if self.y >= judge_y:
                self.y = judge_y
                self.status = RoundStatus.JUDGING
            return

        if self.status is RoundStatus.JUDGING:
            self.judge_elapsed += dt
            if self.judge_elapsed >= self.judge_window:
                self.outcome = RoundOutcome.MISS
                self.status = RoundStatus.RESOLVED

    def submit(self, answer: Direction) -> RoundOutcome | None:
        if self.status is not RoundStatus.JUDGING:
            return None

        self.user_answer = answer
        self.outcome = (
            RoundOutcome.CORRECT
            if answer is self.direction
            else RoundOutcome.WRONG
        )
        self.status = RoundStatus.RESOLVED
        return self.outcome

    @property
    def response_time_ms(self) -> int:
        return max(0, int(self.judge_elapsed * 1000))

    @property
    def judge_progress(self) -> float:
        if self.judge_window <= 0:
            return 1.0
        return min(1.0, self.judge_elapsed / self.judge_window)


@dataclass(frozen=True)
class AttemptRecord:
    question_number: int
    direction: Direction
    font_size: int
    user_answer: Direction | None
    outcome: RoundOutcome
    response_time_ms: int
    score_after: int


@dataclass(frozen=True)
class RoundEvent:
    question_number: int
    direction: Direction
    user_answer: Direction | None
    outcome: RoundOutcome
    points: int
    combo: int
    score: int
    accuracy: float
    is_last_question: bool


@dataclass
class GameSession:
    total_questions: int = 20
    rng: random.Random = field(default_factory=random.Random)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.score = 0
        self.combo = 0
        self.best_combo = 0
        self.correct_count = 0
        self.rounds: list[AttemptRecord] = []
        self.current_round: FallingRound | None = None
        self.last_event: RoundEvent | None = None
        self.inter_round_delay = 0.8
        self._round_finalized = False
        self._last_direction: Direction | None = None

    @property
    def question_number(self) -> int:
        if self.complete:
            return self.total_questions
        return min(self.total_questions, len(self.rounds) + 1)

    @property
    def accuracy(self) -> float:
        if not self.rounds:
            return 0.0
        return self.correct_count / len(self.rounds)

    @property
    def complete(self) -> bool:
        return len(self.rounds) >= self.total_questions

    @property
    def wrong_count(self) -> int:
        return sum(
            record.outcome is RoundOutcome.WRONG for record in self.rounds
        )

    @property
    def miss_count(self) -> int:
        return sum(
            record.outcome is RoundOutcome.MISS for record in self.rounds
        )

    def update(
        self,
        dt: float,
        judge_y: float,
        gesture: Direction | None = None,
    ) -> RoundEvent | None:
        if self.complete:
            return None

        if self.current_round is None:
            self.inter_round_delay -= dt
            if self.inter_round_delay <= 0:
                self._spawn_round()
            return None

        self.current_round.update(dt, judge_y)

        if (
            gesture is not None
            and self.current_round.status is RoundStatus.JUDGING
        ):
            self.current_round.submit(gesture)

        event = None
        if (
            self.current_round.status is RoundStatus.RESOLVED
            and not self._round_finalized
        ):
            event = self._finalize_round()

        if (
            self.current_round is not None
            and self.current_round.status is RoundStatus.RESOLVED
        ):
            self.inter_round_delay -= dt
            if self.inter_round_delay <= 0 and not self.complete:
                self.current_round = None
                self._spawn_round()

        return event

    def _spawn_round(self) -> None:
        if self.complete:
            return

        question_number = len(self.rounds) + 1
        config = question_config(question_number)
        candidates = [
            direction
            for direction in Direction
            if direction is not self._last_direction
        ]
        direction = self.rng.choice(candidates)
        self._last_direction = direction
        self.current_round = FallingRound(
            question_number=question_number,
            direction=direction,
            font_size=config.font_size,
            speed=config.speed,
            judge_window=config.judge_window,
            x_ratio=self.rng.uniform(0.18, 0.82),
        )
        self._round_finalized = False
        self.inter_round_delay = 0.8

    def _finalize_round(self) -> RoundEvent:
        current = self.current_round
        if current is None or current.outcome is None:
            raise RuntimeError("当前题目尚未完成判定")

        self._round_finalized = True
        points = 0
        if current.outcome is RoundOutcome.CORRECT:
            self.correct_count += 1
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
            points = 10 + min(self.combo - 1, 5) * 2
            self.score += points
        elif current.outcome is RoundOutcome.WRONG:
            self.combo = 0
            points = -3
            self.score = max(0, self.score + points)
        else:
            self.combo = 0

        record = AttemptRecord(
            question_number=current.question_number,
            direction=current.direction,
            font_size=current.font_size,
            user_answer=current.user_answer,
            outcome=current.outcome,
            response_time_ms=current.response_time_ms,
            score_after=self.score,
        )
        self.rounds.append(record)

        event = RoundEvent(
            question_number=current.question_number,
            direction=current.direction,
            user_answer=current.user_answer,
            outcome=current.outcome,
            points=points,
            combo=self.combo,
            score=self.score,
            accuracy=self.accuracy,
            is_last_question=self.complete,
        )
        self.last_event = event
        self.inter_round_delay = 0.85
        return event
