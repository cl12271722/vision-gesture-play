from __future__ import annotations

import math
import os
import time
from enum import Enum
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2
import numpy as np
import pygame

from .logic import (
    BooleanStabilizer,
    Direction,
    GameSession,
    GestureStabilizer,
    RoundEvent,
    RoundOutcome,
    RoundStatus,
)
from .report import ReportPaths, build_advice, save_report
from .vision import VisionResult, VisionSystem


WIDTH = 1100
HEIGHT = 700
FPS = 60

HEADER_HEIGHT = 112
GAME_RECT = pygame.Rect(40, 134, 735, 522)
SIDEBAR_RECT = pygame.Rect(800, 134, 260, 522)
JUDGE_Y = 586
E_START_Y = 176

BG = (244, 245, 240)
INK = (28, 37, 48)
MUTED = (100, 112, 124)
NAVY = (25, 37, 54)
NAVY_LIGHT = (36, 51, 72)
WHITE = (255, 255, 255)
CORAL = (235, 92, 67)
CORAL_DARK = (186, 64, 48)
TEAL = (28, 129, 122)
GREEN = (41, 158, 103)
YELLOW = (244, 188, 66)
PALE = (225, 232, 229)
RED = (205, 67, 64)


class AppState(Enum):
    WAITING_FACE = "等待人脸"
    READY = "准备开始"
    RUNNING = "测试中"
    PAUSED_FACE = "人脸暂停"
    ROUND_COMPLETE = "本轮完成"


class FontBook:
    """缓存中文字体，避免每帧重复加载。"""

    def __init__(self) -> None:
        regular_candidates = [
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
        ]
        bold_candidates = [
            Path("C:/Windows/Fonts/msyhbd.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
        ]
        self._regular = self._pick_font(regular_candidates)
        self._bold = self._pick_font(bold_candidates, fallback=self._regular)
        self._cache: dict[tuple[int, bool], pygame.font.Font] = {}

    @staticmethod
    def _pick_font(
        candidates: list[Path],
        fallback: Path | None = None,
    ) -> Path | None:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return fallback

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._cache:
            path = self._bold if bold else self._regular
            self._cache[key] = pygame.font.Font(
                str(path) if path is not None else None,
                size,
            )
        return self._cache[key]


def draw_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    position: tuple[int, int],
    anchor: str = "topleft",
) -> pygame.Rect:
    image = font.render(text, True, color)
    rect = image.get_rect()
    setattr(rect, anchor, position)
    surface.blit(image, rect)
    return rect


def draw_dashed_line(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    y: int,
    x_start: int,
    x_end: int,
    dash: int = 12,
    gap: int = 8,
    width: int = 2,
) -> None:
    x = x_start
    while x < x_end:
        pygame.draw.line(surface, color, (x, y), (min(x + dash, x_end), y), width)
        x += dash + gap


def draw_direction_arrow(
    surface: pygame.Surface,
    center: tuple[int, int],
    direction: Direction,
    color: tuple[int, int, int],
    scale: float = 1.0,
) -> None:
    dx = float(direction.dx)
    dy = float(direction.dy)
    start = (
        int(center[0] - dx * 11 * scale),
        int(center[1] - dy * 11 * scale),
    )
    tip = (
        int(center[0] + dx * 18 * scale),
        int(center[1] + dy * 18 * scale),
    )
    perpendicular = (-dy, dx)
    head_left = (
        int(tip[0] - dx * 10 * scale + perpendicular[0] * 6 * scale),
        int(tip[1] - dy * 10 * scale + perpendicular[1] * 6 * scale),
    )
    head_right = (
        int(tip[0] - dx * 10 * scale - perpendicular[0] * 6 * scale),
        int(tip[1] - dy * 10 * scale - perpendicular[1] * 6 * scale),
    )
    pygame.draw.line(surface, color, start, tip, max(2, int(4 * scale)))
    pygame.draw.polygon(surface, color, [tip, head_left, head_right])


def wrap_text(
    text: str,
    font: pygame.font.Font,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for character in text:
        candidate = current + character
        if current and font.size(candidate)[0] > max_width:
            lines.append(current)
            current = character
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def frame_to_surface(frame: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface_array = np.transpose(rgb, (1, 0, 2))
    return pygame.surfarray.make_surface(surface_array)


class GameApp:
    """手势视力训练游戏主程序。"""

    def __init__(
        self,
        demo: bool = False,
        camera_index: int = 0,
        face_check: bool = True,
        report_dir: Path | None = None,
    ) -> None:
        pygame.display.init()
        pygame.font.init()
        pygame.display.set_caption("指尖识 E · 手势视力训练")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.fonts = FontBook()

        self.demo = demo
        self.face_check = face_check and not demo
        self.report_dir = (
            report_dir
            if report_dir is not None
            else Path(__file__).resolve().parent.parent / "logs"
        )
        self.vision: VisionSystem | None = None
        if not demo:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "hand_landmarker.task"
            )
            self.vision = VisionSystem(
                model_path=model_path,
                camera_index=camera_index,
                use_face_detection=self.face_check,
            )

        self.session = GameSession(total_questions=20)
        self.direction_stabilizer = GestureStabilizer()
        self.ok_stabilizer = BooleanStabilizer()
        self.state = (
            AppState.READY
            if demo or not self.face_check
            else AppState.WAITING_FACE
        )

        self.stable_direction: Direction | None = None
        self.ok_stable = False
        self.face_visible = self.demo or not self.face_check
        self.face_seen_time = 1.0 if self.face_visible else 0.0
        self.face_missing_time = 0.0
        self.camera_surface: pygame.Surface | None = None

        self.answer_armed = True
        self.last_round_object = None
        self.start_ok_armed = True
        self.start_ok_hold = 0.0
        self.feedback: RoundEvent | None = None
        self.feedback_timer = 0.0

        self.report_paths: ReportPaths | None = None
        self.report_error: str | None = None
        self.report_saved = False
        self.advice: list[str] = []
        self.running = True
        self._started_monotonic = time.monotonic()

    def run(self, max_frames: int | None = None) -> None:
        frame_count = 0
        try:
            while self.running:
                dt = min(self.clock.tick(FPS) / 1000.0, 0.1)
                self._handle_events()
                timestamp_ms = int(
                    (time.monotonic() - self._started_monotonic) * 1000
                )
                self._update_vision(timestamp_ms)
                self._update(dt)
                self._draw()
                pygame.display.flip()

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            if self.vision is not None:
                self.vision.close()
            pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE and self.state in (
                    AppState.READY,
                    AppState.ROUND_COMPLETE,
                ):
                    self._begin_session()

    def _update_vision(self, timestamp_ms: int) -> None:
        if self.demo:
            keys = pygame.key.get_pressed()
            raw_direction = None
            if keys[pygame.K_UP]:
                raw_direction = Direction.UP
            elif keys[pygame.K_DOWN]:
                raw_direction = Direction.DOWN
            elif keys[pygame.K_LEFT]:
                raw_direction = Direction.LEFT
            elif keys[pygame.K_RIGHT]:
                raw_direction = Direction.RIGHT

            self.stable_direction = self.direction_stabilizer.update(raw_direction)
            self.ok_stable = self.ok_stabilizer.update(keys[pygame.K_o])
            self.face_visible = True
            return

        if self.vision is None:
            return

        result: VisionResult = self.vision.read(timestamp_ms)
        self.stable_direction = self.direction_stabilizer.update(
            result.hand_direction
        )
        self.ok_stable = self.ok_stabilizer.update(result.ok_gesture)
        self.face_visible = result.face_detected

        preview_width = 228
        preview_height = 171
        camera_surface = frame_to_surface(result.frame)
        self.camera_surface = pygame.transform.smoothscale(
            camera_surface,
            (preview_width, preview_height),
        )

    def _update(self, dt: float) -> None:
        if self.feedback_timer > 0:
            self.feedback_timer = max(0.0, self.feedback_timer - dt)

        self._update_face_state(dt)
        self._update_start_gesture(dt)

        if self.state is not AppState.RUNNING:
            return

        current_round = self.session.current_round
        if (
            current_round is not None
            and current_round is not self.last_round_object
        ):
            self.last_round_object = current_round
            self.answer_armed = self.stable_direction is None

        if self.stable_direction is None:
            self.answer_armed = True

        gesture = self.stable_direction if self.answer_armed else None
        event = self.session.update(dt, JUDGE_Y, gesture)
        if event is not None:
            self.feedback = event
            self.feedback_timer = 1.15
            self.answer_armed = False

            if event.is_last_question:
                self._complete_session()

    def _update_face_state(self, dt: float) -> None:
        if not self.face_check:
            self.face_visible = True
            return

        if self.face_visible:
            self.face_seen_time += dt
            self.face_missing_time = 0.0
        else:
            self.face_seen_time = 0.0
            self.face_missing_time += dt

        if (
            self.state is AppState.WAITING_FACE
            and self.face_seen_time >= 0.65
        ):
            self.state = AppState.READY
            self.start_ok_armed = True
            self.start_ok_hold = 0.0

        if (
            self.state is AppState.READY
            and self.face_missing_time >= 1.2
        ):
            self.state = AppState.WAITING_FACE

        if (
            self.state is AppState.RUNNING
            and self.face_missing_time >= 0.9
        ):
            self.state = AppState.PAUSED_FACE
            self.start_ok_hold = 0.0

        if (
            self.state is AppState.PAUSED_FACE
            and self.face_seen_time >= 0.75
        ):
            self.state = AppState.RUNNING

    def _update_start_gesture(self, dt: float) -> None:
        if self.state not in (AppState.READY, AppState.ROUND_COMPLETE):
            self.start_ok_hold = 0.0
            return

        if not self.ok_stable:
            self.start_ok_armed = True
            self.start_ok_hold = 0.0
            return

        if not self.start_ok_armed:
            return

        self.start_ok_hold += dt
        if self.start_ok_hold >= 0.75:
            self._begin_session()

    def _begin_session(self) -> None:
        self.session.reset()
        self.state = AppState.RUNNING
        self.feedback = None
        self.feedback_timer = 0.0
        self.report_paths = None
        self.report_error = None
        self.report_saved = False
        self.advice = []
        self.last_round_object = None
        self.answer_armed = self.stable_direction is None
        self.start_ok_armed = False
        self.start_ok_hold = 0.0

    def _complete_session(self) -> None:
        self.state = AppState.ROUND_COMPLETE
        if self.report_saved:
            return

        self.advice = build_advice(
            self.session.rounds,
            self.session.accuracy,
        )
        try:
            self.report_paths = save_report(
                records=self.session.rounds,
                score=self.session.score,
                accuracy=self.session.accuracy,
                output_dir=self.report_dir,
            )
        except OSError as exc:
            self.report_error = f"报告保存失败：{exc}"
        self.report_saved = True

    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_header()
        self._draw_game_panel()
        self._draw_sidebar()

        if self.state is not AppState.RUNNING:
            self._draw_state_overlay()

    def _draw_header(self) -> None:
        pygame.draw.rect(self.screen, CORAL, (40, 26, 8, 56), border_radius=4)
        draw_text(
            self.screen,
            "指尖识 E",
            self.fonts.get(37, bold=True),
            INK,
            (62, 22),
        )
        draw_text(
            self.screen,
            "手势体感视力训练",
            self.fonts.get(15),
            MUTED,
            (64, 69),
        )

        draw_text(
            self.screen,
            "得分",
            self.fonts.get(13),
            MUTED,
            (785, 19),
        )
        draw_text(
            self.screen,
            f"{self.session.score:03d}",
            self.fonts.get(30, bold=True),
            INK,
            (784, 39),
        )

        draw_text(
            self.screen,
            "正确率",
            self.fonts.get(13),
            MUTED,
            (940, 19),
        )
        draw_text(
            self.screen,
            f"{self.session.accuracy:.0%}",
            self.fonts.get(31, bold=True),
            TEAL,
            (939, 38),
        )

        draw_text(
            self.screen,
            f"{self.session.question_number:02d} / {self.session.total_questions:02d}",
            self.fonts.get(14, bold=True),
            MUTED,
            (1058, 63),
            anchor="topright",
        )
        pygame.draw.line(self.screen, PALE, (40, 111), (1062, 111), 1)

    def _draw_game_panel(self) -> None:
        shadow = GAME_RECT.move(0, 5)
        pygame.draw.rect(self.screen, (219, 222, 216), shadow, border_radius=24)
        pygame.draw.rect(self.screen, NAVY, GAME_RECT, border_radius=24)

        inner = GAME_RECT.inflate(-2, -2)
        for x in range(inner.left + 90, inner.right, 90):
            pygame.draw.line(
                self.screen,
                (33, 48, 67),
                (x, inner.top + 62),
                (x, inner.bottom - 28),
                1,
            )

        draw_text(
            self.screen,
            "下落区",
            self.fonts.get(13, bold=True),
            (138, 157, 174),
            (GAME_RECT.left + 20, GAME_RECT.top + 17),
        )
        draw_text(
            self.screen,
            f"第 {self.session.question_number} 题",
            self.fonts.get(15, bold=True),
            WHITE,
            (GAME_RECT.right - 20, GAME_RECT.top + 16),
            anchor="topright",
        )

        line_color = (
            YELLOW
            if self.session.current_round is not None
            and self.session.current_round.status is RoundStatus.JUDGING
            else (94, 111, 128)
        )
        draw_dashed_line(
            self.screen,
            line_color,
            JUDGE_Y,
            GAME_RECT.left + 24,
            GAME_RECT.right - 24,
            dash=13,
            gap=8,
            width=3,
        )
        draw_text(
            self.screen,
            "判定线",
            self.fonts.get(12, bold=True),
            line_color,
            (GAME_RECT.left + 22, JUDGE_Y + 11),
        )

        self._draw_round()
        self._draw_feedback()

    def _draw_round(self) -> None:
        current = self.session.current_round
        if current is None:
            return

        center = (
            int(
                GAME_RECT.left
                + 82
                + current.x_ratio * (GAME_RECT.width - 164)
            ),
            int(current.y),
        )
        radius = max(34, int(current.font_size * 0.78))

        if current.status is RoundStatus.JUDGING:
            pulse = 5 + int(4 * math.sin(pygame.time.get_ticks() / 90))
            pygame.draw.circle(
                self.screen,
                YELLOW,
                center,
                radius + pulse,
                3,
            )
            progress_rect = pygame.Rect(
                center[0] - radius - 10,
                center[1] - radius - 10,
                (radius + 10) * 2,
                (radius + 10) * 2,
            )
            pygame.draw.arc(
                self.screen,
                CORAL,
                progress_rect,
                -math.pi / 2,
                -math.pi / 2 + math.pi * 2 * current.judge_progress,
                4,
            )

        shadow_center = (center[0] + 5, center[1] + 7)
        pygame.draw.circle(self.screen, (14, 24, 37), shadow_center, radius)
        pygame.draw.circle(self.screen, (247, 247, 241), center, radius)

        icon_size = max(30, int(current.font_size * 0.62))
        letter = self._make_e_surface(icon_size, CORAL_DARK)
        letter = pygame.transform.rotate(letter, current.direction.rotation)
        self.screen.blit(letter, letter.get_rect(center=center))

    @staticmethod
    def _make_e_surface(size: int, color: tuple[int, int, int]) -> pygame.Surface:
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        unit = size / 10
        bar = max(2, int(unit * 1.15))
        radius = max(1, bar // 2)
        pygame.draw.rect(
            surface,
            color,
            (int(unit * 2.3), int(unit * 1.4), int(unit * 4.2), bar),
            border_radius=radius,
        )
        pygame.draw.rect(
            surface,
            color,
            (int(unit * 2.3), int(unit * 1.4), bar, int(unit * 7.2)),
            border_radius=radius,
        )
        pygame.draw.rect(
            surface,
            color,
            (int(unit * 2.3), int(unit * 4.45), int(unit * 3.4), bar),
            border_radius=radius,
        )
        pygame.draw.rect(
            surface,
            color,
            (int(unit * 2.3), int(unit * 7.5), int(unit * 4.2), bar),
            border_radius=radius,
        )
        return surface

    def _draw_feedback(self) -> None:
        if self.feedback is None or self.feedback_timer <= 0:
            return

        event = self.feedback
        if event.outcome is RoundOutcome.CORRECT:
            text = f"判断正确  +{event.points}"
            color = GREEN
        elif event.outcome is RoundOutcome.WRONG:
            text = f"方向错误  {event.points}"
            color = RED
        else:
            text = "本轮未作答"
            color = YELLOW

        if event.combo >= 2 and event.outcome is RoundOutcome.CORRECT:
            text += f"  ·  {event.combo} 连击"

        draw_text(
            self.screen,
            text,
            self.fonts.get(22, bold=True),
            color,
            (GAME_RECT.centerx, JUDGE_Y + 43),
            anchor="midtop",
        )

    def _draw_sidebar(self) -> None:
        pygame.draw.rect(
            self.screen,
            (222, 225, 219),
            SIDEBAR_RECT.move(0, 5),
            border_radius=20,
        )
        pygame.draw.rect(
            self.screen,
            WHITE,
            SIDEBAR_RECT,
            border_radius=20,
        )

        draw_text(
            self.screen,
            "实时状态",
            self.fonts.get(17, bold=True),
            INK,
            (SIDEBAR_RECT.left + 15, SIDEBAR_RECT.top + 14),
        )

        preview_rect = pygame.Rect(
            SIDEBAR_RECT.left + 15,
            SIDEBAR_RECT.top + 48,
            228,
            171,
        )
        if self.camera_surface is not None:
            self.screen.blit(self.camera_surface, preview_rect)
        else:
            pygame.draw.rect(self.screen, NAVY, preview_rect, border_radius=12)
            draw_text(
                self.screen,
                "演示模式",
                self.fonts.get(18, bold=True),
                WHITE,
                preview_rect.center,
                anchor="center",
            )
            draw_text(
                self.screen,
                "方向键模拟手势",
                self.fonts.get(13),
                (154, 170, 184),
                (preview_rect.centerx, preview_rect.centery + 28),
                anchor="midtop",
            )

        border_color = GREEN if self.face_visible else RED
        pygame.draw.rect(
            self.screen,
            border_color,
            preview_rect,
            3,
            border_radius=12,
        )

        status_y = preview_rect.bottom + 12
        pygame.draw.circle(
            self.screen,
            border_color,
            (SIDEBAR_RECT.left + 24, status_y + 10),
            5,
        )
        status_text = "人脸正常" if self.face_visible else "未检测到人脸"
        draw_text(
            self.screen,
            status_text,
            self.fonts.get(14, bold=True),
            INK if self.face_visible else RED,
            (SIDEBAR_RECT.left + 36, status_y),
        )

        gesture_y = status_y + 38
        draw_text(
            self.screen,
            "当前手势",
            self.fonts.get(13),
            MUTED,
            (SIDEBAR_RECT.left + 16, gesture_y),
        )
        if self.stable_direction is None:
            draw_text(
                self.screen,
                "等待识别",
                self.fonts.get(18, bold=True),
                MUTED,
                (SIDEBAR_RECT.left + 163, gesture_y - 2),
                anchor="topright",
            )
        else:
            draw_direction_arrow(
                self.screen,
                (SIDEBAR_RECT.right - 35, gesture_y + 10),
                self.stable_direction,
                TEAL,
                scale=0.9,
            )
            draw_text(
                self.screen,
                self.stable_direction.label,
                self.fonts.get(20, bold=True),
                TEAL,
                (SIDEBAR_RECT.right - 64, gesture_y - 3),
                anchor="topright",
            )

        stats_y = gesture_y + 40
        pygame.draw.line(
            self.screen,
            PALE,
            (SIDEBAR_RECT.left + 16, stats_y),
            (SIDEBAR_RECT.right - 16, stats_y),
            1,
        )
        self._draw_sidebar_stat(
            "连击",
            str(self.session.combo),
            SIDEBAR_RECT.left + 16,
            stats_y + 15,
        )
        self._draw_sidebar_stat(
            "最佳连击",
            str(self.session.best_combo),
            SIDEBAR_RECT.left + 130,
            stats_y + 15,
        )
        self._draw_sidebar_stat(
            "正确",
            str(self.session.correct_count),
            SIDEBAR_RECT.left + 16,
            stats_y + 61,
        )
        self._draw_sidebar_stat(
            "错误 / 未答",
            f"{self.session.wrong_count} / {self.session.miss_count}",
            SIDEBAR_RECT.left + 130,
            stats_y + 61,
        )

        self._draw_progress_dots(stats_y + 125)

    def _draw_sidebar_stat(
        self,
        label: str,
        value: str,
        x: int,
        y: int,
    ) -> None:
        draw_text(
            self.screen,
            label,
            self.fonts.get(12),
            MUTED,
            (x, y),
        )
        draw_text(
            self.screen,
            value,
            self.fonts.get(21, bold=True),
            INK,
            (x, y + 17),
        )

    def _draw_progress_dots(self, y: int) -> None:
        start_x = SIDEBAR_RECT.left + 20
        gap = 22
        for index in range(self.session.total_questions):
            column = index % 10
            row = index // 10
            center = (start_x + column * gap, y + row * 24)
            if index < len(self.session.rounds):
                record = self.session.rounds[index]
                if record.outcome is RoundOutcome.CORRECT:
                    color = GREEN
                elif record.outcome is RoundOutcome.WRONG:
                    color = RED
                else:
                    color = YELLOW
            elif (
                self.session.current_round is not None
                and index == len(self.session.rounds)
            ):
                color = CORAL
            else:
                color = (211, 216, 216)
            pygame.draw.circle(self.screen, color, center, 5)

    def _draw_state_overlay(self) -> None:
        if self.state is AppState.ROUND_COMPLETE:
            self._draw_summary_overlay()
            return

        overlay = pygame.Surface(GAME_RECT.size, pygame.SRCALPHA)
        overlay.fill((12, 22, 34, 214))
        self.screen.blit(overlay, GAME_RECT.topleft)

        if self.state is AppState.WAITING_FACE:
            self._draw_face_waiting()
        elif self.state is AppState.READY:
            self._draw_ready_overlay()
        elif self.state is AppState.PAUSED_FACE:
            self._draw_face_pause()

    def _draw_face_waiting(self) -> None:
        center_x = GAME_RECT.centerx
        face_center = (center_x, GAME_RECT.top + 174)
        pygame.draw.circle(self.screen, (68, 87, 105), face_center, 52)
        pygame.draw.circle(
            self.screen,
            WHITE,
            (face_center[0] - 18, face_center[1] - 10),
            4,
        )
        pygame.draw.circle(
            self.screen,
            WHITE,
            (face_center[0] + 18, face_center[1] - 10),
            4,
        )
        pygame.draw.arc(
            self.screen,
            WHITE,
            pygame.Rect(face_center[0] - 24, face_center[1] + 3, 48, 30),
            0.2,
            math.pi - 0.2,
            3,
        )
        draw_text(
            self.screen,
            "请面向摄像头",
            self.fonts.get(34, bold=True),
            WHITE,
            (center_x, GAME_RECT.top + 252),
            anchor="midtop",
        )
        draw_text(
            self.screen,
            "检测到人脸后进入手势准备",
            self.fonts.get(17),
            (186, 201, 211),
            (center_x, GAME_RECT.top + 306),
            anchor="midtop",
        )

    def _draw_ready_overlay(self) -> None:
        center_x = GAME_RECT.centerx
        draw_text(
            self.screen,
            "手势准备",
            self.fonts.get(35, bold=True),
            WHITE,
            (center_x, GAME_RECT.top + 84),
            anchor="midtop",
        )

        circle_a = (center_x - 24, GAME_RECT.top + 187)
        circle_b = (center_x + 24, GAME_RECT.top + 187)
        pygame.draw.circle(self.screen, NAVY_LIGHT, circle_a, 31, 7)
        pygame.draw.circle(self.screen, NAVY_LIGHT, circle_b, 31, 7)
        pygame.draw.circle(self.screen, YELLOW, circle_a, 5)
        pygame.draw.circle(self.screen, YELLOW, circle_b, 5)

        draw_text(
            self.screen,
            "用拇指和食指比出 OK",
            self.fonts.get(22, bold=True),
            YELLOW,
            (center_x, GAME_RECT.top + 242),
            anchor="midtop",
        )
        draw_text(
            self.screen,
            "保持约 1 秒开始 20 题测试",
            self.fonts.get(17),
            (198, 210, 219),
            (center_x, GAME_RECT.top + 282),
            anchor="midtop",
        )
        draw_text(
            self.screen,
            "E 到达判定线时，用食指指向开口方向",
            self.fonts.get(17),
            WHITE,
            (center_x, GAME_RECT.top + 334),
            anchor="midtop",
        )

    def _draw_face_pause(self) -> None:
        center_x = GAME_RECT.centerx
        draw_text(
            self.screen,
            "测试已暂停",
            self.fonts.get(36, bold=True),
            WHITE,
            (center_x, GAME_RECT.top + 122),
            anchor="midtop",
        )
        draw_text(
            self.screen,
            "暂时没检测到人脸",
            self.fonts.get(21, bold=True),
            YELLOW,
            (center_x, GAME_RECT.top + 190),
            anchor="midtop",
        )
        draw_text(
            self.screen,
            "回到摄像头前会自动继续",
            self.fonts.get(17),
            (198, 210, 219),
            (center_x, GAME_RECT.top + 232),
            anchor="midtop",
        )

    def _draw_summary_overlay(self) -> None:
        scrim = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        scrim.fill((20, 28, 38, 128))
        self.screen.blit(scrim, (0, 0))

        modal = pygame.Rect(120, 66, 860, 568)
        pygame.draw.rect(
            self.screen,
            (211, 214, 208),
            modal.move(0, 7),
            border_radius=24,
        )
        pygame.draw.rect(self.screen, WHITE, modal, border_radius=24)
        pygame.draw.rect(
            self.screen,
            CORAL,
            (modal.left + 32, modal.top + 29, 8, 44),
            border_radius=4,
        )
        draw_text(
            self.screen,
            "本轮完成",
            self.fonts.get(34, bold=True),
            INK,
            (modal.left + 56, modal.top + 24),
        )
        draw_text(
            self.screen,
            "20 题测试结果",
            self.fonts.get(14),
            MUTED,
            (modal.left + 58, modal.top + 70),
        )

        stat_y = modal.top + 112
        self._draw_summary_stat(
            "正确率",
            f"{self.session.accuracy:.0%}",
            modal.left + 56,
            stat_y,
            TEAL,
        )
        self._draw_summary_stat(
            "总得分",
            str(self.session.score),
            modal.left + 258,
            stat_y,
            CORAL,
        )
        self._draw_summary_stat(
            "最佳连击",
            str(self.session.best_combo),
            modal.left + 460,
            stat_y,
            YELLOW,
        )

        draw_text(
            self.screen,
            "用眼建议",
            self.fonts.get(19, bold=True),
            INK,
            (modal.left + 56, modal.top + 216),
        )

        advice_font = self.fonts.get(15)
        line_y = modal.top + 253
        for advice in self.advice[:4]:
            lines = wrap_text(f"· {advice}", advice_font, modal.width - 112)
            for line in lines:
                draw_text(
                    self.screen,
                    line,
                    advice_font,
                    (66, 77, 88),
                    (modal.left + 56, line_y),
                )
                line_y += 23

        report_y = modal.bottom - 95
        pygame.draw.line(
            self.screen,
            PALE,
            (modal.left + 56, report_y - 14),
            (modal.right - 56, report_y - 14),
            1,
        )
        if self.report_paths is not None:
            report_text = f"逐题日志已保存：{self.report_paths.csv_path.name}"
            report_color = GREEN
        elif self.report_error is not None:
            report_text = self.report_error
            report_color = RED
        else:
            report_text = "正在保存测试日志"
            report_color = MUTED
        draw_text(
            self.screen,
            report_text,
            self.fonts.get(13),
            report_color,
            (modal.left + 56, report_y),
        )

        start_text = (
            "比出 OK 手势开始下一轮"
            if self.face_check
            else "按空格开始下一轮"
        )
        if self.demo:
            start_text = "按住 O 键开始下一轮"
        draw_text(
            self.screen,
            start_text,
            self.fonts.get(17, bold=True),
            INK,
            (modal.centerx, modal.bottom - 54),
            anchor="midtop",
        )

    def _draw_summary_stat(
        self,
        label: str,
        value: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        draw_text(
            self.screen,
            label,
            self.fonts.get(13),
            MUTED,
            (x, y),
        )
        draw_text(
            self.screen,
            value,
            self.fonts.get(34, bold=True),
            color,
            (x, y + 20),
        )
