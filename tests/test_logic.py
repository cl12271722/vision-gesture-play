from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from game.logic import (
    BooleanStabilizer,
    Direction,
    FallingRound,
    GameSession,
    RoundOutcome,
    RoundStatus,
    classify_index_direction,
    detect_ok_gesture,
)
from game.report import save_report


class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


def make_landmarks() -> list[Point]:
    return [Point(0.5, 0.5) for _ in range(21)]


class DirectionTests(unittest.TestCase):
    def test_index_pointing_up(self) -> None:
        landmarks = make_landmarks()
        landmarks[0] = Point(0.5, 0.9)
        landmarks[5] = Point(0.5, 0.62)
        landmarks[6] = Point(0.5, 0.47)
        landmarks[8] = Point(0.5, 0.30)

        direction = classify_index_direction(landmarks, 640, 480)

        self.assertIs(direction, Direction.UP)

    def test_detect_ok_gesture(self) -> None:
        landmarks = make_landmarks()
        landmarks[0] = Point(0.50, 0.88)
        landmarks[4] = Point(0.53, 0.52)
        landmarks[5] = Point(0.36, 0.65)
        landmarks[6] = Point(0.33, 0.55)
        landmarks[8] = Point(0.53, 0.52)
        landmarks[9] = Point(0.50, 0.60)
        landmarks[10] = Point(0.52, 0.42)
        landmarks[12] = Point(0.52, 0.25)
        landmarks[14] = Point(0.63, 0.44)
        landmarks[16] = Point(0.66, 0.30)
        landmarks[18] = Point(0.75, 0.50)
        landmarks[20] = Point(0.78, 0.39)

        self.assertTrue(detect_ok_gesture(landmarks, 640, 480))

    def test_boolean_stabilizer_requires_repeated_frames(self) -> None:
        stabilizer = BooleanStabilizer()
        self.assertFalse(stabilizer.update(True))
        self.assertFalse(stabilizer.update(True))
        self.assertFalse(stabilizer.update(True))
        self.assertFalse(stabilizer.update(True))
        self.assertTrue(stabilizer.update(True))


class RoundTests(unittest.TestCase):
    def test_correct_answer_resolves_round(self) -> None:
        round_item = FallingRound(
            question_number=1,
            direction=Direction.LEFT,
            font_size=90,
            speed=100,
            judge_window=1.0,
            x_ratio=0.5,
            y=100,
        )
        round_item.update(10.0, judge_y=500)
        outcome = round_item.submit(Direction.LEFT)

        self.assertIs(outcome, RoundOutcome.CORRECT)
        self.assertIs(round_item.status, RoundStatus.RESOLVED)

    def test_no_answer_becomes_miss(self) -> None:
        round_item = FallingRound(
            question_number=1,
            direction=Direction.RIGHT,
            font_size=90,
            speed=100,
            judge_window=0.5,
            x_ratio=0.5,
            y=100,
        )
        round_item.update(10.0, judge_y=500)
        round_item.update(0.6, judge_y=500)

        self.assertIs(round_item.outcome, RoundOutcome.MISS)
        self.assertIs(round_item.status, RoundStatus.RESOLVED)


class SessionTests(unittest.TestCase):
    def finish_round_with_correct_answer(self, session: GameSession) -> None:
        if session.current_round is None:
            session.update(1.0, judge_y=500)

        current = session.current_round
        self.assertIsNotNone(current)
        assert current is not None
        session.update(10.0, judge_y=500)
        event = session.update(0.0, judge_y=500, gesture=current.direction)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertIs(event.outcome, RoundOutcome.CORRECT)

    def test_full_twenty_question_session(self) -> None:
        session = GameSession(total_questions=20, rng=random.Random(7))

        for question_number in range(1, 21):
            self.finish_round_with_correct_answer(session)
            self.assertEqual(len(session.rounds), question_number)
            if question_number < 20:
                session.update(1.0, judge_y=500)

        self.assertTrue(session.complete)
        self.assertEqual(session.correct_count, 20)
        self.assertAlmostEqual(session.accuracy, 1.0)
        self.assertGreater(session.score, 200)

    def test_report_files_are_created(self) -> None:
        session = GameSession(total_questions=20, rng=random.Random(11))
        for question_number in range(1, 21):
            self.finish_round_with_correct_answer(session)
            if question_number < 20:
                session.update(1.0, judge_y=500)

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save_report(
                records=session.rounds,
                score=session.score,
                accuracy=session.accuracy,
                output_dir=Path(temp_dir),
            )
            self.assertTrue(paths.csv_path.exists())
            self.assertTrue(paths.summary_path.exists())
            csv_text = paths.csv_path.read_text(encoding="utf-8-sig")
            summary_text = paths.summary_path.read_text(encoding="utf-8")
            self.assertIn("E字方向", csv_text)
            self.assertIn("未作答题数", summary_text)


if __name__ == "__main__":
    unittest.main()
