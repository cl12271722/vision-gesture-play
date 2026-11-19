from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from game.app import HEIGHT, WIDTH, AppState, GameApp


class UserInterfaceSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.app = GameApp(
            demo=True,
            report_dir=Path(self.temp_dir.name),
        )

    def tearDown(self) -> None:
        pygame.quit()
        self.temp_dir.cleanup()

    def test_ready_screen_renders(self) -> None:
        self.app._update_vision(0)
        self.app._update(0.1)
        self.app._draw()

        self.assertEqual(self.app.screen.get_size(), (WIDTH, HEIGHT))
        self.assertEqual(self.app.state, AppState.READY)
        self.assertNotEqual(
            self.app.screen.get_at((50, 160)),
            self.app.screen.get_at((1050, 680)),
        )

    def test_summary_screen_renders(self) -> None:
        self.app._begin_session()
        for index in range(20):
            self.app._update(1.0)
            current = self.app.session.current_round
            self.assertIsNotNone(current)
            assert current is not None
            self.app._update(10.0)
            self.app.session.update(0.0, 586, current.direction)
            if index < 19:
                self.app._update(1.0)

        self.app._complete_session()
        self.app._draw()

        self.assertEqual(self.app.state, AppState.ROUND_COMPLETE)
        self.assertIsNotNone(self.app.report_paths)
        assert self.app.report_paths is not None
        self.assertTrue(self.app.report_paths.csv_path.exists())

        preview_dir = os.environ.get("FINGER_E_PREVIEW_DIR")
        if preview_dir:
            pygame.image.save(
                self.app.screen,
                Path(preview_dir) / "finger_e_summary.png",
            )


if __name__ == "__main__":
    unittest.main()
