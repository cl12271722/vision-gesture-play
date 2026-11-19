from __future__ import annotations

import argparse
import sys

from game.app import GameApp
from game.vision import VisionError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="指尖识 E：手势体感视力训练游戏",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="不连接摄像头，使用方向键模拟手势进行演示",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="摄像头编号，默认使用 0",
    )
    parser.add_argument(
        "--no-face-check",
        action="store_true",
        help="关闭人脸检测，仅使用手势进行游戏",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="运行指定帧数后退出，主要用于自动测试",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        app = GameApp(
            demo=args.demo,
            camera_index=args.camera,
            face_check=not args.no_face_check,
        )
        app.run(max_frames=args.frames)
    except VisionError as exc:
        print(f"视觉模块启动失败：{exc}", file=sys.stderr)
        print(
            "可以先运行 python main.py --demo 检查游戏界面。",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
