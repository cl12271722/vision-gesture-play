from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .logic import Direction, classify_index_direction, detect_ok_gesture


class VisionError(RuntimeError):
    """视觉模块无法启动时抛出的异常。"""


@dataclass(frozen=True)
class VisionResult:
    frame: np.ndarray
    hand_direction: Direction | None
    hand_detected: bool
    ok_gesture: bool
    face_detected: bool
    face_box: tuple[int, int, int, int] | None


class VisionSystem:
    """统一管理摄像头、手部识别和人脸检测。"""

    def __init__(
        self,
        model_path: Path,
        camera_index: int = 0,
        use_face_detection: bool = True,
    ) -> None:
        if not model_path.exists():
            raise VisionError(
                f"没有找到手部模型文件：{model_path}\n"
                "请确认 hand_landmarker.task 位于项目根目录。"
            )

        # 通过内存缓冲区加载模型，绕开 Windows 中文路径兼容问题
        self._model_data = model_path.read_bytes()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_buffer=self._model_data
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.62,
            min_hand_presence_confidence=0.62,
            min_tracking_confidence=0.62,
        )
        try:
            self._landmarker = (
                mp.tasks.vision.HandLandmarker.create_from_options(options)
            )
        except Exception as exc:
            raise VisionError(f"手部识别模型加载失败：{exc}") from exc
        self._use_face_detection = use_face_detection
        self._face_detector: mp.tasks.vision.FaceDetector | None = None
        self._face_model_data: bytes | None = None
        self._last_face_check_ms = -10_000
        self._last_face_detected = False
        self._last_face_box: tuple[int, int, int, int] | None = None
        self._closed = False

        if use_face_detection:
            face_model_path = model_path.with_name(
                "blaze_face_short_range.task"
            )
            if not face_model_path.exists():
                self._landmarker.close()
                raise VisionError(
                    f"没有找到人脸模型文件：{face_model_path}\n"
                    "请确认 blaze_face_short_range.task 位于项目根目录。"
                )

            self._face_model_data = face_model_path.read_bytes()
            face_options = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_buffer=self._face_model_data
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                min_detection_confidence=0.5,
            )
            try:
                self._face_detector = (
                    mp.tasks.vision.FaceDetector.create_from_options(
                        face_options
                    )
                )
            except Exception as exc:
                self._landmarker.close()
                raise VisionError(f"人脸检测模型加载失败：{exc}") from exc

        try:
            self._capture = self._open_camera(camera_index)
        except VisionError:
            if self._face_detector is not None:
                self._face_detector.close()
            self._landmarker.close()
            raise

    @staticmethod
    def _open_camera(camera_index: int) -> cv2.VideoCapture:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        for backend in backends:
            capture = cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            capture.set(cv2.CAP_PROP_FPS, 30)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture

        raise VisionError(
            "无法打开摄像头。请检查摄像头是否被其他程序占用，"
            "或尝试在命令中增加 --camera 1。"
        )

    def read(self, timestamp_ms: int) -> VisionResult:
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise VisionError("摄像头已连接，但无法读取画面。")

        # 镜像画面更符合用户面对摄像头时的左右直觉
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        direction = None
        ok_gesture = False
        hand_detected = bool(result.hand_landmarks)

        if hand_detected:
            landmarks = result.hand_landmarks[0]
            direction = classify_index_direction(landmarks, width, height)
            ok_gesture = detect_ok_gesture(landmarks, width, height)
            mp.tasks.vision.drawing_utils.draw_landmarks(
                frame,
                landmarks,
                mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
            )

        face_detected, face_box = self._detect_face(
            mp_image,
            timestamp_ms,
        )
        if face_box is not None:
            x, y, w, h = face_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (48, 196, 143), 2)

        return VisionResult(
            frame=frame,
            hand_direction=direction,
            hand_detected=hand_detected,
            ok_gesture=ok_gesture,
            face_detected=face_detected,
            face_box=face_box,
        )

    def _detect_face(
        self,
        mp_image: mp.Image,
        timestamp_ms: int,
    ) -> tuple[bool, tuple[int, int, int, int] | None]:
        if not self._use_face_detection or self._face_detector is None:
            return True, None

        # 人脸检测控制为每 250 毫秒一次，降低 CPU 占用
        if timestamp_ms - self._last_face_check_ms < 250:
            return self._last_face_detected, self._last_face_box

        self._last_face_check_ms = timestamp_ms
        result = self._face_detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if not result.detections:
            self._last_face_detected = False
            self._last_face_box = None
            return False, None

        largest = max(
            result.detections,
            key=lambda item: (
                item.bounding_box.width * item.bounding_box.height
            ),
        )
        box = (
            int(largest.bounding_box.origin_x),
            int(largest.bounding_box.origin_y),
            int(largest.bounding_box.width),
            int(largest.bounding_box.height),
        )
        self._last_face_detected = True
        self._last_face_box = box
        return True, box

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._capture.release()
        if self._face_detector is not None:
            self._face_detector.close()
        self._landmarker.close()

    def __enter__(self) -> "VisionSystem":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
