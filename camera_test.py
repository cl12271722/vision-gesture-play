import time
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"没有找到手部模型文件：{MODEL_PATH}\n"
        "请确认 hand_landmarker.task 和 camera_test.py 位于同一个文件夹。"
    )

# 先由 Python 读取模型，避免 MediaPipe 原生库无法处理中文路径
model_data = MODEL_PATH.read_bytes()

# 创建新版 MediaPipe 手部关键点识别器
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_buffer=model_data),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise RuntimeError("无法打开摄像头，请检查摄像头是否被其他程序占用。")

print("摄像头已启动，伸出手查看关键点，按 Q 退出")
start_time = time.monotonic()

try:
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("无法读取摄像头画面")
                break

            # 镜像显示更符合面对摄像头的操作习惯
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int((time.monotonic() - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for hand_landmarks in result.hand_landmarks:
                mp.tasks.vision.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
                )

            cv2.imshow("MediaPipe camera test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
finally:
    camera.release()
    cv2.destroyAllWindows()
