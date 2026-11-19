# 指尖识 E：手势体感视力训练游戏

这是一个基于 Python、OpenCV、MediaPipe 和 Pygame 的课程大作业。程序通过摄像头检测人脸与手部关键点，用 OK 手势开始测试，用食指指向 E 字开口方向进行作答，并自动生成逐题过程日志和用眼建议。

当前版本已经完成核心游戏流程，可以直接在本地运行。

## 已实现功能

- 摄像头实时画面与手部关键点显示。
- MediaPipe 手部关键点识别与食指方向判断。
- OK 手势识别：保持约 1 秒后开始测试。
- 人脸检测：无人脸时自动暂停，回到摄像头前自动继续。
- E 字随机朝向上、下、左、右并逐题下落。
- E 字到达判定线后开启方向作答窗口。
- 固定 20 题测试，完成后自动暂停并显示总结。
- 字号、下落速度和判定时间随题号逐步变化。
- 得分、连击、正确、错误和未作答统计。
- 右上角实时显示正确率。
- 自动保存逐题 CSV 过程日志。
- 自动生成 TXT 总结报告和护眼建议。
- 无摄像头时可使用方向键进行演示模式测试。

本项目的视力训练结果仅用于课程展示和趣味练习，不能替代专业医学检查。

## 游戏流程

1. 启动后先检测人脸。
2. 人脸稳定后，用拇指和食指比出 OK 手势。
3. 保持 OK 手势约 1 秒，开始 20 题测试。
4. E 字从上方下落，到达判定线后用食指指向开口方向。
5. 方向正确时计分，方向错误会扣分并清空连击。
6. 20 题完成后自动暂停并生成总结报告。

## 快速运行

### 1. 进入项目目录

```cmd
cd /d "D:\HuaweiMoveData\Users\深林\Desktop\python大作业"
```

### 2. 激活虚拟环境

```cmd
.\.venv\Scripts\activate.bat
```

激活成功后，命令提示符前应出现 `(.venv)`。

### 3. 启动游戏

```cmd
python main.py
```

也可以直接双击：

```text
run_game.bat
```

### 4. 使用方向键演示

没有摄像头或需要检查界面时，可以运行：

```cmd
python main.py --demo
```

演示模式操作：

- 按住 `O` 键模拟 OK 手势开始。
- 使用键盘方向键模拟手指方向。
- 按 `Esc` 退出。

## 摄像头与参数

默认使用编号为 `0` 的摄像头：

```cmd
python main.py
```

如果默认摄像头无法打开，可以尝试：

```cmd
python main.py --camera 1
```

关闭人脸检测但保留手势识别：

```cmd
python main.py --no-face-check
```

## 环境要求

| 项目 | 建议配置 |
| --- | --- |
| 系统 | Windows 10 或 Windows 11 64 位 |
| Python | 3.10.11 |
| 内存 | 8 GB 及以上 |
| 摄像头 | 笔记本内置摄像头或 USB 摄像头 |
| 显卡 | 无特殊要求，普通 CPU 即可运行 |

当前验证依赖版本：

```text
mediapipe==1.2.3
opencv-contrib-python==5.0.0.93
pygame==2.6.1
numpy==2.2.6
```

依赖清单位于 `requirements.txt`。

## 安装依赖

如果虚拟环境尚未创建：

```cmd
py -3.10 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

依赖导入测试：

```cmd
python -c "import cv2, mediapipe, pygame, numpy; print('依赖导入成功')"
```

## 手部模型

项目根目录必须包含以下两个 MediaPipe 模型：

```text
hand_landmarker.task
blaze_face_short_range.task
```

两个模型分别用于手部关键点识别和人脸检测。

如果文件丢失，可以从官方地址重新下载：

<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>

<https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite>

程序通过 `model_asset_buffer` 从内存加载模型，以避免 Windows 中文路径导致 MediaPipe 原生库报 `Unable to open file`。

## 摄像头测试

项目中的 `camera_test.py` 可以单独验证摄像头、手部模型和关键点识别：

```cmd
python camera_test.py
```

正常运行时，窗口中会显示手部的 21 个关键点。按 `Q` 退出。

## 项目结构

```text
python大作业/
├── main.py                       # 游戏启动入口
├── run_game.bat                  # Windows 快捷启动脚本
├── requirements.txt              # Python 依赖清单
├── hand_landmarker.task          # MediaPipe 手部关键点模型
├── blaze_face_short_range.task   # MediaPipe 人脸检测模型
├── camera_test.py                # 摄像头与手部关键点测试
├── README.md                     # 项目说明
├── game/
│   ├── __init__.py
│   ├── app.py                    # Pygame 界面与游戏状态机
│   ├── logic.py                  # 方向、OK 手势、题目与计分逻辑
│   ├── report.py                 # CSV 日志和建议报告
│   └── vision.py                 # 摄像头、MediaPipe 和人脸检测
├── tests/
│   ├── test_logic.py             # 核心逻辑测试
│   └── test_ui.py                # 无窗口界面冒烟测试
├── logs/                         # 运行后自动生成测试报告
├── 功能.docx                     # 功能需求文档
├── 环境配置.txt                  # 原始环境说明
└── 课程大作业安排.pptx           # 课程提交与答辩要求
```

`.venv/`、`__pycache__/` 和 `logs/` 属于本地生成内容，不应提交给队友。

## 技术实现

### 手部识别

`game/vision.py` 使用新版 MediaPipe Tasks API：

```python
mp.tasks.vision.HandLandmarker
```

摄像头画面先水平镜像，再从 MediaPipe 返回的 21 个手部关键点中取食指掌指关节和指尖，计算方向向量。

方向向量按角度划分为四个 90 度区间：

- 右：`-45°` 到 `45°`
- 下：`45°` 到 `135°`
- 左：`135°` 到 `225°`
- 上：`225°` 到 `315°`

方向识别结果使用连续多帧投票稳定，减少画面抖动造成的误判。

### OK 手势

OK 手势满足以下条件时识别成功：

- 拇指指尖与食指指尖距离足够近。
- 中指、无名指和小指保持伸开。
- 连续多帧检测成功后状态才会稳定。

### 人脸暂停

`game/vision.py` 使用 MediaPipe FaceDetector 和 `blaze_face_short_range.task` 模型检测人脸。人脸检测每隔约 250 毫秒执行一次，以降低 CPU 占用。

- 起始状态未检测到人脸：等待人脸。
- 游戏中连续无人脸约 0.9 秒：自动暂停。
- 人脸恢复约 0.75 秒：自动继续本轮测试。

## 报告输出

完成 20 题后，程序会在 `logs/` 目录生成两个文件。

CSV 过程日志字段：

| 字段 | 说明 |
| --- | --- |
| 题号 | 当前是第几题 |
| E字方向 | 上、下、左、右 |
| 字号 | 本题 E 字的显示字号 |
| 用户作答 | 用户手势方向，未作答时显示“未作答” |
| 判定结果 | 正确、错误或未作答 |
| 反应时间毫秒 | 进入判定区后到完成作答的时间 |
| 累计得分 | 本题结束后的总分 |

TXT 总结报告包含：

- 完成题数。
- 正确、错误和未作答题数。
- 正确率与总得分。
- 容易出错的方向。
- 根据本轮表现生成的用眼建议。

## 自动化测试

运行全部测试：

```cmd
python -m unittest discover -s tests -v
```

测试覆盖：

- 食指上下左右方向分类。
- OK 手势识别与多帧稳定。
- 正确、错误和未作答判定。
- 完整 20 题计分流程。
- CSV 和 TXT 报告生成。
- 准备页与总结页的无窗口渲染。

## 常见问题

### `module 'mediapipe' has no attribute 'solutions'`

新版 MediaPipe 已移除旧版 `mp.solutions.hands`。本项目使用新版：

```python
mp.tasks.vision.HandLandmarker
```

不要直接替换成旧教程中的 `mp.solutions` 写法。

### `Unable to open file ...`

该问题通常由 Windows 中文路径引起。当前程序已经把两个模型读取为内存字节并传给 MediaPipe，不要再改回直接传入 `model_asset_path`。同时确认 `hand_landmarker.task` 和 `blaze_face_short_range.task` 都位于项目根目录。

### `ModuleNotFoundError`

通常是没有激活项目虚拟环境，或依赖安装到了其他 Python。检查：

```cmd
python --version
where python
```

`where python` 第一行应指向：

```text
...\python大作业\.venv\Scripts\python.exe
```

### 摄像头无法打开

- 在 Windows“设置 → 隐私和安全性 → 相机”中允许桌面应用访问摄像头。
- 关闭腾讯会议、浏览器会议页面等正在占用摄像头的程序。
- 检查摄像头物理遮挡开关。
- 尝试运行 `python main.py --camera 1`。

### 手势方向经常跳动

- 保证手部有充足照明。
- 食指尽量伸直，并保持指向动作约半秒。
- 手不要离摄像头太近或太远。
- 使用纯色背景可以减少识别干扰。

## 建议的双人分工

| 成员 | 主要任务 |
| --- | --- |
| 成员 A | 摄像头、MediaPipe 手部识别、OK 手势、人脸检测、报告生成 |
| 成员 B | Pygame 界面、E 字动画、判定窗口、计分、难度和交互 |
| 共同完成 | 联合调试、测试记录、PPT、录屏和现场答辩 |

## 课程答辩建议

汇报时可以重点展示：

- 项目解决的趣味视力训练问题。
- 人脸检测和 OK 手势开始测试。
- 手指方向与 E 字方向的实时匹配。
- 20 题流程、右上角正确率和难度变化。
- 无人脸自动暂停。
- 逐题 CSV 日志和自动生成的用眼建议。
- 两名成员的具体分工和测试结果。
>>>>>>> 9cccdd9 (fitst commit)
