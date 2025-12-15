from collections import Counter, deque
import math
import os
import sqlite3
import time
import winsound

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from plyer import notification
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO


# ----------------- 配置区 -----------------

# 表情映射（HardlyHumans 的 8 类标签）
emotion_map_zh = {
    'anger': '愤怒',
    'contempt': '轻蔑/不屑',
    'disgust': '厌恶',
    'fear': '恐惧',
    'happy': '开心/微笑',
    'neutral': '中性',
    'sad': '悲伤',
    'surprise': '惊讶'
}

# 数据库配置
DB_FILE = 'emotion_monitor.db'
TABLE_NAME = 'emotion_records'
conn = None
cursor = None

# 姿态 & 情绪 & 久坐监控阈值
POSTURE_ANGLE_THRESHOLD = 12.0        # 肩膀倾斜阈值 (度)
ABNORMAL_DURATION_THRESHOLD = 3.0     # 异常持续时间阈值 (秒)
ALERT_COOLDOWN = 30.0                 # 报警冷却时间 (秒)
LONG_SITTING_THRESHOLD = 3600.0       # 久坐阈值 (秒，1小时)
ABSENCE_RESET_THRESHOLD = 30.0        # 离座重置阈值 (秒，30秒无人视为离开)

# 表情识别相关配置
EMOTION_CONF_THRESHOLD = 0.6          # 表情分类置信度阈值
EMOTION_SMOOTH_WINDOW = 5            # 表情平滑窗口（帧数）
emotion_window = deque(maxlen=EMOTION_SMOOTH_WINDOW)

# ----------------- 数据库相关 -----------------


def init_db():
    global conn, cursor
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                emotion TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        ''')
        conn.commit()
        print(f"数据库 '{DB_FILE}' 已连接。")
    except sqlite3.Error as e:
        print(f"数据库初始化失败: {e}")
        exit()


def close_db():
    if conn:
        conn.close()
        print("数据库连接已关闭。")


# ----------------- 工具函数 -----------------


def calculate_angle(p1, p2):
    """计算两点连线相对于水平线的角度 (度数)"""
    x1, y1 = p1
    x2, y2 = p2
    if x2 - x1 == 0:
        return 90.0
    slope = (y2 - y1) / (x2 - x1)
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)
    return abs(angle_deg)


def send_alert(title, message):
    """发送系统通知和声音提示"""
    try:
        winsound.Beep(800, 200)
    except RuntimeError:
        # 某些环境不支持 Beep 就忽略
        pass

    try:
        notification.notify(
            title=title,
            message=message,
            app_name='AI 情绪&姿态监控',
            timeout=5
        )
        print(f"!!! 发送通知: [{title}] {message} !!!")
    except Exception as e:
        print(f"发送通知失败: {e}")


def classify_emotion_from_face(face_bgr, processor, classifier, device):
    """对裁剪出的人脸做表情分类，返回 (英文标签, 置信度)"""
    if face_bgr is None or face_bgr.size == 0:
        return None, 0.0

    img = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = classifier(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        conf, idx = torch.max(probs, dim=0)

    label = classifier.config.id2label[idx.item()]  # e.g. "anger"
    return label, float(conf)

def draw_chinese_text(img_bgr, text, pos, font_size=22, color=(255,255,255)):
    """
    在 OpenCV 图像上绘制中文（使用 PIL）
    img_bgr: OpenCV BGR 图像
    text: 显示内容
    pos: (x, y)
    """
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    #  Windows 11 默认字体：微软雅黑
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)

    draw.text(pos, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ----------------- 初始化 -----------------

init_db()

# 1. 人脸检测模型 (用你的 best.pt 只做人脸框)
emotion_model_path = os.path.join('YoloV8-Human-Emotion-Detection', 'best.pt')
print(f"正在加载人脸检测模型(仅用于人脸框): {emotion_model_path} ...")
face_detector = YOLO(emotion_model_path)

# 2. 表情分类模型 (HardlyHumans / ViT)
print("正在加载表情分类模型: HardlyHumans/Facial-expression-detection ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_processor = AutoImageProcessor.from_pretrained("HardlyHumans/Facial-expression-detection")
emotion_classifier = AutoModelForImageClassification.from_pretrained(
    "HardlyHumans/Facial-expression-detection"
).to(device)
emotion_classifier.eval()

# 3. 姿态估计模型 (YOLO11n-pose)
print("正在加载姿态模型: yolo11n-pose.pt ...")
model_pose = YOLO('yolo11n-pose.pt')

# ----------------- 状态追踪变量 -----------------

last_record_time = time.time()
record_interval = 1.0  # 每秒记录一次数据库

# 报警状态
posture_start_time = None          # 坐姿异常开始时间
last_posture_alert_time = 0        # 上次坐姿提醒时间

anger_start_time = None            # 愤怒情绪开始时间
last_anger_alert_time = 0          # 上次情绪提醒时间

# 久坐追踪
sitting_start_time = None          # 开始坐下的时间
last_person_seen_time = 0          # 最后一次看到人的时间
last_sitting_alert_time = 0        # 上次久坐提醒时间

# 表情缓存
smoothed_emotion = "neutral"
smoothed_emotion_conf = 0.0

# 摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    close_db()
    exit()

print("\n--- 全能监控系统启动 ---")
print(f"监控规则:")
print(f"1. 坐姿倾斜 > {POSTURE_ANGLE_THRESHOLD}度 (持续{ABNORMAL_DURATION_THRESHOLD}s)")
print(f"2. 愤怒情绪 (持续{ABNORMAL_DURATION_THRESHOLD}s)")
print(f"3. 久坐提醒 > {int(LONG_SITTING_THRESHOLD/60)}分钟 (离座{int(ABSENCE_RESET_THRESHOLD)}s重置)")
print("按 'q' 键退出程序。")

last_face_results = None
last_pose_results = None
frame_count = 0

is_posture_bad = False
shoulder_angle = 0.0

# ----------------- 主循环 -----------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    annotated_frame = frame.copy()

    # ---------- 交替推理：一帧做人脸，一帧做姿态 ----------
    # 注意：绘制/情绪识别可以用缓存结果，但“是否有人在/离座重置”只能用本帧新推理
    fresh_face_results = None
    fresh_pose_results = None
    if frame_count % 2 != 0:
        results_face = face_detector(frame, verbose=False)
        last_face_results = results_face
        fresh_face_results = results_face
        results_pose = last_pose_results
    else:
        results_pose = model_pose(frame, verbose=False)
        last_pose_results = results_pose
        fresh_pose_results = results_pose
        results_face = last_face_results

    # ---------- 1. 表情检测（YOLO 框 + HardlyHumans 分类 + 平滑） ----------
    current_raw_emotion = None
    current_raw_conf = 0.0

    if results_face:
        for r in results_face:
            if hasattr(r, 'boxes') and r.boxes:
                # 只取置信度最高的人脸框
                best_box = max(r.boxes, key=lambda x: float(x.conf))
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)

                h, w, _ = frame.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    break

                face_crop = frame[y1:y2, x1:x2]

                emo_en, emo_conf = classify_emotion_from_face(
                    face_crop, emotion_processor, emotion_classifier, device
                )
                if emo_en is None:
                    break

                current_raw_emotion = emo_en
                current_raw_conf = emo_conf

                # 高置信度结果加入平滑窗口
                if emo_conf >= EMOTION_CONF_THRESHOLD:
                    emotion_window.append(emo_en)

                # 表情时间平滑
                if len(emotion_window) > 0:
                  most_common, _ = Counter(emotion_window).most_common(1)[0]
                  smoothed_emotion = most_common
                  smoothed_emotion_conf = (
                    current_raw_conf if current_raw_emotion == smoothed_emotion else EMOTION_CONF_THRESHOLD
                )

                emo_zh = emotion_map_zh.get(smoothed_emotion, smoothed_emotion)

                # 画框和文字
                # 画框
                color = (0, 140, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # 显示中文标签 (使用平滑后的结果)
                label_text = f"{emo_zh} {smoothed_emotion_conf:.2f}"
                annotated_frame = draw_chinese_text(annotated_frame, label_text, (x1 + 5, y1 - 28), font_size=22, color=(255, 255, 255))

    if results_pose:
        for r in results_pose:
            if r.keypoints is None or r.keypoints.data is None or len(r.keypoints.data) == 0:
                continue

            kpts = r.keypoints.data[0].cpu().numpy()
            if len(kpts) <= 6:
                continue

            l_shoulder = kpts[5]
            r_shoulder = kpts[6]
            if l_shoulder[2] > 0.5 and r_shoulder[2] > 0.5:
                p1 = (l_shoulder[0], l_shoulder[1])
                p2 = (r_shoulder[0], r_shoulder[1])
                shoulder_angle = calculate_angle(p1, p2)
                is_posture_bad = shoulder_angle > POSTURE_ANGLE_THRESHOLD

                # 肩膀连线（颜色根据姿态）
                color = (0, 0, 255) if is_posture_bad else (0, 255, 0)
                cv2.line(annotated_frame, (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])), color, 3)
                cv2.putText(annotated_frame, f"Angle: {shoulder_angle:.1f}",
                            (int(p1[0]), int(p1[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 简单骨架连线（上半身 + 头部）
            skeleton = [
                (5, 7), (7, 9),        # 左臂
                (6, 8), (8, 10),       # 右臂
                (5, 6),                # 肩膀连接
                (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            ]

            for p1_idx, p2_idx in skeleton:
                if p1_idx < len(kpts) and p2_idx < len(kpts):
                    x1, y1, c1 = kpts[p1_idx]
                    x2, y2, c2 = kpts[p2_idx]
                    if c1 > 0.5 and c2 > 0.5:
                        if (p1_idx, p2_idx) in [(5, 6), (6, 5)]:
                            color = (0, 0, 255) if is_posture_bad else (0, 255, 0)
                        else:
                            color = (0, 255, 0)
                        cv2.line(annotated_frame, (int(x1), int(y1)),
                                 (int(x2), int(y2)), color, 2)

            # 关键点绘制
            for i, (x, y, c) in enumerate(kpts):
                if c > 0.5:
                    color = (0, 0, 255) if i <= 4 else (255, 0, 0)
                    cv2.circle(annotated_frame, (int(x), int(y)), 3, color, -1)

    # 坐姿报警逻辑
    if is_posture_bad:
        if posture_start_time is None:
            posture_start_time = current_time
        elif current_time - posture_start_time >= ABNORMAL_DURATION_THRESHOLD:
            if current_time - last_posture_alert_time > ALERT_COOLDOWN:
                send_alert("坐姿提醒", f"肩膀严重倾斜 ({shoulder_angle:.1f}度)，请坐正！")
                last_posture_alert_time = current_time
    else:
        posture_start_time = None

    # ---------- 3. 是否有人在 & 久坐逻辑 ----------
    # 只用本帧新推理的结果判断是否在座位，避免缓存导致“人走了还算在”
    def has_face(res_list):
        if not res_list:
            return False
        for rr in res_list:
            if hasattr(rr, "boxes") and rr.boxes is not None and len(rr.boxes) > 0:
                return True
        return False

    def has_pose(res_list):
        if not res_list:
            return False
        for rr in res_list:
            if rr.keypoints is not None and rr.keypoints.data is not None and len(rr.keypoints.data) > 0:
                conf = getattr(rr.keypoints, 'conf', None)
                if conf is None:
                    return True
                if conf.sum() > 1.0:
                    return True
        return False

    face_present = has_face(fresh_face_results)
    pose_present = has_pose(fresh_pose_results)
    is_person_present = face_present or pose_present

    if is_person_present:
        last_person_seen_time = current_time
        if sitting_start_time is None:
            sitting_start_time = current_time
            print(f"[{time.strftime('%H:%M:%S')}] 检测到用户，开始久坐计时...")

        sitting_duration = current_time - sitting_start_time

        if sitting_duration > LONG_SITTING_THRESHOLD:
            if current_time - last_sitting_alert_time > ALERT_COOLDOWN:
                send_alert("健康提醒",
                           f"您已经连续坐了 {int(sitting_duration/60)} 分钟了，起来活动一下吧！")
                last_sitting_alert_time = current_time

        duration_str = time.strftime('%H:%M:%S', time.gmtime(sitting_duration))
        cv2.putText(annotated_frame, f"Sitting: {duration_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        if sitting_start_time is not None and (current_time - last_person_seen_time > ABSENCE_RESET_THRESHOLD):
            print(f"[{time.strftime('%H:%M:%S')}] 用户离开超过 {ABSENCE_RESET_THRESHOLD} 秒，久坐计时重置。")
            sitting_start_time = None

    # ---------- 4. 情绪报警（用平滑后的情绪） ----------
    current_emotion = smoothed_emotion

    if current_emotion == 'anger':
        if anger_start_time is None:
            anger_start_time = current_time
        elif current_time - anger_start_time >= ABNORMAL_DURATION_THRESHOLD:
            if current_time - last_anger_alert_time > ALERT_COOLDOWN:
                send_alert("情绪提醒", "检测到愤怒情绪持续，深呼吸放松一下吧~")
                last_anger_alert_time = current_time
    else:
        anger_start_time = None

    # ---------- 5. 数据库记录（每秒一次，记录平滑后的情绪） ----------
    if current_time - last_record_time >= record_interval:
        last_record_time = current_time
        if smoothed_emotion_conf >= EMOTION_CONF_THRESHOLD:
            try:
                cursor.execute(
                    f'INSERT INTO {TABLE_NAME} (emotion, confidence) VALUES (?, ?)',
                    (smoothed_emotion, round(smoothed_emotion_conf, 2))
                )
                conn.commit()
                print(f"记录: {emotion_map_zh.get(smoothed_emotion, smoothed_emotion)} "
                      f"| 姿态倾斜: {shoulder_angle:.1f}° "
                      f"| conf={smoothed_emotion_conf:.2f}")
            except Exception as e:
                print(f"写入数据库失败: {e}")

    # ---------- 显示画面 ----------
    cv2.imshow('AI Health Monitor', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_db()
print("--- 监控结束 ---")
