import cv2
from ultralytics import YOLO
import os
import sqlite3
import time

# --- 模型加载 ---
# 1. 表情检测模型 (自定义训练的 v8)
emotion_model_path = os.path.join('YoloV8-Human-Emotion-Detection', 'best.pt')
print(f"正在加载表情模型: {emotion_model_path} ...")
model_emotion = YOLO(emotion_model_path)

# 2. 姿态估计模型 (最新的 YOLO11n-pose)
# 第一次运行时会自动下载 yolo11n-pose.pt
print("正在加载姿态模型: yolo11n-pose.pt ...")
model_pose = YOLO('yolo11n-pose.pt')

# 定义英文表情到中文的映射
emotion_map_zh = {
    'anger': '愤怒',
    'content': '满足',
    'disgust': '厌恶',
    'fear': '恐惧',
    'happy': '开心/微笑',
    'neutral': '中性',
    'sad': '悲伤',
    'surprise': '惊讶'
}

# --- 数据库配置 ---
DB_FILE = 'emotion_monitor.db'
TABLE_NAME = 'emotion_records'
conn = None
cursor = None

def init_db():
    """初始化 SQLite 数据库和表"""
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
    """关闭数据库连接"""
    if conn:
        conn.close()
        print("数据库连接已关闭。")

init_db()

# --- 记录频率控制 ---
last_record_time = time.time()
record_interval = 1.0 

# --- 摄像头设置 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    close_db()
    exit()

print("\n--- 双模型监控系统启动 (表情 + 姿态) ---")
print("按 'q' 键退出程序。")

# 缓存上一帧的检测结果，用于交替显示的补帧
last_emotion_results = None
last_pose_results = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
    # --- 交替推理策略 ---
    # 奇数帧: 跑表情模型
    # 偶数帧: 跑姿态模型
    # 这样可以将 FPS 提升一倍，视觉上几乎无感
    
    if frame_count % 2 != 0:
        # 运行表情检测
        results_emotion = model_emotion(frame, verbose=False)
        last_emotion_results = results_emotion
        # 如果有姿态缓存，直接沿用
        results_pose = last_pose_results
    else:
        # 运行姿态检测
        results_pose = model_pose(frame, verbose=False)
        last_pose_results = results_pose
        # 如果有表情缓存，直接沿用
        results_emotion = last_emotion_results

    # --- 绘制逻辑 ---
    
    # 1. 绘制表情 (使用 ultralytics 自带的 plot)
    # 即使当前帧没跑模型，也用缓存结果绘制，保证画面不闪烁
    if last_emotion_results:
        annotated_frame = last_emotion_results[0].plot()
    else:
        annotated_frame = frame.copy()

    # 2. 绘制姿态 (手动叠加到 annotated_frame 上)
    if last_pose_results:
        for r in last_pose_results:
            if r.keypoints is not None and r.keypoints.data is not None:
                # 获取关键点数据 (1, 17, 3) -> (x, y, conf)
                kpts = r.keypoints.data[0].cpu().numpy()
                
                # 定义骨架连接关系 (索引)
                skeleton = [
                    (5, 7), (7, 9),       # 左臂
                    (6, 8), (8, 10),      # 右臂
                    (5, 6),               # 肩膀连接
                    (5, 11), (6, 12),     # 躯干
                    (11, 12),             # 髋部连接
                    (1, 2), (1, 3), (2, 4), # 脸部关键点
                ]
                
                # 绘制连接线
                for p1, p2 in skeleton:
                    if p1 < len(kpts) and p2 < len(kpts):
                        x1, y1, conf1 = kpts[p1]
                        x2, y2, conf2 = kpts[p2]
                        if conf1 > 0.5 and conf2 > 0.5: # 仅绘制置信度高的点
                            cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 绘制关键点
                for i, (x, y, conf) in enumerate(kpts):
                    if conf > 0.5:
                        # 头部关键点用红色，身体用蓝色
                        color = (0, 0, 255) if i < 5 else (255, 0, 0)
                        cv2.circle(annotated_frame, (int(x), int(y)), 4, color, -1)

    # --- 数据记录 (仅记录表情) ---
    current_time = time.time()
    if current_time - last_record_time >= record_interval:
        # 注意：我们要用最新的 *表情* 结果来记录，哪怕这一帧跑的是姿态
        if last_emotion_results: 
            last_record_time = current_time
            for r in last_emotion_results:
                if hasattr(r, 'boxes') and r.boxes:
                    best_box = max(r.boxes, key=lambda x: x.conf)
                    cls_id = int(best_box.cls)
                    conf = float(best_box.conf)
                    emotion_en = model_emotion.names[cls_id]
                    emotion_zh = emotion_map_zh.get(emotion_en, emotion_en)
                    
                    print(f"[{time.strftime('%H:%M:%S')}] 表情: {emotion_zh} ({conf:.2f})")
                    
                    try:
                        cursor.execute(f'''
                            INSERT INTO {TABLE_NAME} (emotion, confidence)
                            VALUES (?, ?)
                        ''', (emotion_en, round(conf, 2)))
                        conn.commit()
                    except sqlite3.Error as e:
                        print(f"写入失败: {e}")

    # 显示画面
    cv2.imshow('Emotion & Pose Monitor', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_db()
print("--- 监控结束 ---")
