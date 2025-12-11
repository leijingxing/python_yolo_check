import cv2
from ultralytics import YOLO
import os
import sqlite3
import time
import math
import winsound
from plyer import notification

# --- 模型加载 ---
# 1. 表情检测模型 (自定义训练的 v8)
emotion_model_path = os.path.join('YoloV8-Human-Emotion-Detection', 'best.pt')
print(f"正在加载表情模型: {emotion_model_path} ...")
model_emotion = YOLO(emotion_model_path)

# 2. 姿态估计模型 (最新的 YOLO11n-pose)
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

init_db()

# --- 辅助函数 ---
def calculate_angle(p1, p2):
    """计算两点连线相对于水平线的角度 (度数)"""
    x1, y1 = p1
    x2, y2 = p2
    # 计算斜率
    if x2 - x1 == 0:
        return 90.0
    slope = (y2 - y1) / (x2 - x1)
    angle_rad = math.atan(slope)
    angle_deg = math.degrees(angle_rad)
    return abs(angle_deg)

def send_alert(title, message):
    """发送系统通知和声音提示"""
    # 声音提示 (频率 800Hz, 持续 200ms)
    winsound.Beep(800, 200)
    
    # 系统通知 (非阻塞)
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

# --- 监控参数配置 ---
POSTURE_ANGLE_THRESHOLD = 25.0  # 肩膀倾斜阈值 (度)
ABNORMAL_DURATION_THRESHOLD = 3.0 # 异常持续时间阈值 (秒)
ALERT_COOLDOWN = 30.0 # 报警冷却时间 (秒)
LONG_SITTING_THRESHOLD = 3600.0 # 久坐阈值 (秒，1小时)
ABSENCE_RESET_THRESHOLD = 30.0 # 离座重置阈值 (秒，30秒无人视为离开)

# 状态追踪变量
last_record_time = time.time()
record_interval = 1.0 

# 报警状态
posture_start_time = None 
last_posture_alert_time = 0 

anger_start_time = None 
last_anger_alert_time = 0 

# 久坐追踪
sitting_start_time = None # 开始坐下的时间
last_person_seen_time = 0 # 最后一次看到人的时间
last_sitting_alert_time = 0 # 上次久坐报警时间

# --- 摄像头设置 ---
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

# 缓存上一帧结果
last_emotion_results = None
last_pose_results = None
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()
    
    # --- 交替推理 ---
    if frame_count % 2 != 0:
        results_emotion = model_emotion(frame, verbose=False)
        last_emotion_results = results_emotion
        results_pose = last_pose_results
    else:
        results_pose = model_pose(frame, verbose=False)
        last_pose_results = results_pose
        results_emotion = last_emotion_results

    # --- 绘制逻辑 ---
    
    # 始终在当前帧的副本上进行绘制
    annotated_frame = frame.copy()

    # 1. 绘制表情 (手动绘制，性能最高且无闪烁)
    if last_emotion_results:
        for r in last_emotion_results:
            if hasattr(r, 'boxes') and r.boxes:
                for box in r.boxes:
                    # 获取坐标 (x1, y1, x2, y2)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # 获取类别和置信度
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # 获取表情名称
                    emotion_en = model_emotion.names[cls_id]
                    emotion_zh = emotion_map_zh.get(emotion_en, emotion_en)
                    
                    # 绘制矩形框 (颜色: 橙色 (0, 165, 255) in BGR is (255, 165, 0))
                    # 这里用 BGR: (0, 140, 255)
                    color = (0, 140, 255) 
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # 绘制标签背景和文字
                    label = f"{emotion_en} {conf:.2f}"
                    # 计算文字大小
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    # 画文字背景框
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                    # 画文字 (白色)
                    cv2.putText(annotated_frame, label, (x1, y1 - 6), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 2. 绘制姿态 (手动叠加到 annotated_frame 上)
    # 始终使用最新的姿态结果绘制，保证画面上骨架的连续性
    if last_pose_results:
        for r in last_pose_results:
            if r.keypoints is not None and r.keypoints.data is not None and len(r.keypoints.data) > 0:
                # 获取关键点数据 (1, 17, 3) -> (x, y, conf)
                kpts = r.keypoints.data[0].cpu().numpy()
                
                # 定义骨架连接关系 (索引) - 仅绘制上半身和面部
                # 0:鼻子, 1:左眼, 2:右眼, 3:左耳, 4:右耳
                # 5:左肩, 6:右肩, 7:左肘, 8:右肘, 9:左腕, 10:右腕
                skeleton = [
                    (5, 7), (7, 9),       # 左臂
                    (6, 8), (8, 10),      # 右臂
                    (5, 6),               # 肩膀连接
                    # (5, 11), (6, 12),     # 躯干连接，可能被遮挡导致视觉效果不佳
                    (0, 1), (0, 2), (1, 3), (2, 4), # 鼻子到眼睛/耳朵 (面部)
                ]
                
                # 绘制连接线
                for p1_idx, p2_idx in skeleton:
                    # 确保索引在有效范围内
                    if p1_idx < len(kpts) and p2_idx < len(kpts):
                        x1, y1, conf1 = kpts[p1_idx]
                        x2, y2, conf2 = kpts[p2_idx]
                        # 仅绘制置信度高的连接线
                        if conf1 > 0.5 and conf2 > 0.5: 
                            # 坐姿不正时，肩膀线变红，其他保持绿色
                            if (p1_idx == 5 and p2_idx == 6) or (p1_idx == 6 and p2_idx == 5):
                                color = (0, 0, 255) if is_posture_bad else (0, 255, 0)
                            else:
                                color = (0, 255, 0) # 其他连接线绿色
                            cv2.line(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # 绘制关键点
                for i, (x, y, conf) in enumerate(kpts):
                    # 仅绘制置信度高的关键点
                    if conf > 0.5:
                        # 头部关键点用红色，身体用蓝色 (这里只是上半身)
                        color = (0, 0, 255) if i <= 4 else (255, 0, 0) # 0-4 是头部关键点
                        cv2.circle(annotated_frame, (int(x), int(y)), 3, color, -1)

    # --- 数据记录 (每秒) ---

    # --- 检测是否有人 ---
    is_person_present = False
    
    # 检查表情模型是否检测到人脸
    if last_emotion_results:
        for r in last_emotion_results:
            if hasattr(r, 'boxes') and len(r.boxes) > 0:
                is_person_present = True
                break
    
    # 或者检查姿态模型是否检测到人
    if not is_person_present and last_pose_results:
        for r in last_pose_results:
            if r.keypoints is not None and len(r.keypoints.data) > 0:
                 # 确保关键点置信度总和不是0 (空检测)
                 if r.keypoints.conf is not None and r.keypoints.conf.sum() > 1.0:
                    is_person_present = True
                    break

    # --- 久坐逻辑处理 ---
    if is_person_present:
        last_person_seen_time = current_time
        if sitting_start_time is None:
            sitting_start_time = current_time # 刚坐下
            print(f"[{time.strftime('%H:%M:%S')}] 检测到用户，开始久坐计时...")
        
        # 计算已坐时长
        sitting_duration = current_time - sitting_start_time
        
        # 触发久坐报警
        if sitting_duration > LONG_SITTING_THRESHOLD:
            if current_time - last_sitting_alert_time > ALERT_COOLDOWN:
                send_alert("健康提醒", f"您已经连续坐了 {int(sitting_duration/60)} 分钟了，起来活动一下吧！")
                last_sitting_alert_time = current_time
                
        # 在画面上显示久坐时长
        duration_str = time.strftime('%H:%M:%S', time.gmtime(sitting_duration))
        cv2.putText(annotated_frame, f"Sitting: {duration_str}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                   
    else:
        # 当前无人，检查是否超时重置
        if sitting_start_time is not None and (current_time - last_person_seen_time > ABSENCE_RESET_THRESHOLD):
            print(f"[{time.strftime('%H:%M:%S')}] 用户离开超过 {ABSENCE_RESET_THRESHOLD}秒，久坐计时重置。")
            sitting_start_time = None # 重置
            
    # --- 1. 姿态分析与监测 ---
    is_posture_bad = False
    shoulder_angle = 0.0
    
    if last_pose_results:
        for r in last_pose_results:
            # 增加空值和长度检查，防止 IndexError
            if r.keypoints is not None and r.keypoints.data is not None and len(r.keypoints.data) > 0:
                kpts = r.keypoints.data[0].cpu().numpy() # (17, 3)
                
                # 提取左右肩膀关键点 (索引 5 和 6)
                if len(kpts) > 6:
                    l_shoulder = kpts[5] # x, y, conf
                    r_shoulder = kpts[6]
                    
                    # 只有当两个肩膀置信度都较高时才计算
                    if l_shoulder[2] > 0.5 and r_shoulder[2] > 0.5:
                        p1 = (l_shoulder[0], l_shoulder[1])
                        p2 = (r_shoulder[0], r_shoulder[1])
                        
                        # 计算角度
                        shoulder_angle = calculate_angle(p1, p2)
                        
                        # 判断是否倾斜
                        if shoulder_angle > POSTURE_ANGLE_THRESHOLD:
                            is_posture_bad = True
                            
                        # 绘制肩膀连线 (正常绿，异常红)
                        color = (0, 0, 255) if is_posture_bad else (0, 255, 0)
                        cv2.line(annotated_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 3)
                        
                        # 显示角度数值
                        cv2.putText(annotated_frame, f"Angle: {shoulder_angle:.1f}", (int(p1[0]), int(p1[1])-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                # 未检测到有效关键点，跳过
                pass

    # 坐姿报警逻辑
    if is_posture_bad:
        if posture_start_time is None:
            posture_start_time = current_time # 开始计时
        elif current_time - posture_start_time >= ABNORMAL_DURATION_THRESHOLD:
            # 持续时间达标，检查冷却
            if current_time - last_posture_alert_time > ALERT_COOLDOWN:
                send_alert("坐姿提醒", f"肩膀严重倾斜 ({shoulder_angle:.1f}度)，请坐正！")
                last_posture_alert_time = current_time
    else:
        posture_start_time = None # 恢复正常，重置计时

    # --- 2. 情绪分析与监测 ---
    current_emotion = "neutral"
    
    if last_emotion_results:
        for r in last_emotion_results:
            if hasattr(r, 'boxes') and r.boxes:
                best_box = max(r.boxes, key=lambda x: x.conf)
                cls_id = int(best_box.cls)
                current_emotion = model_emotion.names[cls_id]
    
    # 愤怒报警逻辑
    if current_emotion == 'anger':
        if anger_start_time is None:
            anger_start_time = current_time
        elif current_time - anger_start_time >= ABNORMAL_DURATION_THRESHOLD:
            if current_time - last_anger_alert_time > ALERT_COOLDOWN:
                send_alert("情绪提醒", "检测到愤怒情绪持续，深呼吸放松一下吧~")
                last_anger_alert_time = current_time
    else:
        anger_start_time = None

    # --- 数据库记录 (每秒) ---
    if current_time - last_record_time >= record_interval:
        if last_emotion_results: 
            last_record_time = current_time
            # (简化的记录逻辑，沿用之前的结构)
            for r in last_emotion_results:
                if hasattr(r, 'boxes') and r.boxes:
                    best_box = max(r.boxes, key=lambda x: x.conf)
                    # ... 写入数据库代码保持简洁，只记录
                    cls_id = int(best_box.cls)
                    conf = float(best_box.conf)
                    emotion_en = model_emotion.names[cls_id]
                    try:
                        cursor.execute(f'INSERT INTO {TABLE_NAME} (emotion, confidence) VALUES (?, ?)', 
                                      (emotion_en, round(conf, 2)))
                        conn.commit()
                        print(f"记录: {emotion_map_zh.get(emotion_en, emotion_en)} | 姿态倾斜: {shoulder_angle:.1f}°")
                    except: pass

    # 显示画面
    cv2.imshow('AI Health Monitor', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
close_db()
print("--- 监控结束 ---")
