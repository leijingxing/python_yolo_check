import sqlite3
import os
import datetime

# 数据库文件路径
DB_FILE = 'emotion_monitor.db'

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

def get_db_connection():
    """获取数据库连接"""
    if not os.path.exists(DB_FILE):
        print(f"错误: 数据库文件 '{DB_FILE}' 未找到。请先运行监控程序生成数据。")
        return None
    try:
        return sqlite3.connect(DB_FILE)
    except sqlite3.Error as e:
        print(f"数据库连接失败: {e}")
        return None

def print_stats(cursor, title, date_filter=None):
    """
    通用统计打印函数
    :param cursor: 数据库游标
    :param title: 报告标题 (如 "今日数据")
    :param date_filter: SQL WHERE 子句的时间过滤条件 (如 "date(timestamp) = date('now')")
    """
    print(f"\n>>> {title} <<<")
    
    where_clause = ""
    if date_filter:
        where_clause = f"WHERE {date_filter}"

    # 1. 统计该时间段的总记录数
    cursor.execute(f"SELECT COUNT(*) FROM emotion_records {where_clause}")
    total_records = cursor.fetchone()[0]
    
    if total_records == 0:
        print("  (暂无数据)")
        return

    print(f"  总记录数: {total_records}")

    # 2. 统计各类情绪分布
    cursor.execute(f"""
        SELECT emotion, COUNT(*) as count 
        FROM emotion_records 
        {where_clause}
        GROUP BY emotion 
        ORDER BY count DESC
    """)
    results = cursor.fetchall()
    
    print(f"  {'情绪类型':<10} | {'中文名称':<10} | {'数量':<6} | {'占比':<6}")
    print("  " + "-" * 45)
    
    for emotion_en, count in results:
        emotion_zh = emotion_map_zh.get(emotion_en, emotion_en)
        percentage = (count / total_records) * 100
        print(f"  {emotion_en:<14} | {emotion_zh:<12} | {count:<8} | {percentage:.1f}%")

def analyze_all_periods():
    conn = get_db_connection()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        print(f"====== 情绪监控综合统计报告 ======")
        print(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. 全部历史数据
        print_stats(cursor, "全部历史数据")

        # 2. 今日数据
        # SQLite: date(timestamp, 'localtime') 确保使用本地时间
        print_stats(cursor, "今日数据 (Today)", "date(timestamp, 'localtime') = date('now', 'localtime')")

        # 3. 本周数据 (从周一开始)
        # SQLite 'now', 'weekday 0', '-6 days' 计算逻辑可能较复杂，这里使用 strftime('%W')
        # 注意: strftime('%W') 周一为一周第一天。
        # 简单起见，这里筛选年份相同且周数相同的数据
        print_stats(cursor, "本周数据 (Current Week)", 
                    "strftime('%Y-%W', timestamp, 'localtime') = strftime('%Y-%W', 'now', 'localtime')")

        # 4. 本月数据
        print_stats(cursor, "本月数据 (Current Month)", 
                    "strftime('%Y-%m', timestamp, 'localtime') = strftime('%Y-%m', 'now', 'localtime')")

    except sqlite3.Error as e:
        print(f"数据库查询出错: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_all_periods()
