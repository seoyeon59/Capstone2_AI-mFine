"""
# SQLite 버전 (TEST용)
import sqlite3

conn = sqlite3.connect('capstone2.db')
c = conn.cursor()

c.executescript('''
-- 1️⃣ 사용자 테이블
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(50) PRIMARY KEY NOT NULL UNIQUE,
    password TEXT NOT NULL,
    username TEXT NOT NULL,
    phone_number INTEGER NOT NULL,
    non_guardian_name TEXT NOT NULL,
    mail TEXT    
);

-- 2️⃣ 카메라 테이블
CREATE TABLE IF NOT EXISTS cameras (
    user_id VARCHAR(50) PRIMARY KEY,
    camera_url TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 3️⃣ AI 학습 테이블
-- 3️⃣ AI 학습 테이블 (PK를 id로 변경)
CREATE TABLE IF NOT EXISTS ai_learning (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- 새로운 PK
    video TEXT NOT NULL,
    file_id TEXT NOT NULL,
    frame INT NOT NULL,
    timestamp FLOAT,
    neck_angle FLOAT,
    neck_angular_velocity FLOAT,
    neck_angular_acceleration FLOAT,
    neck_fast_ratio FLOAT,
    neck_stationary_ratio FLOAT,
    neck_peak_interval FLOAT,
    shoulder_balance_angle FLOAT,
    shoulder_balance_angular_velocity FLOAT,
    shoulder_balance_angular_acceleration FLOAT,
    shoulder_balance_fast_ratio FLOAT,
    shoulder_balance_stationary_ratio FLOAT,
    shoulder_balance_peak_interval FLOAT,
    shoulder_left_angle FLOAT,
    shoulder_left_angular_velocity FLOAT,
    shoulder_left_angular_acceleration FLOAT,
    shoulder_left_fast_ratio FLOAT,
    shoulder_left_stationary_ratio FLOAT,
    shoulder_left_peak_interval FLOAT,
    shoulder_right_angle FLOAT,
    shoulder_right_angular_velocity FLOAT,
    shoulder_right_angular_acceleration FLOAT,
    shoulder_right_fast_ratio FLOAT,
    shoulder_right_stationary_ratio FLOAT,
    shoulder_right_peak_interval FLOAT,
    elbow_left_angle FLOAT,
    elbow_left_angular_velocity FLOAT,
    elbow_left_angular_acceleration FLOAT,
    elbow_left_fast_ratio FLOAT,
    elbow_left_stationary_ratio FLOAT,
    elbow_left_peak_interval FLOAT,
    elbow_right_angle FLOAT,
    elbow_right_angular_velocity FLOAT,
    elbow_right_angular_acceleration FLOAT,
    elbow_right_fast_ratio FLOAT,
    elbow_right_stationary_ratio FLOAT,
    elbow_right_peak_interval FLOAT,
    hip_left_angle FLOAT,
    hip_left_angular_velocity FLOAT,
    hip_left_angular_acceleration FLOAT,
    hip_left_fast_ratio FLOAT,
    hip_left_stationary_ratio FLOAT,
    hip_left_peak_interval FLOAT,
    hip_right_angle FLOAT,
    hip_right_angular_velocity FLOAT,
    hip_right_angular_acceleration FLOAT,
    hip_right_fast_ratio FLOAT,
    hip_right_stationary_ratio FLOAT,
    hip_right_peak_interval FLOAT,
    knee_left_angle FLOAT,
    knee_left_angular_velocity FLOAT,
    knee_left_angular_acceleration FLOAT,
    knee_left_fast_ratio FLOAT,
    knee_left_stationary_ratio FLOAT,
    knee_left_peak_interval FLOAT,
    knee_right_angle FLOAT,
    knee_right_angular_velocity FLOAT,
    knee_right_angular_acceleration FLOAT,
    knee_right_fast_ratio FLOAT,
    knee_right_stationary_ratio FLOAT,
    knee_right_peak_interval FLOAT,
    torso_left_angle FLOAT,
    torso_left_angular_velocity FLOAT,
    torso_left_angular_acceleration FLOAT,
    torso_left_fast_ratio FLOAT,
    torso_left_stationary_ratio FLOAT,
    torso_left_peak_interval FLOAT,
    torso_right_angle FLOAT,
    torso_right_angular_velocity FLOAT,
    torso_right_angular_acceleration FLOAT,
    torso_right_fast_ratio FLOAT,
    torso_right_stationary_ratio FLOAT,
    torso_right_peak_interval FLOAT,
    spine_angle FLOAT,
    spine_angular_velocity FLOAT,
    spine_angular_acceleration FLOAT,
    spine_fast_ratio FLOAT,
    spine_stationary_ratio FLOAT,
    spine_peak_interval FLOAT,
    ankle_left_angle FLOAT,
    ankle_left_angular_velocity FLOAT,
    ankle_left_angular_acceleration FLOAT,
    ankle_right_angle FLOAT,
    ankle_right_angular_velocity FLOAT,
    ankle_right_angular_acceleration FLOAT,
    center_speed FLOAT,
    center_acceleration FLOAT,
    ankle_left_fast_ratio FLOAT,
    ankle_left_stationary_ratio FLOAT,
    ankle_left_peak_interval FLOAT,
    ankle_right_fast_ratio FLOAT,
    ankle_right_stationary_ratio FLOAT,
    ankle_right_peak_interval FLOAT,
    center_displacement FLOAT,
    center_velocity_change FLOAT,
    center_mean_speed FLOAT,
    center_mean_acceleration FLOAT,
    Label VARCHAR(1)
);


-- 4️⃣ 실시간 화면 테이블
CREATE TABLE IF NOT EXISTS realtime_screen (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    neck_angle FLOAT,
    neck_angular_velocity FLOAT,
    neck_angular_acceleration FLOAT,
    shoulder_balance_angle FLOAT,
    shoulder_balance_angular_velocity FLOAT,
    shoulder_balance_angular_acceleration FLOAT,
    shoulder_left_angle FLOAT,
    shoulder_left_angular_velocity FLOAT,
    shoulder_left_angular_acceleration FLOAT,
    shoulder_right_angle FLOAT,
    shoulder_right_angular_velocity FLOAT,
    shoulder_right_angular_acceleration FLOAT,
    elbow_left_angle FLOAT,
    elbow_left_angular_velocity FLOAT,
    elbow_left_angular_acceleration FLOAT,
    elbow_right_angle FLOAT,
    elbow_right_angular_velocity FLOAT,
    elbow_right_angular_acceleration FLOAT,
    hip_left_angle FLOAT,
    hip_left_angular_velocity FLOAT,
    hip_left_angular_acceleration FLOAT,
    hip_right_angle FLOAT,
    hip_right_angular_velocity FLOAT,
    hip_right_angular_acceleration FLOAT,
    knee_left_angle FLOAT,
    knee_left_angular_velocity FLOAT,
    knee_left_angular_acceleration FLOAT,
    knee_right_angle FLOAT,
    knee_right_angular_velocity FLOAT,
    knee_right_angular_acceleration FLOAT,
    torso_left_angle FLOAT,
    torso_left_angular_velocity FLOAT,
    torso_left_angular_acceleration FLOAT,
    torso_right_angle FLOAT,
    torso_right_angular_velocity FLOAT,
    torso_right_angular_acceleration FLOAT,
    spine_angle FLOAT,
    spine_angular_velocity FLOAT,
    spine_angular_acceleration FLOAT,
    ankle_left_angle FLOAT,
    ankle_left_angular_velocity FLOAT,
    ankle_left_angular_acceleration FLOAT,
    ankle_right_angle FLOAT,
    ankle_right_angular_velocity FLOAT,
    ankle_right_angular_acceleration FLOAT,
    center_speed FLOAT,
    center_acceleration FLOAT,
    center_displacement FLOAT,
    center_velocity_change FLOAT,
    center_mean_speed FLOAT,
    center_mean_acceleration FLOAT,
    Label TEXT,
    risk_score REAL,
    
    FOREIGN KEY(user_id) REFERENCES users(id)
);
''')

conn.commit()
conn.close()

"""

# AWS 버전
import pymysql
# ------------------------------
# 0️⃣ AWS RDS 연결 (DB 지정하지 않음)
# ------------------------------
conn = pymysql.connect(
    host="127.0.0.1",
    user="root",
    password="bear0205!@!@".encode('utf-8').decode('unicode_escape'),
    charset="utf8mb4",
    autocommit=True
)

cur = conn.cursor()
cur.execute("CREATE DATABASE IF NOT EXISTS capstone2 CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
cur.close()
conn.close()
print("✅ capstone2 데이터베이스 생성 완료!")

# ------------------------------
# 1️⃣ AWS RDS(MySQL) 연결 정보 입력
# ------------------------------
conn = pymysql.connect(
    host="127.0.0.1",                       # AWS RDS 엔드포인트
    port=3306,                               # 기본 포트
    user="root",                    # MySQL 사용자명
    password="bear0205!@!@".encode('utf-8').decode('unicode_escape'),
    database="capstone2",                    # 사용할 DB 이름
    charset="utf8mb4",
    autocommit=True
)

cur = conn.cursor()

# ------------------------------
# 2️⃣ 테이블 생성 스크립트 (모든 TEXT → VARCHAR)
# ------------------------------
sql_statements = [
    # 1️⃣ 사용자 테이블
    """
    CREATE TABLE IF NOT EXISTS users (
        id VARCHAR(50) PRIMARY KEY,
        password VARCHAR(255) NOT NULL,
        username VARCHAR(100) NOT NULL,
        phone_number BIGINT NOT NULL,
        non_guardian_name VARCHAR(255) NOT NULL,
        mail VARCHAR(255)
    ) ENGINE=InnoDB;
    """,

    # 2️⃣ 카메라 테이블
    """
    CREATE TABLE IF NOT EXISTS cameras (
        user_id VARCHAR(50) PRIMARY KEY,
        camera_url VARCHAR(255) NOT NULL,
        FOREIGN KEY (user_id) REFERENCES users(id)
    ) ENGINE=InnoDB;
    """,

    # 3️⃣ AI 학습 테이블
    """
    CREATE TABLE IF NOT EXISTS ai_learning (
        id INT AUTO_INCREMENT PRIMARY KEY,
        video VARCHAR(255) NOT NULL,
        file_id VARCHAR(255) NOT NULL,
        frame INT NOT NULL,
        timestamp FLOAT,
        neck_angle FLOAT,
        neck_angular_velocity FLOAT,
        neck_angular_acceleration FLOAT,
        neck_fast_ratio FLOAT,
        neck_stationary_ratio FLOAT,
        neck_peak_interval FLOAT,
        shoulder_balance_angle FLOAT,
        shoulder_balance_angular_velocity FLOAT,
        shoulder_balance_angular_acceleration FLOAT,
        shoulder_balance_fast_ratio FLOAT,
        shoulder_balance_stationary_ratio FLOAT,
        shoulder_balance_peak_interval FLOAT,
        shoulder_left_angle FLOAT,
        shoulder_left_angular_velocity FLOAT,
        shoulder_left_angular_acceleration FLOAT,
        shoulder_left_fast_ratio FLOAT,
        shoulder_left_stationary_ratio FLOAT,
        shoulder_left_peak_interval FLOAT,
        shoulder_right_angle FLOAT,
        shoulder_right_angular_velocity FLOAT,
        shoulder_right_angular_acceleration FLOAT,
        shoulder_right_fast_ratio FLOAT,
        shoulder_right_stationary_ratio FLOAT,
        shoulder_right_peak_interval FLOAT,
        elbow_left_angle FLOAT,
        elbow_left_angular_velocity FLOAT,
        elbow_left_angular_acceleration FLOAT,
        elbow_left_fast_ratio FLOAT,
        elbow_left_stationary_ratio FLOAT,
        elbow_left_peak_interval FLOAT,
        elbow_right_angle FLOAT,
        elbow_right_angular_velocity FLOAT,
        elbow_right_angular_acceleration FLOAT,
        elbow_right_fast_ratio FLOAT,
        elbow_right_stationary_ratio FLOAT,
        elbow_right_peak_interval FLOAT,
        hip_left_angle FLOAT,
        hip_left_angular_velocity FLOAT,
        hip_left_angular_acceleration FLOAT,
        hip_left_fast_ratio FLOAT,
        hip_left_stationary_ratio FLOAT,
        hip_left_peak_interval FLOAT,
        hip_right_angle FLOAT,
        hip_right_angular_velocity FLOAT,
        hip_right_angular_acceleration FLOAT,
        hip_right_fast_ratio FLOAT,
        hip_right_stationary_ratio FLOAT,
        hip_right_peak_interval FLOAT,
        knee_left_angle FLOAT,
        knee_left_angular_velocity FLOAT,
        knee_left_angular_acceleration FLOAT,
        knee_left_fast_ratio FLOAT,
        knee_left_stationary_ratio FLOAT,
        knee_left_peak_interval FLOAT,
        knee_right_angle FLOAT,
        knee_right_angular_velocity FLOAT,
        knee_right_angular_acceleration FLOAT,
        knee_right_fast_ratio FLOAT,
        knee_right_stationary_ratio FLOAT,
        knee_right_peak_interval FLOAT,
        torso_left_angle FLOAT,
        torso_left_angular_velocity FLOAT,
        torso_left_angular_acceleration FLOAT,
        torso_left_fast_ratio FLOAT,
        torso_left_stationary_ratio FLOAT,
        torso_left_peak_interval FLOAT,
        torso_right_angle FLOAT,
        torso_right_angular_velocity FLOAT,
        torso_right_angular_acceleration FLOAT,
        torso_right_fast_ratio FLOAT,
        torso_right_stationary_ratio FLOAT,
        torso_right_peak_interval FLOAT,
        spine_angle FLOAT,
        spine_angular_velocity FLOAT,
        spine_angular_acceleration FLOAT,
        spine_fast_ratio FLOAT,
        spine_stationary_ratio FLOAT,
        spine_peak_interval FLOAT,
        ankle_left_angle FLOAT,
        ankle_left_angular_velocity FLOAT,
        ankle_left_angular_acceleration FLOAT,
        ankle_right_angle FLOAT,
        ankle_right_angular_velocity FLOAT,
        ankle_right_angular_acceleration FLOAT,
        center_speed FLOAT,
        center_acceleration FLOAT,
        ankle_left_fast_ratio FLOAT,
        ankle_left_stationary_ratio FLOAT,
        ankle_left_peak_interval FLOAT,
        ankle_right_fast_ratio FLOAT,
        ankle_right_stationary_ratio FLOAT,
        ankle_right_peak_interval FLOAT,
        center_displacement FLOAT,
        center_velocity_change FLOAT,
        center_mean_speed FLOAT,
        center_mean_acceleration FLOAT,
        Label VARCHAR(10)
    ) ENGINE=InnoDB;
    """,

    # 4️⃣ 실시간 화면 테이블
    """
    CREATE TABLE IF NOT EXISTS realtime_screen (
        id VARCHAR(50) PRIMARY KEY,
        user_id VARCHAR(50),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        neck_angle FLOAT,
        neck_angular_velocity FLOAT,
        neck_angular_acceleration FLOAT,
        shoulder_balance_angle FLOAT,
        shoulder_balance_angular_velocity FLOAT,
        shoulder_balance_angular_acceleration FLOAT,
        shoulder_left_angle FLOAT,
        shoulder_left_angular_velocity FLOAT,
        shoulder_left_angular_acceleration FLOAT,
        shoulder_right_angle FLOAT,
        shoulder_right_angular_velocity FLOAT,
        shoulder_right_angular_acceleration FLOAT,
        elbow_left_angle FLOAT,
        elbow_left_angular_velocity FLOAT,
        elbow_left_angular_acceleration FLOAT,
        elbow_right_angle FLOAT,
        elbow_right_angular_velocity FLOAT,
        elbow_right_angular_acceleration FLOAT,
        hip_left_angle FLOAT,
        hip_left_angular_velocity FLOAT,
        hip_left_angular_acceleration FLOAT,
        hip_right_angle FLOAT,
        hip_right_angular_velocity FLOAT,
        hip_right_angular_acceleration FLOAT,
        knee_left_angle FLOAT,
        knee_left_angular_velocity FLOAT,
        knee_left_angular_acceleration FLOAT,
        knee_right_angle FLOAT,
        knee_right_angular_velocity FLOAT,
        knee_right_angular_acceleration FLOAT,
        torso_left_angle FLOAT,
        torso_left_angular_velocity FLOAT,
        torso_left_angular_acceleration FLOAT,
        torso_right_angle FLOAT,
        torso_right_angular_velocity FLOAT,
        torso_right_angular_acceleration FLOAT,
        spine_angle FLOAT,
        spine_angular_velocity FLOAT,
        spine_angular_acceleration FLOAT,
        ankle_left_angle FLOAT,
        ankle_left_angular_velocity FLOAT,
        ankle_left_angular_acceleration FLOAT,
        ankle_right_angle FLOAT,
        ankle_right_angular_velocity FLOAT,
        ankle_right_angular_acceleration FLOAT,
        center_speed FLOAT,
        center_acceleration FLOAT,
        center_displacement FLOAT,
        center_velocity_change FLOAT,
        center_mean_speed FLOAT,
        center_mean_acceleration FLOAT,
        Label VARCHAR(10),
        risk_score FLOAT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    ) ENGINE=InnoDB;
    """
]

# ------------------------------
# 실행
# ------------------------------
for stmt in sql_statements:
    cur.execute(stmt)

print("✅ All tables created successfully in MySQL!")

cur.close()
conn.close()