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
CREATE TABLE IF NOT EXISTS ai_learning (
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
    Label VARCHAR(1),

    PRIMARY KEY (video, file_id, frame)
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
# 1️⃣ AWS RDS(MySQL) 연결 정보 입력
# ------------------------------
conn = pymysql.connect(
    host="your-aws-endpoint.amazonaws.com",  # AWS RDS 엔드포인트
    port=3306,                               # 기본 포트
    user="your_username",                    # MySQL 사용자명
    password="your_password",                # 비밀번호
    database="capstone2",                    # 사용할 DB 이름
    charset="utf8mb4",
    autocommit=True
)

cur = conn.cursor()

# ------------------------------
# 2️⃣ 테이블 생성 스크립트 (모든 TEXT → VARCHAR)
# ------------------------------
sql_script = '''
-- 1️⃣ 사용자 테이블
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(50) PRIMARY KEY,
    password VARCHAR(255) NOT NULL,
    username VARCHAR(100) NOT NULL UNIQUE,
    phone_number INTEGER NOT NULL,
    non_guardian_name TEXT NOT NULL,
    mail TEXT   
) ENGINE=InnoDB;

-- 2️⃣ 카메라 테이블
CREATE TABLE IF NOT EXISTS cameras (
    user_id VARCHAR(50) PRIMARY KEY,
    camera_url VARCHAR(255) NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
) ENGINE=InnoDB;

-- 3️⃣ AI 학습 테이블
CREATE TABLE IF NOT EXISTS ai_learning (
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
    Label VARCHAR(10),

    PRIMARY KEY (video, file_id, frame)
) ENGINE=InnoDB;

-- 4️⃣ 실시간 화면 테이블
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
    Label VARCHAR(10),
    risk_score FLOAT,

    FOREIGN KEY(user_id) REFERENCES users(id)
) ENGINE=InnoDB;
'''

# ------------------------------
# 3️⃣ 실행
# ------------------------------
for statement in sql_script.split(';'):
    stmt = statement.strip()
    if stmt:
        cur.execute(stmt)

print("✅ All tables created successfully in AWS RDS (MySQL)!")

# ------------------------------
# 4️⃣ 연결 종료
# ------------------------------
cur.close()
conn.close()
"""