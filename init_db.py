import sqlite3

conn = sqlite3.connect('capstone2.db')
c = conn.cursor()

c.executescript('''
-- 1️⃣ 사용자 테이블
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(50) PRIMARY KEY,
    password TEXT NOT NULL,
    username TEXT NOT NULL UNIQUE,
    observer_name TEXT
);

-- 2️⃣ 카메라 테이블
CREATE TABLE IF NOT EXISTS cameras (
    user_id VARCHAR(50) PRIMARY KEY,
    camera_url TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 3️⃣ AI 학습 테이블
CREATE TABLE IF NOT EXISTS ai_learning (
    id VARCHAR(50) PRIMARY KEY
    -- 추후 학습 컬럼 추가 가능
);

-- 4️⃣ 실시간 화면 테이블
CREATE TABLE IF NOT EXISTS realtime_screen (
    id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    fall_flag INTEGER,
    risk_score REAL,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
''')

conn.commit()
conn.close()
