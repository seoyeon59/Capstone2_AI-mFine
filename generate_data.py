import sqlite3
import random
import time
import uuid  # 고유 ID 생성용

conn = sqlite3.connect('capstone2.db')
c = conn.cursor()

# ==========================
# 테이블 초기화 (내용 삭제)
# ==========================
c.execute("DELETE FROM realtime_screen")  # 전체 데이터 삭제
conn.commit()
print("[INFO] realtime_screen 테이블 내용 삭제 완료 ✅")


# 사용자 ID
user_id = "test"

try:
    while True:
        frame_id = str(uuid.uuid4())
        fall_flag = random.choice([0, 1])
        risk_score = round(random.uniform(0, 100), 1)

        c.execute("""
        INSERT INTO realtime_screen (id, user_id, fall_flag, risk_score)
        VALUES (?, ?, ?, ?)
        """, (frame_id, user_id, fall_flag, risk_score))

        conn.commit()
        print(f"[INFO] frame {frame_id} 삽입 완료 | 위험 점수: {risk_score} | 낙상: {fall_flag}")

        time.sleep(1)
except KeyboardInterrupt:
    print("실시간 데이터 삽입 종료")
finally:
    conn.close()
