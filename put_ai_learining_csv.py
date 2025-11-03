import sqlite3
import pandas as pd

# CSV 파일 불러오기
csv_file = "Modeling_2.csv"  # CSV 파일 경로
df = pd.read_csv(csv_file)

# DB 연결
db_path = "capstone2.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# CSV 데이터를 DB에 삽입
df.to_sql("ai_learning", conn, if_exists="append", index=False)

# 연결 종료
conn.close()

print("✅ CSV 데이터를 DB에 넣었습니다.")
