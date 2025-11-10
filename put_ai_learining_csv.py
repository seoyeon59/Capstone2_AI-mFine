import pandas as pd
from sqlalchemy import create_engine
import pymysql

# ------------------------------
# 1️⃣ DB 연결 정보
# ------------------------------
DB_USER = "root"
DB_PASSWORD = "bear0205!@!@"   # 특수문자 포함 가능
DB_HOST = "127.0.0.1"          # 또는 AWS RDS 엔드포인트
DB_PORT = 3306
DB_NAME = "capstone2"        # 위에서 만든 DB 이름

# ------------------------------
# 2️⃣ SQLAlchemy 엔진 생성
# ------------------------------
# 주의: 특수문자 포함 비밀번호는 URL 인코딩 필요
from urllib.parse import quote_plus
password_encoded = quote_plus(DB_PASSWORD)

engine = create_engine(f"mysql+pymysql://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4")

# ------------------------------
# 3️⃣ CSV 불러오기
# ------------------------------
csv_file = "Modeling_2.csv"
df = pd.read_csv(csv_file)

print(f"📊 CSV 로드 완료: {df.shape[0]}행 {df.shape[1]}열")

# ------------------------------
# 4️⃣ 데이터 삽입 (append 모드)
# ------------------------------
df.to_sql(name="ai_learning", con=engine, if_exists="append", index=False)

print("✅ CSV 데이터를 MySQL DB(ai_learning 테이블)에 성공적으로 삽입했습니다.")
