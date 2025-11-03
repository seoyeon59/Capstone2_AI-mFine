# 라이브러리 할당
import sqlite3
import pandas as pd
from sklearn.decomposition import PCA # 주성분 분석
from sklearn.preprocessing import StandardScaler # 정규화
from sklearn.tree import DecisionTreeClassifier # DT 모델
import joblib # 모델 저장용

# DB 연결
db_path = "capstone2.db"
conn = sqlite3.connect(db_path)

# 데이터 로드
query = "SELECT * FROM ai_learning"
df = pd.read_sql_query(query, conn)
conn.close()

# X, y 분할
feature_cols = [col for col in df.columns if (
    "angle" in col.lower() or
    "angular_velocity" in col.lower() or
    "angular_acceleration" in col.lower()
)]

X = df[feature_cols]
y = df["Label"]

# 정규화 실시
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 실시
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# DT 모델 학습
dt = DecisionTreeClassifier(
    max_depth=12,   # 제한 없으면 과적합 가능
    random_state=42,
    class_weight='balanced'  # 불균형 대응
)
dt.fit(X_pca, y)

# 모델 및 전처리 저장
joblib.dump(scaler, "pkl/scaler.pkl")
joblib.dump(pca, "pkl/pca.pkl")
joblib.dump(dt, "pkl/decision_tree_model.pkl")
