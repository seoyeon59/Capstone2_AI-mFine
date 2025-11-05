# 라이브러리 할당
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler # 정규화
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier # DT 모델
from sklearn.metrics import accuracy_score
import joblib # 모델 저장용


# DB 연결
db_path = "capstone2.db"
conn = sqlite3.connect(db_path)

# 데이터 로드
query = "SELECT * FROM ai_learning ORDER BY id"
df = pd.read_sql_query(query, conn)
conn.close()


# df = pd.read_csv("Modeling_2.csv")

# X, y 분할
feature_cols = [col for col in df.columns if (
    "angle" in col.lower() or
    "angular_velocity" in col.lower() or
    "angular_acceleration" in col.lower() or
    "center" in col.lower()
)]

X = df[feature_cols]
y = df["Label"]

'''
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )
'''

# 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y)

dt = DecisionTreeClassifier(
    max_depth=12,
    random_state=42,
    class_weight='balanced'
)

dt.fit(X_train_res, y_train_res)

y_train_pred = dt.predict(X_train_res)

train_acc = accuracy_score(y_train_res, y_train_pred)

print(f"✅ Train Accuracy: {train_acc:.4f}")


# 모델 및 전처리 저장
joblib.dump(scaler, "pkl/scaler.pkl")
joblib.dump(dt, "pkl/decision_tree_model.pkl")
