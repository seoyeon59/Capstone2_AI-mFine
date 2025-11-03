from flask import Flask, Response, render_template, request, redirect, session, jsonify
import cv2
import mediapipe as mp
import sqlite3
import numpy as np
import threading
import time
from datetime import datetime
from playsound import playsound  # pip install playsound==1.2.2
import os
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 랜덤값으로 만들기(배포시 수정해야함)

# SQLite 연결
DB_PATH = 'capstone2.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 컬럼명을 dict처럼 사용 가능
    return conn

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 전역 변수 초기화
frame_idx = 0
latest_frame = None
frame_lock = threading.Lock()
prev_angles = {}  # 각도 저장
prev_angular_velocity = {}  # 각속도 저장

# 관절 트리플 (a,b,c)
joint_triplets = [
    ('neck', 0, 11, 12),
    ('shoulder_balance', 11, 0, 12),
    ('shoulder_left', 23, 11, 13),
    ('shoulder_right', 24, 12, 14),
    ('elbow_left', 11, 13, 15),
    ('elbow_right', 12, 14, 16),
    ('hip_left', 11, 23, 25),
    ('hip_right', 12, 24, 26),
    ('knee_left', 23, 25, 27),
    ('knee_right', 24, 26, 28),
    ('ankle_left', 25, 27, 31),
    ('ankle_right', 26, 28, 32),
    ('torso_left', 0, 11, 23),
    ('torso_right', 0, 12, 24),
    ('spine', 0, 23, 24),
]

# ==============================
# 실시간 화면 표시와 관련된 함수
# ==============================
def compute_angle(a, b, c):
    """3점 좌표 a,b,c 기준 b를 꼭지점으로 하는 각도 계산"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# 관절 좌표 -> 각도, 각속도, 각가속도 계산
def calculate_angles(row, fps=30):
    """
    row: dict {관절_x, 관절_y, 관절_z, 관절_v}
    fps: 프레임 속도
    return: dict {각도, 각속도, 각가속도}
    """
    result = {}
    for joint_name, a_idx, b_idx, c_idx in joint_triplets:
        try:
            a = np.array([row[f'{a_idx}_x'], row[f'{a_idx}_y'], row[f'{a_idx}_z']])
            b = np.array([row[f'{b_idx}_x'], row[f'{b_idx}_y'], row[f'{b_idx}_z']])
            c = np.array([row[f'{c_idx}_x'], row[f'{c_idx}_y'], row[f'{c_idx}_z']])

            # 각도
            angle = compute_angle(a, b, c)
            result[f'{joint_name}_angle'] = angle

            # 각속도
            prev_angle = prev_angles.get(f'{joint_name}_angle', angle)
            angular_velocity = (angle - prev_angle) * fps
            result[f'{joint_name}_angular_velocity'] = angular_velocity

            # 각가속도
            prev_vel = prev_angular_velocity.get(f'{joint_name}_angular_velocity', angular_velocity)
            angular_acceleration = (angular_velocity - prev_vel) * fps
            result[f'{joint_name}_angular_acceleration'] = angular_acceleration

            # 이전 값 업데이트
            prev_angles[f'{joint_name}_angle'] = angle
            prev_angular_velocity[f'{joint_name}_angular_velocity'] = angular_velocity

        except KeyError:
            # 좌표 없는 경우 0으로 초기화
            result[f'{joint_name}_angle'] = 0.0
            result[f'{joint_name}_angular_velocity'] = 0.0
            result[f'{joint_name}_angular_acceleration'] = 0.0

    return result

# 관절 각도, 각속도, 각가속도 관련 내용 DB 저장 함수 (실시간 + 10분 후 삭제)
def save_to_db(data_dict):
    conn = sqlite3.connect('capstone2.db')
    cursor = conn.cursor()

    # timestamp 포함
    data_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    columns = ', '.join(data_dict.keys())
    placeholders = ', '.join(['?'] * len(data_dict))
    sql = f"INSERT INTO realtime_screen ({columns}) VALUES ({placeholders})"
    cursor.execute(sql, tuple(data_dict.values()))

    # 10분 이상 지난 데이터 삭제
    cursor.execute("DELETE FROM realtime_screen WHERE timestamp < datetime('now', '-10 minutes')")

    conn.commit()
    conn.close()

# DB에서 camera_url 가져오기
def get_camera_url(user_id="test"):
    conn = sqlite3.connect('capstone2.db')
    c = conn.cursor()
    c.execute("SELECT camera_url FROM cameras WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    else:
        return None

# IP 웹캠 연결 반복 시도
 # 로그인한 id의 웹캠 불러오기
cap = None  # 전역 카메라 객체
fps = 30 # 기본 FPS

def connect_camera_loop():
    global cap, fps
    while True:
        if cap is None or not cap.isOpened():
            ip_url = get_camera_url("test")
            if ip_url:
                temp_cap = cv2.VideoCapture(ip_url)
                if temp_cap.isOpened():
                    cap = temp_cap
                    fps_val = int(cap.get(cv2.CAP_PROP_FPS))
                    fps = fps_val if fps_val > 0 else 30
                    print("[INFO] IP 웹캠 연결 성공")
                else:
                    print("[WARN] IP 웹캠 연결 실패, 5초 후 재시도")
                    temp_cap.release()
            else:
                print("[WARN] 로그인 유저 ID 없음 또는 camera_url 없음, 3초 후 재시도")
        time.sleep(3)

# 프레임 읽기 스레드
def capture_frames():
    global latest_frame, cap, frame_idx, fps
    while True:
        if cap is None or not cap.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (640, 480))

                # MediaPipe 처리
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                if results.pose_landmarks:
                    # 관절 좌표 추출
                    row = {'frame': frame_idx}
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        row[f"x_{i}"] = lm.x
                        row[f"y_{i}"] = lm.y
                        row[f"z_{i}"] = lm.z
                        row[f"v_{i}"] = lm.visibility

                    calculated = calculate_angles(row, fps=fps) # 각도/각속도/각가속도 계산
                    save_to_db(calculated) # DB 저장

        # 최신 프레임 저장
        with frame_lock:
            latest_frame = frame.copy()
            frame_idx += 1

        time.sleep(1 / fps if fps > 0 else 1 / 30)

# Flask MJPEG 스트리밍
def gen_frames():
    global latest_frame
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# =========================
# 스레드 시작
# =========================
threading.Thread(target=connect_camera_loop, daemon=True).start()
threading.Thread(target=capture_frames, daemon=True).start()

# ==========================
# Flask 라우팅
# ==========================
# 홈 (로그인 페이지)
@app.route('/')
def home():
    return render_template('login.html')

# 로그인 기능
@app.route('/login', methods=['POST'])
def login():
    user_id = request.form['id']   # id 입력
    password = request.form['password'] # passord 입력

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id=? AND password=?", (user_id, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['user_id'] = user_id
        return redirect('/camera')
    else:
        return "이름 또는 비밀번호를 확인하세요."

# 회원가입 기능
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        id = request.form['id']
        password = request.form['password']
        username = request.form['username']
        phone_number = request.form['phone_number']
        non_guardian_name = request.form['non_guardian_name']
        mail = request.form['mail']
        camera_url = request.form['camera_url']  # cameras.camera_url

        conn = get_db_connection()
        cursor = conn.cursor()

        # users 테이블에 삽입
        cursor.execute("""
            INSERT INTO users (id, password, username, phone_number, non_guardian_name, mail)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (id, password, username, phone_number, non_guardian_name, mail))

        # camera 테이블에 삽입
        cursor.execute("""
            INSERT INTO cameras (user_id, camera_url)
            VALUES (?, ?)
        """, (id, camera_url))

        conn.commit()
        conn.close()

        return redirect('/')

    return render_template('register.html')

# 카메라
@app.route('/camera')
def index():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 모델, 전처리기 미리 불러오기 (Flask 앱 시작 시 한 번)
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
model = joblib.load("decision_tree_model.pkl")

# 새로운 위험도 확인 라우트 (수정 필요 : 카메라 연결 후 점수 나오게 실행)
@app.route('/get_score')
def get_score():
    conn = sqlite3.connect('capstone2.db')
    df = pd.read_sql_query("SELECT * FROM realtime_screen ORDER BY timestamp DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({"risk_score": 0.0})  # 데이터 없으면 0 반환

    # feature 선택
    feature_cols = [col for col in df.columns if (
        "angle" in col.lower() or
        "angular_velocity" in col.lower() or
        "angular_acceleration" in col.lower()
    )]
    X = df[feature_cols]

    # NaN 처리
    X = X.fillna(0.0)

    # 전처리 + PCA + 예측
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    pred = model.predict_proba(X_pca)
    pred_label = model.predict(X_pca)

    # 예측 결과를 위험 점수로 변환
    score = pred[0][1] * 100
    label = int(pred_label[0])  # 0: 정상, 1: 낙상

    # DB에 저장
    conn = sqlite3.connect('capstone2.db')
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE realtime_screen
        SET Label = ?, risk_score = ?
        WHERE timestamp = ?
    """, (label, score, df['timestamp'].iloc[0]))
    conn.commit()
    conn.close()

    return jsonify({"risk_score": score})

    # 추후에 주의/경고 알림 보내는 코드 추가 예정


# 낙상 위험 점수 기반 알림 로직 추가
def play_alarm_sound():
    """🔊 서버 스피커에서 경고음 재생"""
    try:
        playsound("static/alarmclockbeepsaif.mp3")
        print("🔊 Alarm sound played!")
    except Exception as e:
        print(f"❌ Alarm Sound Error: {e}")

# ==========================
# 서버 실행
# ==========================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
