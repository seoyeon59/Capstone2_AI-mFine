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
from pykalman import KalmanFilter

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 랜덤값으로 만들기(배포시 수정해야함)

# AI 모델 로드
scaler = joblib.load("pkl/scaler.pkl")
model = joblib.load("pkl/decision_tree_model.pkl")

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

# 계산 처리용 전역 변수
prev_angles = {}  # 각도 저장
prev_angular_velocity = {}  # 각속도 저장
prev_center = None
prev_center_speed = 0.0

# 실시간 처리용 전역 변수
latest_score = 0.0
latest_label = "Normal"

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


# ----- 중심 이동/속도/각속도 계산 -----
def compute_center_dynamics(df, fps=30, left_pelvis='kp23', right_pelvis='kp24'):
    global prev_center, prev_center_speed
    centers = []

    for _, row in df.iterrows():
        try:
            center = np.array([
                (row[f'{left_pelvis}_x'] + row[f'{right_pelvis}_x']) / 2,
                (row[f'{left_pelvis}_y'] + row[f'{right_pelvis}_y']) / 2,
                (row[f'{left_pelvis}_z'] + row[f'{right_pelvis}_z']) / 2
            ])
        except KeyError:
            center = np.array([np.nan, np.nan, np.nan])

        # 초기화
        displacement = 0.0
        speed = 0.0
        acceleration = 0.0
        velocity_change = 0.0

        # 이전 프레임 대비 거리 변화량 계산
        if prev_center is not None:
            displacement = np.linalg.norm(center - prev_center)
            speed = displacement * fps
            acceleration = (speed - prev_center_speed) * fps
            velocity_change = abs(speed - prev_center_speed)
        else:
            displacement, speed, accel, velocity_change = 0.0, 0.0, 0.0, 0.0


        # ✅ DB 스키마에 맞는 필드 구성
        centers.append({
            'center_displacement': displacement,
            'center_speed': speed,
            'center_acceleration': accel,
            'center_velocity_change': velocity_change,
            'center_mean_speed': speed,  # 단일 프레임이므로 mean 대신 현재값
            'center_mean_acceleration': accel
        })

        # 이전값 업데이트
        prev_center = center
        prev_center_speed = speed

    return pd.DataFrame(centers)

# ----- 노이즈 제거 : Kalman ------
def smooth_with_kalman(df, keypoints):
    df_smooth = df.copy()
    for kp in keypoints:
        for axis in ['x', 'y', 'z']:
            col = f'{kp}_{axis}'
            if col not in df.columns:
                continue

            c = df[col].to_numpy()
            kf = KalmanFilter(initial_state_mean=[c[0], 0],
                              transition_matrices=[[1, 1], [0, 1]],
                              observation_matrices=[[1, 0]])
            state_means, _ = kf.filter(c)
            df_smooth[col] = state_means[:, 0]
    return df_smooth

# ----- 중심 정렬 ------
def centralize_kp(df, pelvis_idx=(23, 24)):
    df_central = df.copy()

    pelvis_x = (df[f'kp{pelvis_idx[0]}_x'] + df[f'kp{pelvis_idx[1]}_x']) / 2
    pelvis_y = (df[f'kp{pelvis_idx[0]}_y'] + df[f'kp{pelvis_idx[1]}_y']) / 2
    pelvis_z = (df[f'kp{pelvis_idx[0]}_z'] + df[f'kp{pelvis_idx[1]}_z']) / 2

    kp_x_cols = [c for c in df.columns if '_x' in c]
    kp_y_cols = [c for c in df.columns if '_y' in c]
    kp_z_cols = [c for c in df.columns if '_z' in c]

    for x_col, y_col, z_col in zip(kp_x_cols, kp_y_cols, kp_z_cols):
        df_central[x_col] -= pelvis_x
        df_central[y_col] -= pelvis_y
        df_central[z_col] -= pelvis_z

    return df_central

# ----- 스케일 정규화 -----
def scale_normalize_kp(df, ref_joints=(23, 24)):
    df_scaled = df.copy()
    left_x, left_y, left_z = df[f'kp{ref_joints[0]}_x'], df[f'kp{ref_joints[0]}_y'], df[f'kp{ref_joints[0]}_z']
    right_x, right_y, right_z = df[f'kp{ref_joints[1]}_x'], df[f'kp{ref_joints[1]}_y'], df[f'kp{ref_joints[1]}_z']

    scale = np.sqrt((left_x - right_x)**2 + (left_y - right_y)**2 + (left_z - right_z)**2)
    scale[scale == 0] = 1

    for col in df.columns:
        if any(s in col for s in ['_x', '_y', '_z']):
            df_scaled[col] = df[col] / scale

    return df_scaled

# ----- 각도 계산 -----
def compute_angle(a, b, c):
    """3점 좌표 a,b,c 기준 b를 꼭지점으로 하는 각도 계산"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# ----- 관절 각도/각속도/각가속도 계산 -----
def calculate_angles(row, fps=30):
    global prev_angles, prev_angular_velocity
    result = {}

    for joint_name, a_idx, b_idx, c_idx in joint_triplets:
        try:
            a = np.array([row[f'kp{a_idx}_x'], row[f'kp{a_idx}_y'], row[f'kp{a_idx}_z']])
            b = np.array([row[f'kp{b_idx}_x'], row[f'kp{b_idx}_y'], row[f'kp{b_idx}_z']])
            c = np.array([row[f'kp{c_idx}_x'], row[f'kp{c_idx}_y'], row[f'kp{c_idx}_z']])

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

# ----- DB 저장 함수(실시간 + 10분 후 삭제) -----
def save_to_db(data_dict):
    try:
        # SQLite 연결
        conn = sqlite3.connect('capstone2.db')
        cursor = conn.cursor()

        # 현재 시각 추가
        data_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # center_x/y/z 제거 (DB 컬럼에 없음)
        filtered_data = {
            k: v for k, v in data_dict.items()
            if k not in ['center_x', 'center_y', 'center_z']
        }

        # 딕셔너리 키/값을 SQL에 삽입
        columns = ', '.join(data_dict.keys())
        placeholders = ', '.join(['?'] * len(data_dict))
        sql = f"INSERT INTO realtime_screen ({columns}) VALUES ({placeholders})"
        cursor.execute(sql, tuple(data_dict.values()))

        # 10분 이상 지난 데이터 삭제 (로컬 타임 기준
        cursor.execute("DELETE FROM realtime_screen WHERE timestamp < datetime('now', 'localtime', '-10 minutes')")

        conn.commit()

    except Exception as e:
        print("DB 저장 중 오류:", e)

    finally:
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

# ------ 프레임 읽기 스레드 ------
def capture_frames():
    global latest_frame, cap, frame_idx, fps, latest_score, latest_label
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
                    # ----- 관절 좌표 추출-----
                    row = {'frame': frame_idx}
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        row[f'kp{i}_x'] = lm.x
                        row[f'kp{i}_y'] = lm.y
                        row[f'kp{i}_z'] = lm.z
                        row[f'kp{i}_visibility'] = lm.visibility

                    # 한 프레임을 DataFrame 형태로 변환
                    df = pd.DataFrame([row])

                    # 중심 이동/속도 계산
                    center_df = compute_center_dynamics(df, fps=fps)
                    center_info = center_df.iloc[-1].to_dict()

                    # 칼만 필터로 노이즈 제거
                    keypoints = [f'kp{i}' for i in range(len(results.pose_landmarks.landmark))]
                    df = smooth_with_kalman(df, keypoints)

                    # 중심 정렬
                    df = centralize_kp(df, pelvis_idx=(23, 24))

                    # 스케일 정규화
                    df = scale_normalize_kp(df, ref_joints=(23, 24))

                    # 각도/각속도/각가속도 계산
                    row_processed = df.iloc[0].to_dict()
                    calculated = calculate_angles(row_processed, fps=fps)

                    # 중심 이동 정보 병합
                    calculated.update(center_info)

                    # AI 모델로 예측 수행
                    try:
                        # feature 선택
                        feature_cols = [col for col in calculated.keys() if (
                                "angle" in col.lower() or
                                "angular_velocity" in col.lower() or
                                "angular_acceleration" in col.lower() or
                                "center" in col.lower()
                        )]

                        X = pd.DataFrame([[calculated[col] for col in feature_cols]], columns=feature_cols)
                        X = X.fillna(0.0)

                        # ✅ scaler가 학습할 때 사용한 피처 순서대로 재정렬
                        X = X.reindex(columns=scaler.feature_names_in_, fill_value=0.0)

                        # 전처리 + 예측
                        X_scaled = scaler.transform(X)
                        pred = model.predict_proba(X_scaled)
                        pred_label = model.predict(X_scaled)

                        # 예측 결과 반영
                        score = float(pred[0][1] * 100)
                        label = int(pred_label[0])

                        calculated["risk_score"] = score
                        calculated["Label"] = label

                        # 화면 표시용 전역변수 업데이트
                        latest_score = score
                        latest_label = "Fall" if label == 1 else "Normal"

                    except Exception as e:
                        print("⚠️ 실시간 예측 오류:", e)
                        calculated["risk_score"] = 0.0
                        calculated["Label"] = 0

                    # DB 저장
                    save_to_db(calculated) # DB 저장

        # 최신 프레임 저장
        with frame_lock:
            latest_frame = frame.copy()
            frame_idx += 1

        # FPS 제어
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
# 홈 (로그인 페이지)
@app.route('/')
def home():
    return render_template('login.html')

# ------ 로그인 기능 -------
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

# ----- 회원가입 기능 ------
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

        # 서버 측 아이디 중복 체크
        cursor.execute("SELECT id FROM users WHERE id = ?", (id,))
        if cursor.fetchone():  # 이미 존재하면
            conn.close()
            return render_template('register.html', error_msg="이미 존재하는 아이디입니다.")

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

# ------ 아이디어 중복 체크 확인 -------
@app.route('/check_id')
def check_id():
    user_id = request.args.get('id')
    exists = False

    if user_id:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
        if cursor.fetchone():
            exists = True
        conn.close()

    return jsonify({"exists": exists})

# ----- 실시간 화면 및 신고하는 페이지 ------
@app.route('/camera')
def index():
    return render_template('camera.html')

# ----- 실시간 화면 ------
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ----- 새로운 위험도 확인 라우트 ------
@app.route('/get_score')
def get_score():
    conn = sqlite3.connect('capstone2.db')
    df = pd.read_sql_query("SELECT risk_score FROM realtime_screen ORDER BY timestamp DESC LIMIT 1", conn)
    conn.close()

    if df.empty:
        return jsonify({"risk_score": 0.0})  # 데이터 없으면 0 반환

    return jsonify({"risk_score": round(df['risk_score'].iloc[0], 2)})

    # 추후에 주의/경고 알림 보내는 코드 추가 예정

# ----- 낙상 위험 점수 기반 알림 로직 추가 ------
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
