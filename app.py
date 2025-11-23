from flask import Flask, Response, render_template, request, redirect, session, jsonify
import cv2
import mediapipe as mp
import pymysql
import numpy as np
import threading
import time
from datetime import datetime
import pandas as pd
import joblib
from pykalman import KalmanFilter
import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine
import boto3
import io

# ==========================
# 1. 환경 설정 및 변수
# ==========================
app = Flask(__name__)
app.secret_key = os.urandom(24)  # 랜덤값으로 만들기(배포시 수정해야함)

# RDS 연결 정보 (환경 변수를 통해 안전하게 로드)
# EC2에 환경 변수 설정 필요: export DB_HOST='[RDS 엔드포인트]'
DB_HOST = os.environ.get('DB_HOST', 'swu-sw-02-db.cfoqwsiqgd5l.ap-northeast-2.rds.amazonaws.com')  # RDS 엔드포인트
DB_USER = os.environ.get('DB_USER', 'admin')  # RDS 마스터 사용자 이름
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'aimfine2!')  # RDS 마스터 암호
DB_NAME = os.environ.get('DB_NAME', 'capstone2')
DB_PORT = 3306

if not all([DB_HOST, DB_PASSWORD]):
    # 배포 환경에서 이 오류가 나면 환경 변수 설정이 안 된 것임
    print("FATAL ERROR: DB_HOST or DB_PASSWORD environment variables not set.")

# ==========================
# 2. S3에서 파일 로드
# ==========================
# S3 클라이언트 초기화 (EC2 IAM Role을 통해 자동 인증됨)
s3 = boto3.client('s3')
BUCKET_NAME = 'swu-sw-02-s3'  # 사용자님의 S3 버킷 이름

# 모델 로드
def load_model_from_s3(key_name):
    """S3에서 파일을 로드하여 joblib으로 디시리얼라이즈합니다."""
    # S3에서 파일을 객체로 가져옴 (BUCKET_NAME 변수 사용으로 개선)
    response = s3.get_object(Bucket=BUCKET_NAME, Key=key_name)
    # 객체의 Body(내용)를 읽어 메모리(BytesIO)에 저장
    model_data = io.BytesIO(response['Body'].read())
    # joblib을 사용하여 메모리에서 모델을 로드
    return joblib.load(model_data)

# S3애서 파일을 다운로드하여 로컬로 저장
def download_from_s3_to_local(key_name, local_path):
    """S3에서 파일을 로드하여 로컬 파일 시스템에 저장합니다."""
    try:
        s3.download_file(BUCKET_NAME, key_name, local_path)
        print(f"✅ S3 파일 '{key_name}'이 로컬 '{local_path}'에 다운로드되었습니다.")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to download '{key_name}' from S3. Error: {e}")
        return False

# 로컬 임시 파일 경로 설정
LOCAL_VIDEO_PATH = "/tmp/fall1.mp4" # /tmp는 EC2에서 쓰기 권한이 있는 임시 디렉토리

try:
    # S3에서 모델 파일 로드
    scaler = load_model_from_s3("scaler.pkl")
    model = load_model_from_s3("decision_tree_model.pkl")

    # 🔑 비디오 파일 로드 로직 수정
    if download_from_s3_to_local("fall1.mp4", LOCAL_VIDEO_PATH):
        video_source = LOCAL_VIDEO_PATH  # cv2.VideoCapture가 사용할 로컬 경로
    else:
        # 다운로드 실패 시 대체 경로 또는 에러 처리
        video_source = "static/fall1.mp4"  # (로컬 테스트용)
        print("⚠️ S3 비디오 파일 다운로드에 실패했습니다. 로컬 경로를 대체 사용합니다.")

    print("✅ AI Models loaded successfully from S3.")

except Exception as e:
    print(f"❌ ERROR: Failed to load models from S3. Check file names and S3 permissions. Error: {e}")


    # 모델 로드 실패 시 앱 실행 중단 방지를 위해 더미 객체 할당
    class DummyScaler:
        def transform(self, X): return X

        feature_names_in_ = []


    class DummyModel:
        def predict_proba(self, X): return np.array([[1.0, 0.0]])

        def predict(self, X): return np.array([0])


    scaler = DummyScaler()
    model = DummyModel()

# ==========================
# 3. DB 연결 및 엔진 설정 (RDS 엔드포인트 사용)
# ==========================

# SQLAlchemy 엔진 생성 (비밀번호를 URL-safe 인코딩하여 RDS 엔드포인트 사용)
password_encoded = quote_plus(DB_PASSWORD)
try:
    engine = create_engine(
        f"mysql+pymysql://{DB_USER}:{password_encoded}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4",
        pool_recycle=3600  # 1시간마다 연결 재활용 (DB 연결 끊김 방지)
    )
    print("✅ SQLAlchemy Engine configured with RDS endpoint.")
except Exception as e:
    print(f"❌ SQLAlchemy Engine configuration failed: {e}")
    engine = None


# DB 연결 함수 (pymysql을 사용하여 RDS 엔드포인트 사용)
def get_db_connection():
    try:
        conn = pymysql.connect(
            host=DB_HOST,  # RDS 엔드포인트
            port=DB_PORT,
            user=DB_USER,  # RDS 마스터 사용자
            password=DB_PASSWORD,  # 환경 변수에서 로드된 비밀번호
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except Exception as e:
        print(f"❌ DB Connection Error (check RDS host/security group): {e}")
        return None


# ==========================
# 4. MediaPipe 및 기타 로직 (변경 없음)
# ==========================

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

## 전역 변수 초기화
frame_idx = 0
latest_frame = None
frame_lock = threading.Lock()
current_user_id = None

# 카메라 연결 관련 전역 변수
cap = None  # 전역 카메라 객체
fps = 30 # 기본 FPS

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
            displacement, speed, acceleration, velocity_change = 0.0, 0.0, 0.0, 0.0

        # ✅ DB 스키마에 맞는 필드 구성
        centers.append({
            'center_displacement': displacement,
            'center_speed': speed,
            'center_acceleration': acceleration,
            'center_velocity_change': velocity_change,
            'center_mean_speed': speed,  # 단일 프레임이므로 mean 대신 현재값
            'center_mean_acceleration': acceleration
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
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return  # DB 연결 실패 시 종료

        with conn.cursor() as cursor:
            # 현재 시각 추가
            data_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # center_x/y/z 제거 (DB 컬럼에 없음)
            filtered_data = {
                k: v for k, v in data_dict.items()
                if k not in ['center_x', 'center_y', 'center_z']
            }

            # INSERT 실행 (MySQL에서는 ? → %s)
            columns = ', '.join(filtered_data.keys())
            placeholders = ', '.join(['%s'] * len(filtered_data))
            sql = f"INSERT INTO realtime_screen ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, tuple(filtered_data.values()))

            # user_id별 최대 600개 제한 (DB 자원 보호)
            user_id = filtered_data.get('user_id')
            if user_id:
                # Count 쿼리는 커서 재사용 가능 (단일 Connection 내)
                cursor.execute("SELECT COUNT(*) AS cnt FROM realtime_screen WHERE user_id = %s", (user_id,))
                count = cursor.fetchone()['cnt']

                if count > 600:
                    cursor.execute("""
                        DELETE FROM realtime_screen
                        WHERE user_id = %s
                        AND timestamp NOT IN (
                            SELECT t.timestamp FROM (
                                SELECT timestamp
                                FROM realtime_screen
                                WHERE user_id = %s
                                ORDER BY timestamp DESC
                                LIMIT 600
                            ) AS t
                        )
                    """, (user_id, user_id))

            conn.commit()
            print(f"✅ {user_id} 데이터 DB 저장 완료 ({len(filtered_data)}개 컬럼)")

    except Exception as e:
        print("❌ DB 저장 중 오류:", e)
    finally:
        if conn:
            conn.close()


# ------- 로컬 파일용 비디오 캡처 생성 -------
def get_video_capture(file_path):
    try:
        print(f"[INFO] 로컬 비디오 파일 연결 시도: {file_path}")
        # 로컬 파일 경로를 cv2.VideoCapture에 전달합니다.
        cap = cv2.VideoCapture(file_path)
        return cap
    except Exception as e:
        print(f"[ERROR] 비디오 캡처 생성 실패: {e}")
        return None


def connect_camera_loop():
    global cap, fps, current_user_id

    while True:
        try:
            # 이미 연결되어 있으면 프레임 속도에 맞춰 대기
            if cap is not None and cap.isOpened():
                time.sleep(1 / fps if fps > 0 else 0.03)
                continue

            # 비디오 캡처 시도 (로컬 파일 경로 사용)
            temp_cap = get_video_capture(video_source)
            if temp_cap and temp_cap.isOpened():
                cap = temp_cap
                # 파일의 경우 FPS가 0으로 나올 수 있으므로 기본값 설정
                fps_val = int(cap.get(cv2.CAP_PROP_FPS))
                fps = (fps_val if fps_val > 0 else 30)
                print(f"[INFO] 로컬 파일 연결 성공 (FPS: {fps})")
            else:
                print(f"[WARN] 로컬 파일 연결 실패. 경로 확인 필요: {video_source}. 5초 후 재시도")
                time.sleep(5)
                continue

            # 연결 성공 후, 프레임 읽기 스레드가 바로 작업을 시작할 수 있도록 대기
            time.sleep(1 / fps if fps > 0 else 0.03)

        except Exception as e:
            print(f"[ERROR] connect_camera_loop 예외 발생: {e}")
            time.sleep(1)

# ------ 프레임 읽기 스레드 ------
def capture_frames():
    global latest_frame, cap, frame_idx, fps, latest_score, latest_label
    print("[INFO] capture_frames 스레드 시작")

    fail_count = 0

    while True:
        # 🚨 로그인 상태에 따라 AI 분석 로직 실행 여부 결정 🚨
        if current_user_id is None:
            # 로그인되지 않은 경우, AI 분석을 건너뛰고 빈 프레임만 보여주거나 대기
            with frame_lock:
                # 스트리밍이 끊기지 않도록 빈 프레임을 유지 (선택 사항)
                latest_frame = empty_frame
            time.sleep(0.5) # CPU 사용량을 줄이기 위해 대기
            continue

        if cap is None or not cap.isOpened():
            # 카메라가 연결되지 않았고 로그인된 경우: 연결 대기
            with frame_lock:
                latest_frame = empty_frame
            time.sleep(0.2)
            continue

        try:
            # 로컬 파일 루프에 맞춰 프레임 읽기
            ret, frame = cap.read()

            if not ret or frame is None:
                fail_count += 1
                if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                    # 비디오 파일의 끝에 도달하면 0 프레임으로 되돌림 (루프)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("[INFO] 비디오 파일 루프 재시작")
                    fail_count = 0  # 재시작했으니 실패 횟수 초기화
                    time.sleep(0.01)
                    continue

            fail_count = 0

            # 프레임 리사이즈
            frame = cv2.resize(frame, (640, 480))

            # MediaPipe Pose 처리
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                row = {'frame': frame_idx}
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    row[f'kp{i}_x'] = lm.x
                    row[f'kp{i}_y'] = lm.y
                    row[f'kp{i}_z'] = lm.z
                    row[f'kp{i}_visibility'] = lm.visibility

                df = pd.DataFrame([row])
                center_df = compute_center_dynamics(df, fps=fps)
                center_info = center_df.iloc[-1].to_dict()

                keypoints = [f'kp{i}' for i in range(len(results.pose_landmarks.landmark))]
                df = smooth_with_kalman(df, keypoints)
                df = centralize_kp(df, pelvis_idx=(23, 24))
                df = scale_normalize_kp(df, ref_joints=(23, 24))

                row_processed = df.iloc[0].to_dict()
                calculated = calculate_angles(row_processed, fps=fps)
                calculated.update(center_info)

                try:
                    feature_cols = [col for col in calculated.keys() if (
                        "angle" in col.lower() or
                        "angular_velocity" in col.lower() or
                        "angular_acceleration" in col.lower() or
                        "center" in col.lower()
                    )]

                    X = pd.DataFrame([[calculated[col] for col in feature_cols]], columns=feature_cols).fillna(0.0)
                    X = X.reindex(columns=scaler.feature_names_in_, fill_value=0.0)

                    X_scaled = scaler.transform(X)
                    pred = model.predict_proba(X_scaled)
                    pred_label = model.predict(X_scaled)

                    score = float(pred[0][1] * 100)
                    label = int(pred_label[0])

                    calculated["risk_score"] = score
                    calculated["Label"] = label
                    latest_score = score
                    latest_label = "Fall" if label == 1 else "Normal"

                except Exception as e:
                    print("⚠️ 실시간 예측 오류:", e)
                    calculated["risk_score"] = 0.0
                    calculated["Label"] = 0

                # DB 저장
                save_to_db(calculated)

            # 최신 프레임 저장 (lock으로 보호)
            with frame_lock:
                latest_frame = frame.copy()
                frame_idx += 1

        except Exception as e:
            print(f"[ERROR] capture_frames 예외 발생: {e}")
            time.sleep(0.2)

        # FPS 제어: 너무 빠르면 CPU 과다, 너무 느리면 딜레이
        time.sleep(0.005)


# ------ Flask MJPEG 스트리밍 : 수정 제안 --------
empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
def gen_frames():
    global latest_frame
    while True:
        try:
            with frame_lock:
                frame = latest_frame if latest_frame is not None else empty_frame

                # 필요할 경우에만 복사 (안정성용)
                if frame is latest_frame:
                    frame = frame.copy()

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("[WARN] JPEG 인코딩 실패")
                time.sleep(0.05)
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # 너무 빠른 루프 방지 (CPU 보호)
            time.sleep(0.01)

        except Exception as e:
            print(f"[ERROR] gen_frames 예외 발생: {e}")
            time.sleep(0.005)

# ==========================
# Flask 라우팅
# ==========================
# 홈 (로그인 페이지)
@app.route('/')
def home():
    return render_template('login.html')

# ------ 로그인 기능 -------
@app.route('/login', methods=['POST'])
def login():
    global current_user_id
    user_id = request.form['id']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id=%s AND password=%s", (user_id, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['user_id'] = user_id
        current_user_id = user_id  # 스레드에서 사용 가능
        return redirect('/camera')
    else:
        # 로그인 실패 시 로그인 페이지 다시 렌더링 + 에러 메시지 전달
        return render_template('login.html', error_msg="아이디 또는 비밀번호를 확인하세요.")

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
        cursor.execute("SELECT id FROM users WHERE id = %s", (id,))
        if cursor.fetchone():  # 이미 존재하면
            return render_template('register.html', error_msg="이미 존재하는 아이디입니다.")

        # users 테이블에 삽입
        cursor.execute("""
            INSERT INTO users (id, password, username, phone_number, non_guardian_name, mail)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (id, password, username, phone_number, non_guardian_name, mail))

        # cameras 테이블에 삽입
        cursor.execute("""
            INSERT INTO cameras (user_id, camera_url)
            VALUES (%s, %s)
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
        cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
        if cursor.fetchone():
            exists = True
        conn.close()

    return jsonify({"exists": exists})

# ----- 실시간 화면 및 신고하는 페이지 ------
@app.route('/camera')
def index():
    # 사용자 ID는 로그인 상태 확인용으로만 남겨둡니다.
    user_id = session.get('user_id')

    # 로컬 파일 경로를 템플릿에 전달하여, 템플릿에서 참고할 수 있도록 합니다.
    camera_url = 'static/fall1.mp4'
    is_youtube = False  # 로컬 파일이므로 항상 False
    embed_url = None  # 임베드 URL 없음

    return render_template('camera.html',
                           camera_url=camera_url,
                           is_youtube=is_youtube,
                           embed_url=embed_url)
# ----- 실시간 화면 ------
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ----- 새로운 위험도 확인 라우트 ------
@app.route('/get_score')
def get_score():
    # 🔑 최근 N초 동안의 평균 위험 점수를 계산
    N_SECONDS = 2
    try:
        # 최근 1초 동안의 데이터를 모두 불러옴 (MySQL 문법)
        # TIMESTAMPADD(SECOND, -N_SECONDS, NOW())는 현재 시간으로부터 N초 전 시간을 의미
        query = f"""
                SELECT risk_score 
                FROM realtime_screen 
                WHERE timestamp >= TIMESTAMPADD(SECOND, -{N_SECONDS}, NOW())
                ORDER BY timestamp DESC
            """
        df = pd.read_sql_query(query, con=engine)

        if df.empty:
            # 최근 1초간 데이터가 없으면, 가장 최근의 데이터라도 가져옴
            df = pd.read_sql_query(
                "SELECT risk_score FROM realtime_screen ORDER BY timestamp DESC LIMIT 1",
                con=engine
            )

        if df.empty:
            avg_score = 0.0
        else:
            # 🔑 불러온 모든 점수의 평균을 계산
            avg_score = df['risk_score'].mean()

        return jsonify({"risk_score": round(avg_score, 2)})

    except Exception as e:
        print(f"❌ get_score 조회 오류: {e}")
        return jsonify({"risk_score": 0.0})

    # 추후에 주의/경고 알림 보내는 코드 추가 예정
    # 경고음 및 주의임 초기 알람 후 간격 시간
    # 주의 : 최조 주의 알람에서 10분 기준으로 알림 다시 발송
    # 경고 : 최조 경고 알람 (1번)

# ==========================
# 서버 실행 및 스레드 실행
# ==========================
if __name__ == "__main__":
    threading.Thread(target=connect_camera_loop, daemon=True).start()
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    # 배표시 변경 사항
    # app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False, threaded=True)