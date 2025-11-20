import yt_dlp
from flask import Flask, Response, render_template, request, redirect, session, jsonify
import cv2
import mediapipe as mp
import pymysql
import numpy as np
import threading
import time
import os
import io
import pandas as pd
import joblib
import boto3
from pykalman import KalmanFilter
# from playsound import playsound
from urllib.parse import urlparse, parse_qs, quote_plus
from sqlalchemy import create_engine
from datetime import datetime

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
# 2. AI 모델 로드 (S3에서)
# ==========================
# S3 클라이언트 초기화 (EC2 IAM Role을 통해 자동 인증됨)
s3 = boto3.client('s3')
BUCKET_NAME = 'swu-sw-02-s3'  # 사용자님의 S3 버킷 이름


def load_model_from_s3(key_name):
    """S3에서 파일을 로드하여 joblib으로 디시리얼라이즈합니다."""
    # S3에서 파일을 객체로 가져옴 (BUCKET_NAME 변수 사용으로 개선)
    response = s3.get_object(Bucket=BUCKET_NAME, Key=key_name)
    # 객체의 Body(내용)를 읽어 메모리(BytesIO)에 저장
    model_data = io.BytesIO(response['Body'].read())
    # joblib을 사용하여 메모리에서 모델을 로드
    return joblib.load(model_data)


try:
    # S3에서 모델 파일 로드
    scaler = load_model_from_s3("scaler.pkl")
    model = load_model_from_s3("decision_tree_model.pkl")
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
fps = 30  # 기본 FPS

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

    scale = np.sqrt((left_x - right_x) ** 2 + (left_y - right_y) ** 2 + (left_z - right_z) ** 2)
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


# ------- DB에서 camera_url 가져오기 -------
def get_camera_url(user_id):
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        cursor = conn.cursor()
        # camera 테이블에서 user_id에 해당하는 camera_url 조회
        cursor.execute("SELECT camera_url FROM cameras WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and 'camera_url' in row:
            return row['camera_url']
        return None
    except Exception as e:
        print(f"⚠️ 카메라 URL 조회 오류: {e}")
        return None
    finally:
        if conn:
            conn.close()


# ------- IP/유튜브 구분 및 카메라 연결 -------
ydl_opts = {
    "format": "bestvideo[ext=mp4]+bestaudio/best",
    "quiet": True,
    "noplaylist": True,
    "live_from_start": False
}


def get_youtube_direct_url(youtube_url):
    """yt-dlp를 사용하여 YouTube 영상의 직접 스트리밍 URL을 추출합니다."""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # download=False로 설정하여 정보만 추출
            info = ydl.extract_info(youtube_url, download=False)
            # OpenCV VideoCapture에 넣을 수 있는 URL (스트리밍 URL) 반환
            return info['url']
    except Exception as e:
        # 추출 실패 시 오류 메시지 출력 후 None 반환
        print(f"[ERROR] yt-dlp direct URL extraction failed: {e}")
        return None


def get_video_capture(url):
    """주어진 URL 또는 ID를 기반으로 cv2.VideoCapture 객체를 반환합니다."""
    # 1. URL이 정수(로컬 웹캠)인 경우 분리 처리
    if isinstance(url, int):
        print("[INFO] 로컬 웹캠 연결 시도 중...")
        cap = cv2.VideoCapture(url)  # cv2.VideoCapture(0) 실행
        # 로컬 웹캠은 초기화에 시간이 걸릴 수 있음
        if cap.isOpened():
            return cap
        else:
            # 로컬 웹캠은 EC2에서 항상 실패하므로 명확한 오류 로그 출력
            print("[ERROR] 로컬 웹캠 연결 실패. EC2 환경에서는 웹캠이 존재하지 않습니다.")
            return None

    # 2. URL이 문자열이고 유튜브인 경우
    # 🚨 [수정 1] URL 검사 로직 오류 수정 및 디버깅 로그 추가
    if isinstance(url, str) and ("youtube.com" in url or "youtu.be" in url):
        print("[INFO] YouTube 영상 direct URL 추출 중...")
        try:
            direct_url = get_youtube_direct_url(url)

            if not direct_url:
                print("[ERROR] yt-dlp: direct_url 추출 실패로 VideoCapture 시도 불가.")
                return None

            # 디버깅을 위해 추출된 URL 출력 (길이가 길 수 있으므로 50자만 출력)
            print(f"[INFO] YouTube direct stream URL (extracted): {direct_url[:50]}...")

            # 추출된 direct_url로 VideoCapture 시도
            cap = cv2.VideoCapture(direct_url)

            # 🚨 [수정 2] VideoCapture 성공 여부 즉시 검사
            if not cap.isOpened():
                print(f"[ERROR] cv2.VideoCapture({url})로 스트림 열기 실패. 추출 URL: {direct_url[:50]}...")
                cap.release()
                return None

            return cap
        except Exception as e:
            print(f"[ERROR] YouTube direct stream load error: {e}")
            return None

    # 3. URL이 문자열이고 IP 카메라인 경우
    elif isinstance(url, str):
        print("[INFO] IP 카메라 연결 중...")
        cap = cv2.VideoCapture(url)
        # IP 카메라도 연결 성공 여부 검사
        if not cap.isOpened():
            print(f"[ERROR] IP 카메라 스트림 ({url}) 열기 실패.")
            return None
        return cap

    return None  # 유효하지 않은 URL 타입


# ------ IP 웹캠 연결 반복 시도 -------
def connect_camera_loop():
    global cap, fps, current_user_id

    # 기본 테스트용 카메라 URL (없을 경우 로컬 웹캠 사용)
    default_url = 0  # 로컬 웹캠 (IP캠이 없을 때 대체)
    print("[INFO] connect_camera_loop 시작됨")

    while True:
        try:
            # 이미 연결되어 있으면 패스
            if cap is not None and cap.isOpened():
                time.sleep(1)
                continue

            # 현재 로그인 유저 확인
            url = None
            if current_user_id:
                # DB에서 현재 사용자 ID의 카메라 URL 조회
                url = get_camera_url(current_user_id)
                print(f"[DEBUG] 로그인된 사용자({current_user_id})의 URL: {url}")

            # 로그인되어 있지 않거나 URL이 잘못된 경우 → 기본 카메라로 시도
            if not url or not isinstance(url, str) or not url.strip():
                print("[INFO] 로그인 안됨 또는 유효한 URL 없음 → 기본 카메라 연결 시도")
                url = default_url

            # 비디오 캡처 시도
            temp_cap = get_video_capture(url)
            if temp_cap and temp_cap.isOpened():
                cap = temp_cap
                # 실제 FPS 값을 가져와서 설정 (대부분의 웹캠/IP캠은 30)
                fps_val = int(cap.get(cv2.CAP_PROP_FPS))
                fps = fps_val if fps_val > 0 else 30
                # 🚨 [수정 3] 로그에 어떤 URL로 연결 성공했는지 표시
                print(f"[INFO] 카메라 연결 성공 (FPS: {fps}) - Source URL: {url}")
            else:
                # 🚨 [수정 3] 로그에 어떤 URL로 연결 실패했는지 표시
                print(f"[WARN] 카메라 연결 실패 (URL: {url}), 3초 후 재시도")
                time.sleep(3)
                continue

            # 캡처 루프가 끊기지 않도록 대기
            time.sleep(1)

        except Exception as e:
            print(f"[ERROR] connect_camera_loop 예외 발생: {e}")
            time.sleep(1)


# ------ 프레임 읽기 스레드 ------
def capture_frames():
    global latest_frame, cap, frame_idx, fps, latest_score, latest_label
    print("[INFO] capture_frames 스레드 시작")

    fail_count = 0

    while True:
        # 카메라 연결 상태 확인 및 대기
        if cap is None or not cap.isOpened():
            # 빈 프레임 생성 후 대기
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # 텍스트 오버레이 (연결 대기)
            cv2.putText(frame, "Waiting for Camera Connection...", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            with frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.2)
            continue

        try:
            # 프레임 읽기
            ret, frame = cap.read()

            if not ret or frame is None:
                fail_count += 1
                print(f"[WARN] 프레임 읽기 실패 ({fail_count})")
                if fail_count > 10:
                    print("[ERROR] 스트림이 끊긴 것으로 판단, 재연결 시도 예정")
                    cap.release()
                    cap = None  # None으로 설정하여 connect_camera_loop가 재시도하도록 유도
                time.sleep(0.1)
                continue
            fail_count = 0  # 성공 시 카운트 리셋

            # 프레임 리사이즈
            frame = cv2.resize(frame, (640, 480))

            # MediaPipe Pose 처리
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            calculated = {}
            if results.pose_landmarks:
                row = {'frame': frame_idx}
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    # 랜드마크 좌표 추출 (MediaPipe는 0~1 사이의 정규화된 좌표를 반환)
                    row[f'kp{i}_x'] = lm.x
                    row[f'kp{i}_y'] = lm.y
                    row[f'kp{i}_z'] = lm.z
                    row[f'kp{i}_visibility'] = lm.visibility

                df = pd.DataFrame([row])

                # 중심 동역학 계산
                center_df = compute_center_dynamics(df, fps=fps)
                center_info = center_df.iloc[-1].to_dict()

                # 데이터 전처리
                keypoints = [f'kp{i}' for i in range(len(results.pose_landmarks.landmark))]
                df = smooth_with_kalman(df, keypoints)  # 칼만 필터
                df = centralize_kp(df, pelvis_idx=(23, 24))  # 중심 정렬
                df = scale_normalize_kp(df, ref_joints=(23, 24))  # 스케일 정규화

                row_processed = df.iloc[0].to_dict()
                calculated = calculate_angles(row_processed, fps=fps)
                calculated.update(center_info)

                try:
                    # AI 예측을 위한 피처 추출 및 준비
                    feature_cols = [col for col in calculated.keys() if (
                            "angle" in col.lower() or
                            "angular_velocity" in col.lower() or
                            "angular_acceleration" in col.lower() or
                            "center" in col.lower()
                    )]

                    X = pd.DataFrame([[calculated[col] for col in feature_cols]], columns=feature_cols).fillna(0.0)

                    # 로드된 스케일러의 피처 순서에 맞춰 데이터 정렬 및 누락된 피처 0으로 채우기
                    if hasattr(scaler, 'feature_names_in_'):
                        X = X.reindex(columns=scaler.feature_names_in_, fill_value=0.0)

                    X_scaled = scaler.transform(X)
                    pred = model.predict_proba(X_scaled)  # 확률 예측
                    pred_label = model.predict(X_scaled)  # 레이블 예측

                    score = float(pred[0][1] * 100)  # 낙상 확률 (1에 대한 확률)
                    label = int(pred_label[0])

                    calculated["risk_score"] = score
                    calculated["Label"] = label
                    latest_score = score
                    latest_label = "Fall" if label == 1 else "Normal"

                    # 낙상 감지 시 알람 로직
                    if label == 1:
                        # play_alarm_sound() # 실제로 소리 재생을 원할 경우 주석 해제 (EC2에서 소리가 나진 않음)
                        print("🚨 낙상 감지됨: Alarm Triggered")


                except Exception as e:
                    print("⚠️ 실시간 예측 오류:", e)
                    calculated["risk_score"] = 0.0
                    calculated["Label"] = 0

                calculated['user_id'] = current_user_id if current_user_id else "anonymous"  # DB 저장을 위해 user_id 추가
                # DB 저장 (로그인된 사용자 ID가 있는 경우에만 실행)
                if current_user_id:
                    save_to_db(calculated)

                # MediaPipe 랜드마크를 프레임에 그림
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # 관절 색상 (파랑)
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # 선 색상 (초록)
                )

            # 최신 프레임 저장 (lock으로 보호)
            with frame_lock:
                # 프레임에 현재 상태 정보 추가 (디버깅용)
                status_text = f"Status: {latest_label} (Score: {latest_score:.2f}%)"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                latest_frame = frame.copy()
                frame_idx += 1

        except Exception as e:
            print(f"[ERROR] capture_frames 예외 발생: {e}")
            time.sleep(0.2)

        # FPS 제어
        # 영상 스트림의 FPS를 따르거나, 최소 25FPS를 보장하도록 대기
        time.sleep(1 / fps if fps > 0 else 1 / 25)


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
    # 로그아웃 상태 유지
    session.pop('user_id', None)
    global current_user_id
    current_user_id = None

    return render_template('login.html')


# ------ 로그인 기능 -------
@app.route('/login', methods=['POST'])
def login():
    global current_user_id
    user_id = request.form['id']
    password = request.form['password']

    conn = get_db_connection()
    if conn is None:
        return render_template('login.html', error_msg="DB 연결 실패. 관리자에게 문의하세요.")

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id=%s AND password=%s", (user_id, password))
        user = cursor.fetchone()
    finally:
        conn.close()

    if user:
        session['user_id'] = user_id
        current_user_id = user_id  # 스레드에서 사용 가능
        print(f"[INFO] User {user_id} logged in. Current camera loop will try to connect to user's URL.")
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
        if conn is None:
            return render_template('register.html', error_msg="DB 연결 실패. 관리자에게 문의하세요.")

        try:
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
            return redirect('/')
        except Exception as e:
            conn.rollback()
            return render_template('register.html', error_msg=f"회원가입 중 DB 오류 발생: {e}")
        finally:
            conn.close()

    return render_template('register.html')


# ------ 아이디어 중복 체크 확인 -------
@app.route('/check_id')
def check_id():
    user_id = request.args.get('id')
    exists = False

    if user_id:
        conn = get_db_connection()
        if conn is None:
            return jsonify({"exists": False, "error": "DB_CONNECTION_FAILED"})

        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
            if cursor.fetchone():
                exists = True
        finally:
            conn.close()

    return jsonify({"exists": exists})


# ----- 실시간 화면 및 신고하는 페이지 ------
@app.route('/camera')
def index():
    user_id = session.get('user_id')
    # 로그인 상태가 아니면 로그인 페이지로 리다이렉트
    if not user_id:
        return redirect('/')

    camera_url = None
    is_youtube = False
    embed_url = None

    if user_id:
        camera_url = get_camera_url(user_id)  # DB에서 가져오기
        if camera_url:
            # YouTube URL 확인
            if "youtube.com" in camera_url or "youtu.be" in camera_url:
                is_youtube = True

                # embed URL 변환
                video_id = None
                parsed_url = urlparse(camera_url)

                if "youtube.com/watch" in camera_url:
                    query = parse_qs(parsed_url.query)
                    video_id = query.get("v", [None])[0]
                elif "youtu.be" in camera_url:
                    # 'youtu.be/video_id' 형태 처리
                    video_id = parsed_url.path.strip("/")
                elif "youtube.com/shorts" in camera_url:
                    # 'shorts/video_id' 형태 처리
                    video_id = parsed_url.path.split("/")[-1]

                if video_id:
                    # &autoplay=1 추가: 영상 자동 재생
                    # loop=1과 playlist=video_id를 추가하여 자동 반복 재생 시도
                    embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=1&loop=1&playlist={video_id}"
                else:
                    # 영상 ID 못 찾으면 유튜브 처리 취소
                    is_youtube = False
                    embed_url = None

        # current_user_id 전역 변수 설정 (스레드 동기화)
        global current_user_id
        current_user_id = user_id

    return render_template('camera.html',
                           user_id=user_id,  # 사용자 ID 전달
                           camera_url=camera_url,
                           is_youtube=is_youtube,
                           embed_url=embed_url)


# ----- 실시간 화면 ------
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

#
# # ----- 낙상 위험 점수 기반 알림 로직 추가 ------
# def play_alarm_sound():
#     """🔊 서버 스피커에서 경고음 재생 (EC2 환경에서는 작동하지 않을 가능성이 높음)"""
#
#     def _play():
#         try:
#             # playsound 모듈은 EC2 서버 환경에서 소리가 나지 않을 수 있음
#             # 로컬에서만 테스트용으로 활용
#             playsound("static/alarmclockbeepsaif.mp3")
#             print("🔊 Alarm sound played!")
#         except Exception as e:
#             print(f"❌ Alarm Sound Error: {e}")
#
#     # 알림 발생시 Flask가 멈춤을 대비 -> 별도 스레드 생성
#     threading.Thread(target=_play, daemon=True).start()


# ----- 새로운 위험도 확인 라우트 ------
@app.route('/get_score')
def get_score():
    try:
        # SQLAlchemy 엔진을 통해 DB에서 직접 읽기
        if engine is None:
            return jsonify({"risk_score": 0.0, "error": "DB_ENGINE_FAILED"})

        df = pd.read_sql_query(
            "SELECT risk_score FROM realtime_screen ORDER BY timestamp DESC LIMIT 1",
            con=engine
        )

        if df.empty:
            # 전역 변수를 사용하거나, 데이터가 없으면 0 반환
            return jsonify({"risk_score": latest_score})

        return jsonify({"risk_score": round(df['risk_score'].iloc[0], 2)})

    except Exception as e:
        print(f"❌ get_score 조회 오류: {e}")
        return jsonify({"risk_score": latest_score})  # DB 오류 시 실시간 메모리 값 반환


# ==========================
# 서버 실행 및 스레드 실행
# ==========================
if __name__ == "__main__":
    # MediaPipe Drawing Utility import (capture_frames 함수에서 사용됨)
    mp_drawing = mp.solutions.drawing_utils

    # ⚠️ 빈 프레임을 미리 초기화하여 MJPEG 스트리밍 오류 방지
    with frame_lock:
        latest_frame = empty_frame.copy()

    # 카메라 연결 스레드 시작
    threading.Thread(target=connect_camera_loop, daemon=True).start()
    # 프레임 캡처/분석/DB 저장 스레드 시작
    threading.Thread(target=capture_frames, daemon=True).start()

    # 배포시 변경 사항 (debug=False, use_reloader=False)
    # AWS EC2 환경에서 0.0.0.0과 5000 포트 사용
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)