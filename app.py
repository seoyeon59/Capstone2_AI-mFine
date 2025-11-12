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
from playsound import playsound
import yt_dlp
import os
from urllib.parse import urlparse, parse_qs, quote_plus
from sqlalchemy import create_engine

app = Flask(__name__)
app.secret_key = os.urandom(24)  # 랜덤값으로 만들기(배포시 수정해야함)

# AI 모델 로드
scaler = joblib.load("pkl/scaler.pkl")
model = joblib.load("pkl/decision_tree_model.pkl")

# DB 연결
DB_PATH = 'capstone2.db'

# SQLAlchemy 엔진 생성
password = "bear0205!@!@"  # 실제 비밀번호
password_encoded = quote_plus(password)  # URL-safe 인코딩

engine = create_engine(
    f"mysql+pymysql://root:{password_encoded}@127.0.0.1:3306/capstone2?charset=utf8mb4"
)

# ----- DB 연결 ------
def get_db_connection():
    conn = pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="bear0205!@!@",
        database="capstone2",
        cursorclass=pymysql.cursors.DictCursor  # dict 형태로 결과 사용 가능
    )

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
    try:
        with get_db_connection() as conn:
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

                # user_id별 최대 600개 제한
                user_id = filtered_data.get('user_id')
                if user_id:
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

# ------- DB에서 camera_url 가져오기 : 수정 제안 -------
def get_camera_url(user_id):
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT camera_url FROM cameras WHERE user_id = %s", (user_id,))
        row = cursor.fetchone()
        if row and 'camera_url' in row:
            return row['camera_url']  # <-- 이렇게 수정
        return None
    except Exception as e:
        print(f"⚠️ 카메라 URL 조회 오류: {e}")
        return None
    finally:
        if conn:
            conn.close()

# ------- IP/유튜브 구분 및 카메라 연결 : 수정 제안-------
def get_video_capture(url):
    try:
        if "youtube.com" in url or "youtu.be" in url:
            print("[INFO] YouTube 영상 URL 추출 중...")
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
                'nocheckcertificate': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                video_url = info_dict.get("url")
                if not video_url:
                    print("[ERROR] YouTube URL 가져오기 실패")
                    return None
                cap = cv2.VideoCapture(video_url)
                return cap
        else:
            print("[INFO] IP 카메라 연결 중...")
            cap = cv2.VideoCapture(url)
            return cap
    except Exception as e:
        print(f"[ERROR] 비디오 캡처 생성 실패: {e}")
        return None

# ------ IP 웹캠 연결 반복 시도 : 수정 제안 -------
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
            if current_user_id:
                url = get_camera_url(current_user_id)
                print(f"[DEBUG] 로그인된 사용자({current_user_id})의 URL: {url}")
            else:
                url = None

            # 로그인되어 있지 않거나 URL이 잘못된 경우 → 기본 카메라로 시도
            if not url or not isinstance(url, str) or not url.strip():
                print("[INFO] 로그인 안됨 또는 유효한 URL 없음 → 기본 카메라 연결 시도")
                url = default_url

            # 비디오 캡처 시도
            temp_cap = get_video_capture(url)
            if temp_cap and temp_cap.isOpened():
                cap = temp_cap
                fps_val = int(cap.get(cv2.CAP_PROP_FPS))
                fps = fps_val if fps_val > 0 else 30
                print(f"[INFO] 카메라 연결 성공 (FPS: {fps})")
            else:
                print(f"[WARN] 카메라 연결 실패, 3초 후 재시도")
                time.sleep(3)
                continue

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
        if cap is None or not cap.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(0.2)
            continue

        try:
            # ✅ 안정형: grab을 과도하게 하지 않음
            # (YouTube는 grab() 반복 시 연결이 끊김)
            ret, frame = cap.read()

            if not ret or frame is None:
                fail_count += 1
                print(f"[WARN] 프레임 읽기 실패 ({fail_count})")
                if fail_count > 10:
                    print("[ERROR] 스트림이 끊긴 것으로 판단, 재연결 시도 예정")
                    cap.release()
                    cap = None
                time.sleep(0.1)
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
    user_id = session.get('user_id')
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
                if "youtube.com/watch" in camera_url:
                    query = parse_qs(urlparse(camera_url).query)
                    video_id = query.get("v", [None])[0]
                elif "youtu.be" in camera_url:
                    video_id = camera_url.split("/")[-1]
                elif "youtube.com/shorts" in camera_url:
                    video_id = camera_url.split("/")[-1]

                if video_id:
                    embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=1"
                else:
                    # 영상 ID 못 찾으면 유튜브 처리 취소
                    is_youtube = False
                    embed_url = None

    return render_template('camera.html',
                           camera_url=camera_url,
                           is_youtube=is_youtube,
                           embed_url=embed_url)

# ----- 실시간 화면 ------
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ----- 낙상 위험 점수 기반 알림 로직 추가 ------
def play_alarm_sound():
    """🔊 서버 스피커에서 경고음 재생"""
    def _play():
        try:
            playsound("static/alarmclockbeepsaif.mp3")
            print("🔊 Alarm sound played!")
        except Exception as e:
            print(f"❌ Alarm Sound Error: {e}")
    # 알림 발생시 Flask가 멈춤을 대비 -> 별로 스레드 생성
    threading.Thread(target=_play, daemon=True).start()

# ----- 새로운 위험도 확인 라우트 ------
@app.route('/get_score')
def get_score():
    try:
        # SQLAlchemy 엔진으로 직접 읽기
        df = pd.read_sql_query(
            "SELECT risk_score FROM realtime_screen ORDER BY timestamp DESC LIMIT 1",
            con=engine
        )

        if df.empty:
            return jsonify({"risk_score": 0.0})  # 데이터 없으면 0 반환

        return jsonify({"risk_score": round(df['risk_score'].iloc[0], 2)})

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