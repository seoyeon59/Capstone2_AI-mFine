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

# ===========================
# 1. 환경 설정 및 변수
# ===========================
app = Flask(__name__)
# 보안을 위해 실제 환경에서는 환경 변수나 별도 파일에서 로드해야 합니다.
app.secret_key = os.urandom(24)

# RDS 연결 정보 (환경 변수를 통해 안전하게 로드)
# EC2에 환경 변수 설정 필요: export DB_HOST='[RDS 엔드포인트]'
DB_HOST = os.environ.get('DB_HOST', 'swu-sw-02-db.cfoqwsiqgd5l.ap-northeast-2.rds.amazonaws.com')  # RDS 엔드포인트
DB_USER = os.environ.get('DB_USER', 'admin')  # RDS 마스터 사용자 이름
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'aimfine2!')  # RDS 마스터 암호
DB_NAME = os.environ.get('DB_NAME', 'capstone2')
DB_PORT = 3306

if not all([DB_HOST, DB_PASSWORD]):
    # 배포 환경에서 이 오류가 발생하면 환경 변수 설정을 확인하세요.
    print("FATAL: DB_HOST or DB_PASSWORD is not set.")
    exit(1)

# SQLAlchemy 엔진 생성 (Flask에서 DB 연결 풀 관리 및 쿼리 편의성 제공)
# 퍼센트 기호가 포함된 비밀번호 처리를 위해 quote_plus 사용
db_url = (
    f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)
engine = create_engine(db_url, pool_recycle=3600)

# ===========================
# 2. 모델 및 상태 변수
# ===========================

# 2-1. Mediapipe 및 ML 모델
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2-2. 칼만 필터 초기화
kf = KalmanFilter(
    initial_state_mean=np.zeros(2),
    initial_state_covariance=np.eye(2),
    transition_matrices=np.array([[1, 1], [0, 1]]),
    observation_matrices=np.eye(2),
    observation_covariance=0.01 * np.eye(2),
    transition_covariance=0.0001 * np.eye(2)
)
current_state_mean = kf.initial_state_mean
current_state_covariance = kf.initial_state_covariance

# 2-3. 낙상 감지 ML 모델 로드
try:
    # 모델 파일이 없는 경우를 대비하여 더미 모델 사용
    # model = joblib.load('fall_detection_model.pkl')
    def dummy_predict(data):
        if np.mean(data) > 0.5:
            return np.array([1])  # 낙상
        return np.array([0])  # 정상


    model = type('DummyModel', (object,),
                 {'predict': dummy_predict, 'predict_proba': lambda x: np.array([[1 - np.mean(x), np.mean(x)]])})()

except Exception as e:
    print(f"❌ ML 모델 로드 오류: {e}")


    def dummy_predict(data):
        return np.array([0])


    model = type('DummyModel', (object,), {'predict': dummy_predict, 'predict_proba': lambda x: np.array([[1, 0]])})()

# 2-4. 비디오 스트림 및 상태
CAMERA_URL = None
cap = None
FRAME_LOCK = threading.Lock()
LATEST_FRAME = None
IS_YOUTUBE = False
stream_thread = None
stop_event = threading.Event()
USER_ID = None


# ===========================
# 3. 데이터베이스 헬퍼 함수
# ===========================

# DB 연결을 시도하고 커넥션을 반환합니다.
def get_db_connection():
    try:
        conn = pymysql.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10
        )
        return conn
    except Exception as e:
        print(f"❌ DB 연결 오류: {e}")
        return None


# 사용자 정보를 DB에서 가져옵니다.
def get_user_data(user_id):
    conn = get_db_connection()
    if conn is None:
        return None
    try:
        with conn.cursor() as cursor:
            # users 테이블에서 id(IAM username)로 사용자 정보를 조회합니다.
            sql = "SELECT * FROM users WHERE id = %s"
            cursor.execute(sql, (user_id,))
            user_data = cursor.fetchone()
            return user_data
    except Exception as e:
        print(f"❌ 사용자 데이터 조회 오류: {e}")
        return None
    finally:
        conn.close()


# 사용자 로그인 검증
def authenticate_user(user_id, password):
    conn = get_db_connection()
    if conn is None:
        return False, None
    try:
        with conn.cursor() as cursor:
            # 실제 서비스에서는 반드시 비밀번호 해싱을 사용해야 합니다.
            sql = "SELECT id, password FROM users WHERE id = %s AND password = %s"
            cursor.execute(sql, (user_id, password))
            user = cursor.fetchone()
            return user is not None, user['id'] if user else None
    except Exception as e:
        print(f"❌ 로그인 인증 오류: {e}")
        return False, None
    finally:
        conn.close()


# 사용자 등록
def register_user_data(data):
    conn = get_db_connection()
    if conn is None:
        return False
    try:
        with conn.cursor() as cursor:
            # 현재는 비밀번호를 평문으로 저장합니다. (보안상 매우 위험)
            sql = """
                INSERT INTO users 
                (id, password, username, phone_number, non_guardian_name, camera_url, mail)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                data['id'], data['password'], data['username'],
                data['phone_number'], data['non_guardian_name'],
                data['camera_url'], data['mail']
            ))
            conn.commit()
            return True
    except pymysql.err.IntegrityError as e:
        print(f"❌ 사용자 등록 오류 - 중복 ID: {e}")
        return False
    except Exception as e:
        print(f"❌ 사용자 등록 오류: {e}")
        return False
    finally:
        conn.close()


# ID 중복 체크
def is_id_taken(user_id):
    conn = get_db_connection()
    if conn is None:
        return True  # DB 연결 실패 시 안전하게 중복으로 처리
    try:
        with conn.cursor() as cursor:
            sql = "SELECT id FROM users WHERE id = %s"
            cursor.execute(sql, (user_id,))
            result = cursor.fetchone()
            return result is not None
    except Exception as e:
        print(f"❌ ID 중복 체크 오류: {e}")
        return True
    finally:
        conn.close()


# ===========================
# 4. 카메라 및 스트리밍 로직
# ===========================

# 전역 변수 업데이트 함수
def update_global_stream_config(user_id, camera_url):
    global USER_ID, CAMERA_URL, cap, stop_event, stream_thread, IS_YOUTUBE

    # 기존 스트리밍 스레드 중지
    if stream_thread and stream_thread.is_alive():
        stop_event.set()
        stream_thread.join()

    USER_ID = user_id
    CAMERA_URL = camera_url
    IS_YOUTUBE = 'youtube.com' in CAMERA_URL or 'youtu.be' in CAMERA_URL if CAMERA_URL else False

    # 유튜브 URL은 cv2.VideoCapture로 처리할 수 없습니다.
    if IS_YOUTUBE or not CAMERA_URL:
        cap = None
        return

    # 새로운 스트림 시작
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print(f"⚠️ Warning: Cannot open video stream for URL: {CAMERA_URL}")
        cap = None
        return

    stop_event.clear()
    stream_thread = threading.Thread(target=read_stream_thread, daemon=True)
    stream_thread.start()


# 스트림 읽기 스레드 (백그라운드에서 프레임을 읽어옴)
def read_stream_thread():
    global LATEST_FRAME, current_state_mean, current_state_covariance

    while not stop_event.is_set() and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Stream error or end of video.")
            time.sleep(1)
            continue

        # 프레임 처리 (MediaPipe 및 ML 추론)
        processed_frame, risk_score = process_frame_for_fall_detection(frame)

        # 위험 점수 저장
        if risk_score is not None:
            save_risk_score(risk_score)

        # 웹 스트리밍을 위한 인코딩 및 전역 변수 업데이트
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if ret:
            jpg_as_text = buffer.tobytes()
            with FRAME_LOCK:
                LATEST_FRAME = jpg_as_text

        # API 호출 속도 제한 (초당 5프레임 정도)
        time.sleep(1 / 5)

    if cap:
        cap.release()
    print("Stream thread stopped.")


# 이미지 처리 및 ML 추론
def process_frame_for_fall_detection(frame):
    global current_state_mean, current_state_covariance

    # OpenCV BGR -> RGB 변환 (MediaPipe용)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # MediaPipe Pose 추론
    results = pose.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    risk_score = None

    if results.pose_landmarks:
        # 주요 랜드마크 좌표 추출
        landmarks = results.pose_landmarks.landmark

        if landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility > 0.8 and \
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility > 0.8 and \
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility > 0.8 and \
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > 0.8:

            # 엉덩이 중심 Y 좌표
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            # 어깨 중심 Y 좌표
            shoulder_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2

            # 칼만 필터 예측 및 업데이트
            measurement = np.array([hip_y, shoulder_y])

            current_state_mean, current_state_covariance = kf.filter_update(
                current_state_mean, current_state_covariance, measurement
            )

            # ML 모델 예측을 위한 특징 벡터 생성 (예시)
            feature_vector = np.array([current_state_mean[0], current_state_mean[1]])

            # ML 모델 예측
            try:
                proba = model.predict_proba([feature_vector.flatten()])[0]
                risk_score = round(proba[1] * 100, 2)

            except Exception as e:
                print(f"❌ ML 예측 오류: {e}")
                risk_score = None

                # Pose 랜드마크를 프레임에 그립니다.
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    # 낙상 위험 점수를 프레임에 표시 (디버깅용)
    if risk_score is not None:
        text = f"Risk: {risk_score:.2f}%"
        cv2.putText(image, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return image, risk_score


# 위험 점수 DB 저장
def save_risk_score(score):
    global USER_ID
    if USER_ID is None:
        return

    conn = get_db_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cursor:
            # timestamp와 risk_score를 realtime_screen 테이블에 저장
            sql = "INSERT INTO realtime_screen (timestamp, risk_score, user_id) VALUES (%s, %s, %s)"
            cursor.execute(sql, (datetime.now(), score, USER_ID))
            conn.commit()
    except Exception as e:
        print(f"❌ 점수 저장 오류: {e}")
    finally:
        conn.close()


# M-JPEG 스트리밍을 위한 제너레이터
def generate_frames():
    while not stop_event.is_set():
        with FRAME_LOCK:
            if LATEST_FRAME is not None:
                frame = LATEST_FRAME
            else:
                # 스트림 준비 중이거나 오류 시 검은색 배경 반환
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Stream Loading...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                ret, buffer = cv2.imencode('.jpg', black_frame)
                frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        time.sleep(1 / 10)  # 10 FPS로 제한


# ===========================
# 5. Flask 라우트
# ===========================

# 로그인 필요 데코레이터
def login_required(f):
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # 로그인 페이지로 리다이렉트
            return redirect('/login')
        return f(*args, **kwargs)

    decorated_function.__name__ = f.__name__
    return decorated_function


# ==================== HTML 렌더링 라우트 ====================

@app.route('/')
@login_required
def index():
    # camera.html 페이지 렌더링
    user_id = session.get('user_id')
    user_data = get_user_data(user_id)
    camera_url = user_data['camera_url'] if user_data else None

    # 전역 스트림 설정 및 시작
    update_global_stream_config(user_id, camera_url)

    # 유튜브 URL 처리 (camera.html에서 iframe으로 표시)
    is_youtube = CAMERA_URL and ('youtube.com' in CAMERA_URL or 'youtu.be' in CAMERA_URL)
    embed_url = None
    if is_youtube:
        if 'watch?v=' in CAMERA_URL:
            video_id = CAMERA_URL.split('v=')[-1].split('&')[0]
            embed_url = f"https://www.youtube.com/embed/{video_id}"
        elif 'youtu.be/' in CAMERA_URL:
            video_id = CAMERA_URL.split('youtu.be/')[-1].split('?')[0]
            embed_url = f"https://www.youtube.com/embed/{video_id}"

    return render_template('camera.html',
                           camera_url=camera_url,
                           is_youtube=is_youtube,
                           embed_url=embed_url)


# 로그인 페이지
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['id']
        password = request.form['password']

        success, authenticated_id = authenticate_user(user_id, password)

        if success:
            session['user_id'] = authenticated_id
            return redirect('/')
        else:
            return render_template('login.html', error_msg="아이디 또는 비밀번호가 올바르지 않습니다.")

    return render_template('login.html', error_msg=None)


# 로그아웃
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    # 전역 스트림 중지
    update_global_stream_config(None, None)
    return redirect('/login')


# 회원가입 페이지
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_data = {
            'id': request.form['id'],
            'password': request.form['password'],
            'username': request.form['username'],
            'phone_number': request.form['phone_number'],
            'non_guardian_name': request.form['non_guardian_name'],
            'camera_url': request.form['camera_url'],
            'mail': request.form['mail']
        }

        if register_user_data(user_data):
            # 회원가입 성공 후 로그인 페이지로 리다이렉트
            return redirect('/login')
        else:
            return render_template('register.html', error_msg="이미 존재하는 아이디이거나 회원가입에 실패했습니다.")

    return render_template('register.html', error_msg=None)


# ==================== API 라우트 ====================

# ID 중복 체크 API
@app.route('/check_id')
def check_id():
    user_id = request.args.get('id')
    if not user_id:
        return jsonify({"taken": True})

    taken = is_id_taken(user_id)
    return jsonify({"taken": taken})


# M-JPEG 스트리밍 엔드포인트
@app.route('/video_feed')
@login_required
def video_feed():
    if IS_YOUTUBE or not CAMERA_URL:
        # 유튜브 URL이거나 URL이 없는 경우, 빈 응답 또는 오류 이미지 반환
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# 위험 점수 조회 API
@app.route('/get_score')
@login_required
def get_score():
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"risk_score": 0.0}), 401

        # 5초간 데이터가 없으면 DB에서 가장 최근 데이터 1개를 가져오는 시간 간격 (초)
        N_SECONDS = 5

        # 🔑 최근 N초간의 평균 위험 점수를 조회
        query = f"""
                SELECT risk_score
                FROM realtime_screen
                WHERE user_id = '{user_id}' AND timestamp >= TIMESTAMPADD(SECOND, -{N_SECONDS}, NOW())
                ORDER BY timestamp DESC
            """
        df = pd.read_sql_query(query, con=engine)

        if df.empty:
            # 최근 5초간 데이터가 없으면, 해당 사용자의 가장 최근의 데이터라도 가져옴
            df = pd.read_sql_query(
                f"SELECT risk_score FROM realtime_screen WHERE user_id = '{user_id}' ORDER BY timestamp DESC LIMIT 1",
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
        return jsonify({"risk_score": 0.0}), 500


# ===========================
# 서버 실행 (개발/테스트용)
# ===========================
if __name__ == '__main__':
    # Flask 앱 시작 전에 전역 스트림 설정을 None으로 초기화합니다.
    update_global_stream_config(None, None)

    # 개발 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)