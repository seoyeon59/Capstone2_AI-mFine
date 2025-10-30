from flask import Flask, Response, render_template, jsonify, request, redirect, session
import cv2
import mediapipe as mp
import pandas as pd
import sqlite3
import numpy as np
import threading
import time
from alert_utils import send_sms
from playsound import playsound  # pip install playsound==1.2.2

app = Flask(__name__)

# ==========================
# SQLite 연결 (회원가입/로그인용)
# ==========================
DB_PATH = 'capstone2.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 컬럼명을 dict처럼 사용 가능
    return conn

# ==========================
# MediaPipe Pose 초기화
# ==========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 전역 변수 초기화
data = []
frame_idx = 0
latest_frame = None
frame_lock = threading.Lock()

# ==========================
# DB에서 camera_url 가져오기
# ==========================
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

# ==========================
# IP 웹캠 연결
# ==========================
# ip_url = "http://192.168.45.3:8080/video" # DB 연결 후 camera table에서 연결하도록 수정할 예정
ip_url = get_camera_url("test")
cap = cv2.VideoCapture(ip_url)
if not cap.isOpened():
    print("[ERROR] IP 웹캠 연결 실패. 영상 스트리밍 불가, 서버는 계속 실행합니다.")
    cap = None # cap이 None이면 gen_frames에서 검은 화면 표시 #

# FPS 설정 (cap이 있는 경우만)
fps = 30 # 기본값
if cap is not None:
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    if fps_val and fps_val > 0:
        fps = int(fps_val)

# ==========================
# 프레임 읽기 스레드
# ==========================
def capture_frames():
    global latest_frame, cap, frame_idx, data
    while True:
        if cap is None or not cap.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (640, 480))

                # =======================
                # MediaPipe 처리 (주석 유지)
                # =======================
                # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # results = pose.process(rgb_frame)
                # if results.pose_landmarks:
                #     row = {'frame': frame_idx}
                #     for i, lm in enumerate(results.pose_landmarks.landmark):
                #         row[f"x_{i}"] = lm.x
                #         row[f"y_{i}"] = lm.y
                #         row[f"z_{i}"] = lm.z
                #         row[f"v_{i}"] = lm.visibility
                #     data.append(row)

        with frame_lock:
            latest_frame = frame.copy()
            frame_idx += 1
        time.sleep(1 / 30)


threading.Thread(target=capture_frames, daemon=True).start()

# ==========================
# Flask MJPEG 스트리밍
# ==========================
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

# ==========================
# Flask 라우팅
# ==========================
# ==========================
# 홈 (로그인 페이지)
# ==========================
@app.route('/')
def home():
    return render_template('login.html')

# ==========================
# 로그인 기능
# ==========================
@app.route('/login', methods=['POST'])
def login():
    name = request.form['name']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user WHERE name=? AND password=?", (name, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['name'] = name
        print(f"✅ 로그인 성공: {name}")
        return redirect('/camera')
    else:
        print(f"❌ 로그인 실패: {name}")
        return "❌ 로그인 실패! 이름 또는 비밀번호를 확인하세요."

# ==========================
# 회원가입 기능
# ==========================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        password = request.form['password']
        name = request.form['name']
        phone_number = request.form['phone_number']
        non_guardian_name = request.form['non_guardian_name']
        mail = request.form['mail']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user (password, name, phone_number, non_guardian_name, mail)
            VALUES (?, ?, ?, ?, ?)
        """, (password, name, phone_number, non_guardian_name, mail))
        conn.commit()
        conn.close()

        print(f"✅ 회원가입 완료: {name}")
        return redirect('/')
    return render_template('register.html')

# ========================
# 카메라
# ========================
@app.route('/camera')
def index():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================
# 낙상 위험 점수 기반 알림 로직 추가
# ==========================
def play_alarm_sound():
    """🔊 서버 스피커에서 경고음 재생"""
    try:
        playsound("static/alarmclockbeepsaif.mp3")
        print("🔊 Alarm sound played!")
    except Exception as e:
        print(f"❌ Alarm Sound Error: {e}")

# --------------------------
# 새로운 위험도 확인 라우트
# --------------------------
@app.route('/get_score')
def get_score():
    conn = sqlite3.connect('capstone2.db')
    c = conn.cursor()
    c.execute("SELECT risk_score FROM realtime_screen ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    conn.close()

    score = (row[0] / 100) if row else 0.0

    ### 🔔 추가: 위험 점수 기반으로 문자 및 경고 알림
    numeric_score = score * 100  # 0~1 → 0~100 단위로 변경
    user_phone = "+821023902894"  # ⚠️ 사용자 휴대폰 번호 (실제 번호로 수정)

    if numeric_score >= 70:
        msg = f"🚨 낙상 위험이 매우 높습니다! (위험도: {int(numeric_score)}점)\n즉시 확인이 필요합니다."
        print("문자 및 경고음 발송 중...")
        threading.Thread(target=send_sms, args=(user_phone, msg)).start()
        threading.Thread(target=play_alarm_sound).start()
    elif numeric_score >= 50:
        msg = f"⚠️ 낙상 주의: 위험도가 {int(numeric_score)}점입니다. 주의하세요."
        print("주의 문자 발송 중...")
        threading.Thread(target=send_sms, args=(user_phone, msg)).start()

    return jsonify({'score': score})

@app.route('/shutdown')
def shutdown():
    global data
    pd.DataFrame(data).to_csv("pose_keypoints.csv", index=False)
    print("[INFO] CSV 저장 완료 ✅")
    from flask import request
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "Server shutting down..."

# 알림 소리 재생
def play_alarm_sound():
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
