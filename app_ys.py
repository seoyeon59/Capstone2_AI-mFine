from flask import Flask, Response, render_template, request, redirect, session, jsonify
import cv2
import sqlite3
import threading
import time
import os
import numpy as np
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)
app.secret_key = os.urandom(24)

# DB 경로
DB_PATH = 'capstone2.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# 전역 카메라 객체
cap = None
fps = 30
frame_lock = threading.Lock()
latest_frame = None
frame_idx = 0
current_user_id = None

# 카메라 URL 가져오기
def get_camera_url(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT camera_url FROM cameras WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return row[0]
    return None

# IP/유튜브 구분 후 cv2.VideoCapture
def get_video_capture(url):
    if "youtube.com" in url or "youtu.be" in url:
        try:
            import yt_dlp
            ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True, 'no_warnings': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                video_url = info_dict.get("url", None)
                if video_url:
                    cap_temp = cv2.VideoCapture(video_url)
                    if cap_temp.isOpened():
                        print(f"[INFO] YouTube 연결 성공: {url}")
                        return cap_temp
        except Exception as e:
            print(f"[ERROR] YouTube 연결 실패: {e}")
            return None
    else:
        cap_temp = cv2.VideoCapture(url)
        if cap_temp.isOpened():
            print(f"[INFO] IP 카메라 연결 성공: {url}")
            return cap_temp
    return None

# 카메라 연결 반복 시도
def connect_camera_loop():
    global cap, fps, current_user_id
    while True:
        if cap is None or not cap.isOpened():
            if current_user_id:

                ip_url = get_camera_url(current_user_id)
                if ip_url:
                    temp_cap = get_video_capture(ip_url)
                    if temp_cap and temp_cap.isOpened():
                        cap = temp_cap
                        fps_val = int(cap.get(cv2.CAP_PROP_FPS))
                        fps = fps_val if fps_val > 0 else 30
                    else:
                        print("[WARN] 카메라 연결 실패, 3초 후 재시도")
                        if temp_cap:
                            temp_cap.release()
                else:
                    print("[WARN] camera_url 없음, 3초 후 재시도")
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
                print("[WARN] 프레임 읽기 실패")
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (640, 480))

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
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    global current_user_id
    user_id = request.form['id']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id=? AND password=?", (user_id, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['user_id'] = user_id
        current_user_id = user_id
        return redirect('/camera')
    else:
        return "이름 또는 비밀번호를 확인하세요."

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        id = request.form['id']
        password = request.form['password']
        username = request.form['username']
        phone_number = request.form['phone_number']
        non_guardian_name = request.form['non_guardian_name']
        mail = request.form['mail']
        camera_url = request.form['camera_url']

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE id = ?", (id,))
        if cursor.fetchone():
            conn.close()
            return render_template('register.html', error_msg="이미 존재하는 아이디입니다.")

        cursor.execute("""
            INSERT INTO users (id, password, username, phone_number, non_guardian_name, mail)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (id, password, username, phone_number, non_guardian_name, mail))

        cursor.execute("""
            INSERT INTO cameras (user_id, camera_url)
            VALUES (?, ?)
        """, (id, camera_url))

        conn.commit()
        conn.close()
        return redirect('/')

    return render_template('register.html')

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

@app.route('/camera')
def index():
    user_id = session.get('user_id')
    camera_url = None
    is_youtube = False
    embed_url = None

    if user_id:
        camera_url = get_camera_url(user_id)
        if camera_url and ("youtube.com" in camera_url or "youtu.be" in camera_url):
            is_youtube = True
            video_id = None
            if "youtube.com/watch" in camera_url:
                from urllib.parse import parse_qs, urlparse
                query = parse_qs(urlparse(camera_url).query)
                video_id = query.get("v", [None])[0]
            elif "youtu.be" in camera_url:
                video_id = camera_url.split("/")[-1]
            elif "youtube.com/shorts" in camera_url:
                video_id = camera_url.split("/")[-1]

            if video_id:
                embed_url = f"https://www.youtube.com/embed/{video_id}?autoplay=1"
            else:
                is_youtube = False
                embed_url = None

    return render_template('camera.html',
                           camera_url=camera_url,
                           is_youtube=is_youtube,
                           embed_url=embed_url)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)