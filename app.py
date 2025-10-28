from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
import pandas as pd
import sqlite3
import numpy as np
import threading
import time

app = Flask(__name__)

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
@app.route('/')
def index():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_score')
def get_score():
    conn = sqlite3.connect('capstone2.db')
    c = conn.cursor()
    c.execute("SELECT risk_score FROM realtime_screen ORDER BY timestamp DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    score = (row[0] / 100) if row else 0.0
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

# ==========================
# 서버 실행
# ==========================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
