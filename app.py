from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import pandas as pd
import time

app = Flask(__name__)

# ==========================
# MediaPipe Pose 초기화 (백그라운드 처리)
# ==========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# IP 웹캠 연결
# ==========================
# ==========================
# IP 웹캠 연결
# ==========================
ip_url = "http://192.168.45.3:8080/video"
cap = cv2.VideoCapture(ip_url)
if not cap.isOpened():
    print("[ERROR] IP 웹캠 연결 실패. 영상 스트리밍 불가, 서버는 계속 실행합니다.")
    cap = None  # cap이 None이면 gen_frames에서 검은 화면 표시

# FPS 설정 (cap이 있는 경우만)
fps = 30  # 기본값
if cap is not None:
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    if fps_val and fps_val > 0:
        fps = int(fps_val)


# ==========================
# 프레임 생성 (원본 화면만 스트리밍)
# ==========================
def gen_frames():
    global frame_idx, data, cap
    if cap is None or not cap.isOpened():
        print("[ERROR] 카메라가 연결되지 않았습니다. 영상 스트리밍 불가.")
        while True:
            # 빈 화면 스트리밍 (검은색 640x480 이미지)
            import numpy as np
            frame_resized = np.zeros((480, 640, 3), dtype=np.uint8)
            ret2, buffer = cv2.imencode('.jpg', frame_resized)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] 프레임 수신 실패. 더 이상 재연결하지 않습니다.")
                # 검은 화면 스트리밍으로 대체
                import numpy as np
                frame_resized = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame_resized = cv2.resize(frame, (640, 480))
                rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # MediaPipe 처리 (백그라운드)
                results = pose.process(rgb_frame)
                if results.pose_landmarks:
                    row = {'frame': frame_idx}
                    for i, lm in enumerate(results.pose_landmarks.landmark):
                        row[f"x_{i}"] = lm.x
                        row[f"y_{i}"] = lm.y
                        row[f"z_{i}"] = lm.z
                        row[f"v_{i}"] = lm.visibility
                    data.append(row)

            # 영상 스트리밍
            ret2, buffer = cv2.imencode('.jpg', frame_resized)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            frame_idx += 1

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

@app.route('/shutdown')
def shutdown():
    global data
    pd.DataFrame(data).to_csv("pose_keypoints.csv", index=False)
    print("[INFO] CSV 저장 완료 ✅")
    func = None
    from flask import request
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "Server shutting down..."

# ==========================
# 서버 실행
# ==========================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
