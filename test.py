import cv2
import mediapipe as mp
import pandas as pd
import time

# MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ip_url = "http://192.168.45.3:8080/video"
cap = cv2.VideoCapture(ip_url)

if not cap.isOpened():
    print("[ERROR] IP 웹캠 연결 실패. URL 확인 필요.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_idx = 0
data = []

print("[INFO] Pose 추출 시작 (종료: q 키)")

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 프레임 수신 실패. 재시도 중...")
            time.sleep(0.5)
            cap.release()
            cap = cv2.VideoCapture(ip_url)
            continue

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            row = {'frame': frame_idx}
            for i, lm in enumerate(results.pose_landmarks.landmark):
                row[f"x_{i}"] = lm.x
                row[f"y_{i}"] = lm.y
                row[f"z_{i}"] = lm.z
                row[f"v_{i}"] = lm.visibility
            data.append(row)

        cv2.imshow("MediaPipe Pose - IP Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

cap.release()
cv2.destroyAllWindows()
pd.DataFrame(data).to_csv("pose_keypoints.csv", index=False)
print("[INFO] Pose 추출 완료 및 CSV 저장 완료 ✅")
