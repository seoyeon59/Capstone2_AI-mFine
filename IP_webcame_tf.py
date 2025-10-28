import cv2

cap = cv2.VideoCapture("http://192.168.45.3:8080/video")
ret, frame = cap.read()
print(ret, frame)  # True + ndarray 이면 OK, False + None이면 실패