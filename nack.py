import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 비율 기준 (0~1)
THRESHOLD = 0.07  # 7% 이상 코가 어깨보다 앞으로 나올 때 거북목

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # 카메라 쪽 어깨 선택 (오른쪽 어깨 기준)
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

            # 픽셀 좌표
            shoulder_px = (int(shoulder.x * w), int(shoulder.y * h))
            nose_px = (int(nose.x * w), int(nose.y * h))

            # 수평 거리 정규화 (너비 대비 비율)
            diff_x = abs(nose.x - shoulder.x)  # 0~1 범위로 이미 정규화됨

            # 시각화
            cv2.circle(image, shoulder_px, 8, (255, 0, 0), -1)
            cv2.circle(image, nose_px, 8, (0, 255, 0), -1)
            cv2.line(image, shoulder_px, nose_px, (255, 255, 0), 2)

            cv2.putText(image, f"Shoulder-Nose X diff: {diff_x:.3f}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 거북목 판단
            if diff_x > THRESHOLD:
                cv2.putText(image, "Forward Head Detected!",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(image, "Normal Posture",
                            (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("Side Neck Posture (Normalized)", image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
