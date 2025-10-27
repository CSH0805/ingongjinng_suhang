import cv2
import mediapipe as mp
import math

# MediaPipe 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 각도 계산 함수
def calculate_angle(a, b, c):
    # 세 점의 좌표를 각각 a,b,c에 입력
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    # 벡터 계산
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    # 내적(dot product) 이용 각도 계산
    cosine_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (
        math.sqrt(ba[0]**2 + ba[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2)
    )
    angle = math.degrees(math.acos(cosine_angle))
    return angle

# 웹캠 시작
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우 반전 및 색상 변환
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 포즈 감지
        results = pose.process(image)

        # OpenCV 표시용 복구
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # 어깨, 귀, 코 랜드마크 사용
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

            # 각도 계산 (어깨-귀-코)
            angle = calculate_angle(shoulder, ear, nose)

            # 텍스트 출력
            cv2.putText(image, f'Neck Angle: {int(angle)} deg', 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 거북목 판단 (임계값: 40도 미만이면 거북목)
            if angle < 40:
                cv2.putText(image, 'Forward Head Detected!', 
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        except:
            pass

        # 포즈 랜드마크 시각화
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Neck Angle Detection', image)

        # q 누르면 종료
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
