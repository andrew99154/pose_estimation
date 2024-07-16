import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 初始化姿態估計模型
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 開啟鏡頭
cap = cv2.VideoCapture(1)

# 狀態機的初始狀態
state = 's1'  # s1: 未舉起, s2: 舉起但未到指定高度, s3: 完全舉起
cycle_count = 0
correct_cnt = 0
last_angle = 0  # 最後計算的角度
passed = False


def calculate_angle(landmark1, landmark2, landmark3):
    """計算三個點形成的角度"""
    # 獲取坐標
    a = np.array([landmark1.x, landmark1.y])
    b = np.array([landmark2.x, landmark2.y])
    c = np.array([landmark3.x, landmark3.y])
    
    # 根據向量計算角度
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 轉換顏色空間從BGR到RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 處理圖片，取得結果
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 計算角度
        l_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        l_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        l_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

        r_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
        l_angle = calculate_angle(l_elbow, l_shoulder, l_hip)
        r_angle = calculate_angle(r_elbow, r_shoulder, r_hip)
        last_angle = min(l_angle, r_angle)

        # 狀態轉換邏輯
        if state == 's1' and l_angle >= 120 and r_angle >= 120:
            state = 's2'
        elif state == 's2' and l_angle >= 160 and r_angle >= 160:
            state = 's3'
            passed = True
        elif state == 's3' and l_angle < 160 and r_angle < 160:
            state = 's2'
        elif state == 's2' and l_angle < 120 and r_angle < 120:
            state = 's1'
            if passed:
                correct_cnt += 1
                passed = False
            cycle_count += 1  # 完成一個完整的周期

    # 顯示狀態、計數器和角度
    cv2.putText(frame, f'State: {state}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Cycle count: {cycle_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Angle: {int(last_angle)} degrees', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Correct count: {correct_cnt}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

    # 顯示影像
    cv2.imshow('Fitness Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源和關閉視窗
cap.release()
cv2.destroyAllWindows()
