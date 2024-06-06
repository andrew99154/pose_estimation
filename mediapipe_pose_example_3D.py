import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(shoulder,elbow,wrist):

    vector1 = [shoulder.x - elbow.x, shoulder.y - elbow.y, shoulder.z - elbow.z]
    vector2 = [wrist.x - elbow.x, wrist.y - elbow.y, wrist.z - elbow.z]

    inner_mul = vector1[0]*vector2[0] + vector1[1]*vector2[1] + vector1[2]*vector2[2]
    norm1 = np.sqrt(vector1[0]**2 + vector1[1]**2 + vector1[2]**2)
    norm2 = np.sqrt(vector2[0]**2 + vector2[1]**2 + vector2[2]**2)
    radians = np.arccos(inner_mul/(norm1*norm2))

    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle

    return angle 

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])

            # Get coordinates
            # shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            # wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

            elbow_2d = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            # print(angle)
            
        # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )           

            # Visualize angle
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow_2d, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
                       
        except:
            pass
        
        
    
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


