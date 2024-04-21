import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    drawing_mode = False
    prev_thumb_tip = None
    prev_index_tip = None
    prev_center = None
    drawing_color = (255, 0, 0)
    drawing_thickness = 2 
    lines = [] 

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract the coordinates of thumb and index finger landmarks
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Check if thumb and index finger are touching
                thumb_x, thumb_y = int(thumb_tip.x * image.shape[1]), int(thumb_tip.y * image.shape[0])
                index_x, index_y = int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0])
                touching = (thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2 < 70 ** 2

                # Enter or exit drawing mode
                if touching and not drawing_mode:
                    drawing_mode = True
                elif not touching and drawing_mode:
                    drawing_mode = False

                # Draw the thumb and index finger landmarks
                cv2.circle(image, (thumb_x, thumb_y), 5, (0, 0, 255), -1)  # Draw thumb landmark
                cv2.circle(image, (index_x, index_y), 5, (0, 255, 0), -1)  # Draw index finger landmark

                if drawing_mode:
                    if prev_thumb_tip is not None and prev_index_tip is not None:
                        center_x = int((thumb_x + index_x) / 2)
                        center_y = int((thumb_y + index_y) / 2)
                        if prev_center is not None:
                            lines.append((prev_center, (center_x, center_y), drawing_color, drawing_thickness))
                        prev_center = (center_x, center_y)
                else:
                    prev_center = None

                prev_thumb_tip = thumb_tip
                prev_index_tip = index_tip

        for line in lines:
            cv2.line(image, line[0], line[1], line[2], line[3])

        cv2.imshow('Tes', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()