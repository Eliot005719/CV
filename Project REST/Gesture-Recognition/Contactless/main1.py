import cv2
import mediapipe as mp
import pyautogui
import webbrowser

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Open camera feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    left_thumbs_up = False
    right_thumbs_up = False
    hands_joined = False

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Check if thumbs are up
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            handedness = results.multi_handedness[index].classification[0].label

            if thumb_tip.y < index_tip.y:
                if handedness == 'Left':
                    left_thumbs_up = True
                elif handedness == 'Right':
                    right_thumbs_up = True

            # Check if both hands are joined
            if len(results.multi_hand_landmarks) == 2:
                distance_between_hands = abs(
                    results.multi_hand_landmarks[0].landmark[0].x
                    - results.multi_hand_landmarks[1].landmark[0].x
                )
                if distance_between_hands < 0.1:
                    hands_joined = True

    # Perform actions based on hand positions
    if hands_joined:
        if left_thumbs_up:
            webbrowser.open('https://www.youtube.com', new=2)
        if right_thumbs_up:
            webbrowser.open('https://aniwatch.me', new=2)

    # Display the frame
    cv2.imshow('Hand Gesture Control', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
hands.close()
cv2.destroyAllWindows()
