import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def finger_count(hand_landmarks):
    """
    Count how many fingers are extended (1–5).
    """
    count = 0

    # Thumb: check x-axis because it bends sideways
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    if thumb_tip.x < thumb_ip.x:  # Works for right hand (flipped webcam)
        count += 1

    # Index, Middle, Ring, Pinky: tip.y < pip.y means finger is up
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    for tip_idx, pip_idx in zip(finger_tips, finger_pips):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        if tip.y < pip.y:
            count += 1

    return count


def detect_hand_gesture():
    """Start webcam and detect finger count in real-time."""
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame from camera")
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Count fingers
                    fingers = finger_count(hand_landmarks)

                    # Show result on screen
                    cv2.putText(
                        frame,
                        f"Fingers: {fingers}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Finger Counting (1–5)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_hand_gesture()
