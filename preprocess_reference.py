import cv2
import mediapipe as mp
import time
import uuid
from database import save_keypoints_to_db


def extract_keypoints_from_frame(results):
    """Извлекает ключевые точки из результата mediapipe"""
    if not results.pose_landmarks:
        return []

    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.append({
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": landmark.visibility
        })
    return keypoints


def preprocess_reference_video(video_path: str, exercise_id: str):
    cap = cv2.VideoCapture(video_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    user_id = "reference"
    attempt_id = str(uuid.uuid4())
    frame_index = 0
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Переводим изображение в RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Получаем результаты
        results = pose.process(image_rgb)
        keypoints = extract_keypoints_from_frame(results)

        # Время относительно начала обработки
        timestamp = time.time() - start_time

        # Сохраняем в базу данных
        save_keypoints_to_db({
            "user_id": user_id,
            "exercise_id": exercise_id,
            "attempt_id": attempt_id,
            "frame_index": frame_index,
            "timestamp": timestamp,
            "keypoints": keypoints
        })

        frame_index += 1

    cap.release()
    pose.close()
    print(f"[INFO] Reference video processed and saved with attempt_id: {attempt_id}")


preprocess_reference_video("assets/yoga1_reference.mp4", "yoga1")
