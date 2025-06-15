import cv2
import mediapipe as mp
import streamlit as st
import time
import pymongo
from datetime import datetime

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["ai_fitness"]
keypoints_collection = db["keypoints"]

statistics_collection = db["statistics"]


def extract_keypoints(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None

    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        # keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        keypoints.append({
            'x': landmark.x,
            'y': landmark.y,
            'z': landmark.z,
            'visibility': landmark.visibility
        })
    return keypoints


def draw_landmarks(image, keypoints):
    if not keypoints:
        return image

    h, w = image.shape[:2]
    landmark_points = []

    try:
        for point in keypoints:
            if isinstance(point, dict):
                x = point.get("x", 0)
                y = point.get("y", 0)
                z = point.get("z", 0)
                v = point.get("visibility", 0)
            elif isinstance(point, list) or isinstance(point, tuple):
                x, y, z = point[0], point[1], point[2] if len(point) > 2 else 0
                v = point[3] if len(point) > 3 else 1
            else:
                continue

            cx, cy = int(x * w), int(y * h)
            landmark_points.append((cx, cy))
            if v > 0.5:
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                cv2.line(image, landmark_points[start_idx], landmark_points[end_idx], (0, 255, 255), 2)

    except Exception as e:
        print(f"Error drawing landmarks: {e}")
        return image

    return image


def load_reference_keypoints(us_id, ex_id):
    frames = keypoints_collection.find({"user_id": us_id, "exercise_id": ex_id}).sort("frame_index")
    return [frame["keypoints"] for frame in frames]


def save_keypoints(user_id, exercise_id, attempt_id, frame_index, keypoints):
    keypoints_collection.insert_one({
        "user_id": user_id,
        "exercise_id": exercise_id,
        "attempt_id": attempt_id,
        "frame_index": frame_index,
        "timestamp": datetime.utcnow(),
        "keypoints": keypoints
    })


def save_statistics(user_id, exercise_id, attempt_id, stats):
    statistics_collection.insert_one({
        "user_id": user_id,
        "exercise_id": exercise_id,
        "attempt_id": attempt_id,
        "timestamp": datetime.utcnow(),
        **stats
    })


st.title("ИИ Фитнес Трекер")
user_id = st.text_input("Введите имя пользователя")
exercise_id = st.selectbox("Выберите упражнение", ["squat", "yoga1"])

if st.button("Старт"):
    attempt_id = str(int(time.time()))
    cap_video_path = "data/test_video.mp4"
    cap = cv2.VideoCapture(cap_video_path)
    ref_keypoints = load_reference_keypoints("reference", exercise_id)
    ref_video_path = f"assets/{exercise_id}_reference.mp4"
    ref_video = cv2.VideoCapture(ref_video_path)
    frame_index = 0
    attempt_keypoints_seq = []

    col1, col2 = st.columns(2)
    st_frame_cam = col1.empty()
    st_frame_ref = col2.empty()

    while cap.isOpened() and ref_video.isOpened():
        ret_cam, frame_cam = cap.read()
        ret_ref, frame_ref = ref_video.read()

        if not ret_cam or not ret_ref:
            break

        # Обработка пользовательского видео
        keypoints = extract_keypoints(frame_cam)
        save_keypoints(user_id, exercise_id, attempt_id, frame_index, keypoints)
        attempt_keypoints_seq.append(keypoints)
        frame_cam = draw_landmarks(frame_cam, keypoints)

        # Обработка эталонного видео с ключевыми точками из базы
        ref_kp = ref_keypoints[frame_index] if frame_index < len(ref_keypoints) else None
        frame_ref = draw_landmarks(frame_ref, ref_kp)
        st_frame_cam.image(cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB), channels="RGB", caption="Your Video",
                           use_column_width=True)
        st_frame_ref.image(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB), channels="RGB", caption="Reference Video",
                           use_column_width=True)

        frame_index += 1

    cap.release()
    ref_video.release()

    st.session_state["user_id"] = user_id
    st.session_state["exercise_id"] = exercise_id
    st.session_state["attempt_id"] = attempt_id

    st.success('Перейдите в раздел estimate')

    # st.markdown(
    #     "<meta http-equiv='refresh' content='0; url=http://localhost:8502/'>",
    #     unsafe_allow_html=True
    # )

    # st.rerun()
    # if st.button("Проведение анализа"):
    #     ref_kp_flat = [np.array(frame).flatten() for frame in ref_keypoints[:frame_index]]
    #     attempt_kp_flat = [np.array(frame).flatten() for frame in attempt_keypoints_seq[:frame_index]]
    #
    #     stats = compute_statistics(ref_kp_flat, attempt_kp_flat)
    #     save_statistics(user_id, exercise_id, attempt_id, stats)
    #
    #     st.subheader("Statistics")
    #     dtw_distance = stats.get('dtw_distance', None)
    #     if dtw_distance is not None:
    #         st.write(f"DTW Distance: {dtw_distance:.2f}")
    #     else:
    #         st.write("DTW Distance: N/A")
    #     quality_score = stats.get('quality_score', None)
    #     if quality_score is not None:
    #         st.write(f"Quality Score: {quality_score:.2f}%")
    #     else:
    #         st.write("Quality Score: N/A")
    #     st.write(f"Path Length: {stats['path_length']}")
    #
    #     st.success("Attempt completed and saved!")
