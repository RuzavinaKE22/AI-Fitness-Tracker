import streamlit as st
import pymongo
import numpy as np
import time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["ai_fitness"]
collection = db["keypoints"]


def get_sequence_user(user_id, attempt_id):
    docs = list(collection.find({"user_id": user_id, "attempt_id": attempt_id}).sort("frame_index"))
    return [doc['keypoints'] for doc in docs]


def get_sequence_ref(user_id, exercise_id):
    docs = list(collection.find({"user_id": user_id, "exercise_id": exercise_id}).sort("frame_index"))
    return [doc['keypoints'] for doc in docs]


def normalize_keypoints(kps):
    kp = np.array([[p['x'], p['y'], p['z']] for p in kps])
    center = (kp[23] + kp[24]) / 2  # бедра
    scale = np.linalg.norm(kp[11] - kp[12])  # плечи
    return (kp - center) / scale


def angle_between_points(a, b, c):
    ab = np.array(a) - np.array(b)
    cb = np.array(c) - np.array(b)
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def extract_angles(kps_sequence):
    angles = []
    for kps in kps_sequence:
        kp = normalize_keypoints(kps)
        frame_angles = [
            angle_between_points(kp[11], kp[13], kp[15]),
            angle_between_points(kp[12], kp[14], kp[16]),
            angle_between_points(kp[23], kp[25], kp[27]),
            angle_between_points(kp[24], kp[26], kp[28]),
            angle_between_points(kp[11], kp[23], kp[25]),
            angle_between_points(kp[12], kp[24], kp[26]),
        ]
        angles.append(frame_angles)
    return np.array(angles)


def transform(user_data, shift, bias, scale_time=1.0):
    transformed = user_data + bias
    t = np.arange(len(user_data))
    t_scaled = t * scale_time + shift
    t_scaled = np.clip(t_scaled, 0, len(user_data) - 1)
    interpolated = np.array([
        np.interp(t, t_scaled, transformed[:, j], left=transformed[0, j], right=transformed[-1, j])
        for j in range(user_data.shape[1])
    ]).T
    return interpolated


def loss_fn(transformed, ref_data):
    min_len = min(len(transformed), len(ref_data))
    return np.mean((transformed[:min_len] - ref_data[:min_len]) ** 2)


def optimize_user_angles(user_angles, ref_angles, lr=0.0001, n_iters=50000, tolerance=1e-4):
    shift, bias, scale_time = 0.0, 0.0, 1.0
    # prev_loss = float('inf')
    losses = []

    for i in range(n_iters):
        transformed = transform(user_angles, shift, bias, scale_time)
        loss = loss_fn(transformed, ref_angles)
        losses.append(loss)

        if loss < 1800:
            break
        # prev_loss = loss

        eps = 1e-5
        grad_shift = (loss_fn(transform(user_angles, shift + eps, bias, scale_time), ref_angles) - loss) / eps
        grad_bias  = (loss_fn(transform(user_angles, shift, bias + eps, scale_time), ref_angles) - loss) / eps
        grad_stime = (loss_fn(transform(user_angles, shift, bias, scale_time + eps), ref_angles) - loss) / eps

        shift -= lr * grad_shift
        bias -= lr * grad_bias
        scale_time -= lr * grad_stime

        # loss_tmp = loss_fn(transform(user_angles, shift_tmp, bias_tmp, stime_tmp), ref_angles)
        #
        # grad_shift_2 = (loss_fn(transform(user_angles, shift_tmp + eps, bias_tmp, stime_tmp), ref_angles) - loss_tmp) / eps
        # grad_bias_2  = (loss_fn(transform(user_angles, shift_tmp, bias_tmp + eps, stime_tmp), ref_angles) - loss_tmp) / eps
        # grad_stime_2 = (loss_fn(transform(user_angles, shift_tmp, bias_tmp, stime_tmp + eps), ref_angles) - loss_tmp) / eps
        #
        # shift      -= lr * grad_shift_2
        # bias       -= lr * grad_bias_2
        # scale_time -= lr * grad_stime_2

    final_transformed = transform(user_angles, shift, bias, scale_time)
    return final_transformed, (shift, bias, scale_time), losses, i + 1  # количество итераций


def crop_sequences_to_compare(ref_data, user_transformed, scale_time):
    if scale_time >= 1.0:
        min_len = min(len(ref_data), len(user_transformed))
        return ref_data[:min_len], user_transformed[:min_len]
    else:
        min_len = len(user_transformed)
        return ref_data[:min_len], user_transformed


st.title("Оценка качества выполнения упражнения")


with st.sidebar:
    if 'user_id' not in st.session_state:
        user_id = st.text_input("Имя пользователя", "k")
    else:
        user_id = st.text_input("Имя пользователя", st.session_state['user_id'])

    if 'exercise_id' not in st.session_state:
        exercise_id = st.text_input("Эталонное упражнение", "squat")
    else:
        exercise_id = st.text_input("Эталонное упражнение", st.session_state['exercise_id'])

    if 'attempt_id' not in st.session_state:
        attempt_id = st.text_input("ID попытки", "1744629000")
    else:
        attempt_id = st.text_input("ID попытки", st.session_state['attempt_id'])

    run = st.button("Сравнить")

if run:
    start_time = time.time()
    with st.spinner("Загружаем данные и обрабатываем..."):
        ref_kps = get_sequence_ref("reference", exercise_id)
        user_kps = get_sequence_user(user_id, attempt_id)

        ref_angles = extract_angles(ref_kps)
        user_angles = extract_angles(user_kps)

        optimized_user_angles, (shift, bias, scale_time), losses, num_iters = optimize_user_angles(user_angles, ref_angles)

        ref_cropped, user_cropped = crop_sequences_to_compare(ref_angles, optimized_user_angles, scale_time)

        mape_per_joint = mean_absolute_percentage_error(ref_cropped, user_cropped, multioutput='raw_values')
        mape_score = np.mean(mape_per_joint)

        angle_diff = np.abs(np.mean(ref_cropped, axis=0) - np.mean(user_cropped, axis=0))

        final_score = max(0, 100 - mape_score)

    st.subheader("Результаты")
    st.metric("Оценка качества выполнения", f"{final_score:.2f} %")
    st.write("Параметры трансформации:")
    st.markdown(f"- Сдвиг по времени (shift): **{shift:.2f}**")
    st.markdown(f"- Смещение углов (bias): **{bias:.2f}**")
    st.markdown(f"- Масштаб времени (scale_time): **{scale_time:.4f}**")

    st.subheader("Средние отклонения углов (в градусах):")
    joints = ["Правый локоть", "Левый локоть", "Правое колено", "Левое колено", "Правое бедро", "Левое бедро"]
    for j, diff in zip(joints, angle_diff):
        st.write(f"- {j}: {diff:.2f}°")

    st.subheader("MAPE по суставам (в %):")
    for j, mape_val in zip(joints, mape_per_joint):
        st.write(f"- {j}: {mape_val:.2f}%")
    st.write(f"**Средний MAPE по всем суставам:** {mape_score:.2f}%")

    st.subheader("Графики углов до трансформации")
    for i, joint in enumerate(joints):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ref_angles[:, i], label="Эталон", linestyle='--', color='blue')
        ax.plot(user_angles[:, i], label="Пользователь", color='orange')
        ax.set_title(f"{joint}")
        ax.set_xlabel("Кадр")
        ax.set_ylabel("Угол (°)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("Графики углов после трансформации")
    for i, joint in enumerate(joints):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ref_cropped[:, i], label="Эталон", linestyle='--', color='blue')
        ax.plot(user_cropped[:, i], label="Пользователь (трансформированный)", color='green')
        ax.set_title(f"{joint}")
        ax.set_xlabel("Кадр")
        ax.set_ylabel("Угол (°)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    st.subheader("Сходимость функции потерь по итерациям")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color='purple')
    ax.set_xlabel("Итерация")
    ax.set_ylabel("MSE")
    ax.grid(True)
    st.pyplot(fig)

    elapsed = time.time() - start_time
    st.info(f"Время выполнения: {elapsed:.2f} сек")
    st.info(f"Итераций градиентного спуска: {num_iters}")

    st.session_state["user_id"] = user_id
    st.session_state["exercise_id"] = exercise_id
    st.session_state["attempt_id"] = attempt_id

    st.success("Сравнение завершено!")
