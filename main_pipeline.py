import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
from scipy.signal import savgol_filter
import json

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8m.pt"

# Настройки
MIN_TRACK_LENGTH = 20
MIN_EVENTS_PER_TRACK = 1
AGGRESSIVE_WINDOW = 10


def process_video(input_path: Path, output_dir: Path) -> dict:
    output_video = output_dir / "output.mp4"
    report_path = output_dir / "report.json"

    # Инициализация
    print("Загрузка моделей...")
    model = YOLO(MODEL_PATH)

    tracker = sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8
    )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker.frame_rate = fps

    temp_video = output_video.with_name(
        output_video.stem + "_temp" + output_video.suffix
    )
    temp_out = cv2.VideoWriter(
        str(temp_video), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
    )
    if not temp_out.isOpened():
        raise RuntimeError("Не удалось открыть VideoWriter для временного видео.")

    trajectories = {}
    frame_idx = 0

    # Первый проход: Детекция и Трекинг
    print("Трекинг объектов...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, classes=[2, 3, 5, 7], conf=0.5, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        tracks = tracker.update_with_detections(detections)

        for tid, bbox in zip(tracks.tracker_id, tracks.xyxy):
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if tid not in trajectories:
                trajectories[tid] = {
                    "frames": [],
                    "centers": [],
                    "boxes": [],
                    "events": []
                }

            trajectories[tid]["frames"].append(frame_idx)
            trajectories[tid]["centers"].append((cx, cy))
            trajectories[tid]["boxes"].append((x1, y1, x2, y2))

        temp_out.write(frame)
        frame_idx += 1

    cap.release()
    temp_out.release()

    # Анализ Агрессивности
    print("Анализ агрессивного вождения...")
    aggressive_tracks = set()

    for tid, data in trajectories.items():
        if len(data["centers"]) < MIN_TRACK_LENGTH:
            continue

        v, a, j = compute_kinematics(data["centers"], fps)
        events = detect_aggressive_events(v, fps)

        if len(events) >= MIN_EVENTS_PER_TRACK:
            data["events"] = [data["frames"][i + 3] for i in events]
            aggressive_tracks.add(tid)

    print(f"Агрессивных треков: {len(aggressive_tracks)} / {len(trajectories)}")

    # Второй проход: Аннотация
    cap = cv2.VideoCapture(str(temp_video))
    out = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
    )
    if not out.isOpened():
        raise RuntimeError("Не удалось открыть VideoWriter для итогового видео.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tid, data in trajectories.items():
            if frame_idx not in data["frames"]:
                continue

            idx = data["frames"].index(frame_idx)
            x1, y1, x2, y2 = map(int, data["boxes"][idx])

            color = (0, 255, 0)
            for ev in data["events"]:
                if abs(frame_idx - ev) <= AGGRESSIVE_WINDOW:
                    color = (0, 0, 255)
                    break

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    temp_video.unlink()

    # Итог
    print("\nОбнаруженные случаи агрессивного вождения:")
    total_events = 0

    for tid, data in trajectories.items():
        if data["events"]:
            print(f"  ТС ID {tid}: {len(data['events'])} агрессивных события")
            total_events += len(data["events"])

    print(f"\nВсего агрессивных событий: {total_events}")

    report = {
        "output_video": input_path.name,
        "total_tracks": len(trajectories),
        "aggressive_tracks": len(aggressive_tracks),
        "total_events": total_events,
        "events": {
            int(tid): len(data["events"])
            for tid, data in trajectories.items()
            if data["events"]
        }
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "video": "output.mp4",
        "report": "report.json",
        "total_tracks": len(trajectories),
        "aggressive_tracks": len(aggressive_tracks),
        "total_events": total_events,
        "input_video_name": input_path.name
    }


# Вспомогательные функции
def smooth_trajectory(points):
    n = len(points)
    if n < 7:
        return np.array(points)

    window = min(8, n if n % 2 == 1 else n - 1)
    return np.column_stack([
        savgol_filter(np.array(points)[:, 0], window, 1),
        savgol_filter(np.array(points)[:, 1], window, 1)
    ])


def compute_kinematics(centers, fps):
    centers = np.array(centers)
    centers = smooth_trajectory(centers)

    dt = 1.0 / fps
    v = np.diff(centers, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    j = np.diff(a, axis=0) / dt
    return v, a, j


def detect_aggressive_events(v, fps):

    # Параметры
    angle_threshold = 8.0
    curv_threshold = 250.0
    speed_change_threshold = 0.4
    decel_threshold = 450.0
    ttc_threshold = 0.5
    window = max(1, int(0.2 * fps))
    min_consecutive = max(3, int(0.4 * fps))
    score_threshold = 2
    min_speed = 40.0

    # Базовые величины
    speed = np.linalg.norm(v, axis=1) + 1e-6
    v_unit = v / speed[:, None]

    # Углы
    angles = []
    for i in range(len(v_unit) - window):
        dot = np.clip(
            np.dot(v_unit[i], v_unit[i + window]),
            -1.0, 1.0
        )
        angle = np.degrees(np.arccos(dot))
        angles.append(angle)
    angles = np.array(angles)

    # Кривизна
    curvature = angles * speed[:len(angles)]

    # Изменение скорости
    speed_change = np.abs(np.diff(speed)) / speed[:-1]
    speed_change = speed_change[:len(angles)]
    valid = speed[:-window] > min_speed

    # Резкое торможение
    decel = -np.diff(speed) * fps
    decel = np.maximum(decel, 0)
    decel = decel[:len(angles)]

    # Приблизительный TTC
    ttc = speed[:len(decel)] / (decel + 1e-6)

    # Оценка по-баллам
    flags = []
    for i in range(len(angles)):
        if not valid[i]:
            flags.append(False)
            continue

        score = 0

        if angles[i] > angle_threshold:
            score += 2

        if curvature[i] > curv_threshold:
            score += 1

        if speed_change[i] > speed_change_threshold:
            score += 1

        if decel[i] > decel_threshold:
            score += 2

        if ttc[i] < ttc_threshold:
            score += 1

        flags.append(score >= score_threshold)

    # Поиск устойчивых событий
    events = []
    count = 0

    for i, f in enumerate(flags):
        if f:
            count += 1
        else:
            if count >= min_consecutive:
                events.append(i)
            count = 0

    if count >= min_consecutive:
        events.append(len(flags) - 1)

    return events
