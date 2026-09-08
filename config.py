from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "yolov8m.pt"


# Настройки анализа траекторий
MIN_TRACK_LENGTH = 20
MIN_EVENTS_PER_TRACK = 1
AGGRESSIVE_WINDOW = 10


# Пороговые значения детектора агрессивного вождения
ANGLE_THRESHOLD = 8.0
CURV_THRESHOLD = 250.0
SPEED_CHANGE_THRESHOLD = 0.4
DECEL_THRESHOLD = 450.0
TTC_THRESHOLD = 0.5
SCORE_THRESHOLD = 2

MIN_SPEED = 40.0

# Параметры временного анализа
EVENT_WINDOW_SECONDS = 0.2
MIN_CONSECUTIVE_SECONDS = 0.4

# Параметры детекции YOLO
DETECTION_CONFIDENCE = 0.5
DETECTION_CLASSES = [2, 3, 5, 7]


# Параметры ByteTrack
TRACK_ACTIVATION_THRESHOLD = 0.5
LOST_TRACK_BUFFER = 30
MINIMUM_MATCHING_THRESHOLD = 0.8