
import cv2
import numpy as np
import time
import csv
import os
from collections import deque
from datetime import datetime

# ==========================================
# 1. CAMERA CONFIGURATION
# ==========================================

CAM1_URL = "http://192.168.1.104:8080/video"
CAM2_URL = "http://192.168.1.108:8080/video"

cam1 = cv2.VideoCapture(CAM1_URL)
cam2 = cv2.VideoCapture(CAM2_URL)

if not cam1.isOpened() or not cam2.isOpened():
    print("ERROR: Camera connection failed")
    cam1.release()
    cam2.release()
    raise SystemExit

# ==========================================
# 2. ARUCO CONFIGURATION
# ==========================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary, parameters
)

AGV_ID = 10

# ==========================================
# 3. WORLD COORDINATES
# ==========================================

WORLD = {
    0:  (0.10, 0.10),
    1:  (1.40, 0.10),
    21: (0.10, 0.80),
    22: (1.40, 0.80),
    2:  (0.10, 1.40),
    3:  (1.40, 1.40)
}

CAM1_IDS = [0, 1, 21, 22]
CAM2_IDS = [2, 3, 21, 22]

H1 = None
H2 = None

# ==========================================
# 4. TRACKING CONFIGURATION
# ==========================================

ARENA_SIZE = 1.5

MAP_SIZE = 600

# Exponential smoothing factor
ALPHA = 0.25

# Store last 500 positions per camera
track1 = deque(maxlen=500)
track2 = deque(maxlen=500)

smooth1 = None
smooth2 = None

last_seen1 = None
last_seen2 = None

# ==========================================
# 5. CSV DATA LOGGING
# ==========================================

csv_file = open(
    "coverage_ai_tracking.csv",
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "timestamp",
    "camera",
    "raw_x",
    "raw_y",
    "smooth_x",
    "smooth_y"
])

# ==========================================
# 6. DETECT AND CALIBRATE
# ==========================================

def process_camera(frame, camera_name, ref_ids, H,
                   trajectory, smoothed, last_seen):

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    corners, ids, rejected = detector.detectMarkers(gray)

    centres = {}

    if ids is not None:

        ids = ids.flatten()

        cv2.aruco.drawDetectedMarkers(
            frame, corners, ids.reshape(-1, 1)
        )

        for i, marker_id in enumerate(ids):

            pts = corners[i][0]

            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

            centres[int(marker_id)] = (cx, cy)

    # ======================================
    # HOMOGRAPHY CALIBRATION
    # ======================================

    if H is None:

        if all(mid in centres for mid in ref_ids):

            image_points = np.array(
                [centres[mid] for mid in ref_ids],
                dtype=np.float32
            )

            world_points = np.array(
                [WORLD[mid] for mid in ref_ids],
                dtype=np.float32
            )

            H, _ = cv2.findHomography(
                image_points,
                world_points,
                0
            )

            if H is not None:
                print(camera_name, "CALIBRATED")

    # ======================================
    # AGV DETECTION
    # ======================================

    raw_position = None

    if AGV_ID in centres and H is not None:

        cx, cy = centres[AGV_ID]

        pixel = np.array(
            [[[cx, cy]]],
            dtype=np.float32
        )

        world_position = cv2.perspectiveTransform(
            pixel, H
        )

        x, y = world_position[0][0]

        raw_position = (float(x), float(y))

        # ==================================
        # SMOOTH POSITION
        # ==================================

        if smoothed is None:

            smoothed = np.array(
                [x, y], dtype=float
            )

        else:

            smoothed = (
                ALPHA * np.array([x, y])
                + (1 - ALPHA) * smoothed
            )

        sx, sy = smoothed

        # Only accept positions inside arena
        if 0 <= sx <= ARENA_SIZE and 0 <= sy <= ARENA_SIZE:

            trajectory.append((sx, sy))

            last_seen = time.time()

            cv2.circle(
                frame,
                (int(cx), int(cy)),
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                f"X:{sx:.3f} Y:{sy:.3f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            writer.writerow([
                datetime.now().isoformat(),
                camera_name,
                x, y,
                sx, sy
            ])

    # ======================================
    # CAMERA STATUS
    # ======================================

    status = (
        "TRACKING"
        if AGV_ID in centres and H is not None
        else "WAITING"
    )

    cv2.putText(
        frame,
        f"{camera_name}: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if status == "TRACKING"
        else (0, 0, 255),
        2
    )

    if H is None:
        cv2.putText(
            frame,
            "WAITING FOR REFERENCES",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return frame, H, trajectory, smoothed, last_seen

# ==========================================
# 7. DRAW GLOBAL ARENA MAP
# ==========================================

def draw_map():

    canvas = np.ones(
        (MAP_SIZE, MAP_SIZE, 3),
        dtype=np.uint8
    ) * 255

    margin = 35
    usable = MAP_SIZE - 2 * margin

    # Arena boundary
    cv2.rectangle(
        canvas,
        (margin, margin),
        (MAP_SIZE-margin, MAP_SIZE-margin),
        (0, 0, 0),
        3
    )

    cv2.putText(
        canvas,
        "COVERAGE-AI GLOBAL TRACKING",
        (45, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

    def convert_point(point):

        x, y = point

        px = int(margin + (x / ARENA_SIZE) * usable)
        py = int(margin + (y / ARENA_SIZE) * usable)

        return px, py

    # CAM-1 trajectory: red
    for i in range(1, len(track1)):

        p1 = convert_point(track1[i-1])
        p2 = convert_point(track1[i])

        cv2.line(
            canvas, p1, p2,
            (0, 0, 255), 2
        )

    # CAM-2 trajectory: blue
    for i in range(1, len(track2)):

        p1 = convert_point(track2[i-1])
        p2 = convert_point(track2[i])

        cv2.line(
            canvas, p1, p2,
            (255, 0, 0), 2
        )

    # Current positions
    if track1:
        cv2.circle(
            canvas,
            convert_point(track1[-1]),
            8,
            (0, 0, 255),
            -1
        )

    if track2:
        cv2.circle(
            canvas,
            convert_point(track2[-1]),
            8,
            (255, 0, 0),
            -1
        )

    cv2.putText(
        canvas,
        "CAM1 = RED",
        (45, MAP_SIZE-12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

    cv2.putText(
        canvas,
        "CAM2 = BLUE",
        (350, MAP_SIZE-12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        2
    )

    return canvas

# ==========================================
# 8. MAIN LOOP
# ==========================================

print("\nCOVERAGE-AI TRACKING STARTED")
print("Press Q to quit | R to recalibrate")

try:

    while True:

        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()

        if ret1:

            frame1, H1, track1, smooth1, last_seen1 = (
                process_camera(
                    frame1,
                    "CAM-1",
                    CAM1_IDS,
                    H1,
                    track1,
                    smooth1,
                    last_seen1
                )
            )

            cv2.imshow("CAM-1", frame1)

        if ret2:

            frame2, H2, track2, smooth2, last_seen2 = (
                process_camera(
                    frame2,
                    "CAM-2",
                    CAM2_IDS,
                    H2,
                    track2,
                    smooth2,
                    last_seen2
                )
            )

            cv2.imshow("CAM-2", frame2)

        # Show map
        arena_map = draw_map()

        cv2.imshow(
            "COVERAGE-AI - AGV TRAJECTORY",
            arena_map
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):

            H1 = None
            H2 = None

            smooth1 = None
            smooth2 = None

            track1.clear()
            track2.clear()

            print("Recalibration and tracking reset!")

finally:

    cam1.release()
    cam2.release()

    csv_file.close()

    cv2.destroyAllWindows()

    print("Tracking stopped.")
    print("Data saved to coverage_ai_tracking.csv")
