
import cv2
import numpy as np
import time

# ==========================================
# 1. CAMERA CONFIGURATION
# ==========================================

CAM1_URL = "http://192.168.137.97:8080/video"
CAM2_URL = "http://192.168.137.162:8080/video"

cam1 = cv2.VideoCapture(CAM1_URL)
cam2 = cv2.VideoCapture(CAM2_URL)

if not cam1.isOpened() or not cam2.isOpened():
    print("ERROR: Could not connect to both cameras.")
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
# 3. GLOBAL WORLD COORDINATES
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

# Independent homography matrices
H1 = None
H2 = None

# ==========================================
# 4. CAMERA PROCESSING FUNCTION
# ==========================================

def process_camera(frame, camera_name, reference_ids, H):

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    corners, ids, rejected = detector.detectMarkers(gray)

    marker_centres = {}

    if ids is not None:

        ids = ids.flatten()

        for i, marker_id in enumerate(ids):

            marker_id = int(marker_id)

            pts = corners[i][0]

            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

            marker_centres[marker_id] = (cx, cy)

            # Draw every detected marker
            cv2.polylines(
                frame,
                [pts.astype(int)],
                True,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {marker_id}",
                (int(cx), int(cy) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # ======================================
    # 5. CALCULATE HOMOGRAPHY
    # ======================================

    if H is None:

        if all(mid in marker_centres for mid in reference_ids):

            image_points = np.array(
                [marker_centres[mid] for mid in reference_ids],
                dtype=np.float32
            )

            world_points = np.array(
                [WORLD[mid] for mid in reference_ids],
                dtype=np.float32
            )

            H, mask = cv2.findHomography(
                image_points,
                world_points,
                method=0
            )

            if H is not None:

                print(
                    f"\n{camera_name}: HOMOGRAPHY CALIBRATED"
                )

                print(H)

    # ======================================
    # 6. DETECT AGV AND MAP TO WORLD
    # ======================================

    agv_position = None

    if AGV_ID in marker_centres:

        cx, cy = marker_centres[AGV_ID]

        cv2.circle(
            frame,
            (int(cx), int(cy)),
            7,
            (0, 0, 255),
            -1
        )

        if H is not None:

            pixel_point = np.array(
                [[[cx, cy]]],
                dtype=np.float32
            )

            world_point = cv2.perspectiveTransform(
                pixel_point,
                H
            )

            X, Y = world_point[0][0]

            agv_position = (float(X), float(Y))

            cv2.putText(
                frame,
                f"AGV X: {X:.3f}m Y: {Y:.3f}m",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    # ======================================
    # 7. STATUS DISPLAY
    # ======================================

    missing = [
        mid for mid in reference_ids
        if mid not in marker_centres
    ]

    if H is None:

        status = "WAITING FOR REFERENCE MARKERS"

    else:

        status = "HOMOGRAPHY READY"

    color = (
        (0, 255, 0)
        if H is not None
        else (0, 0, 255)
    )

    cv2.putText(
        frame,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    cv2.putText(
        frame,
        f"Missing refs: {missing}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2
    )

    if AGV_ID not in marker_centres:
        cv2.putText(
            frame,
            "AGV ID 10 NOT VISIBLE",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )

    return frame, H, agv_position


# ==========================================
# 8. MAIN LOOP
# ==========================================

print("\nCOVERAGE-AI HOMOGRAPHY SYSTEM")
print("Waiting for calibration markers...")
print("Press Q to quit | R to recalibrate")

while True:

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()

    pos1 = None
    pos2 = None

    if ret1:

        frame1, H1, pos1 = process_camera(
            frame1,
            "CAM-1",
            CAM1_IDS,
            H1
        )

        cv2.imshow("COVERAGE-AI CAM-1", frame1)

    if ret2:

        frame2, H2, pos2 = process_camera(
            frame2,
            "CAM-2",
            CAM2_IDS,
            H2
        )

        cv2.imshow("COVERAGE-AI CAM-2", frame2)

    # ======================================
    # 9. DISPLAY GLOBAL AGV POSITION
    # ======================================

    if pos1 is not None and pos2 is not None:

        print(
            f"\nCAM1: X={pos1[0]:.3f}, Y={pos1[1]:.3f}"
        )

        print(
            f"CAM2: X={pos2[0]:.3f}, Y={pos2[1]:.3f}"
        )

        difference = np.linalg.norm(
            np.array(pos1) - np.array(pos2)
        )

        print(f"Camera disagreement: {difference:.3f} m")

    elif pos1 is not None:

        print(f"CAM1 GLOBAL POSITION: {pos1}")

    elif pos2 is not None:

        print(f"CAM2 GLOBAL POSITION: {pos2}")

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('r'):

        H1 = None
        H2 = None

        print("\nRecalibration requested!")

cam1.release()
cam2.release()

cv2.destroyAllWindows()

print("COVERAGE-AI stopped.")
