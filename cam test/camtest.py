
import cv2
import numpy as np
import threading
import time

# ==========================================
# COVERAGE-AI CAMERA CONFIGURATION
# ==========================================

CAM1_URL = "http://192.168.137.97:8080/video"
CAM2_URL = "http://192.168.137.162:8080/video"

TARGET_ID = 10

# ==========================================
# ARUCO INITIALIZATION
# ==========================================

dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)

# ==========================================
# CAMERA STREAM CLASS
# ==========================================

class CameraStream:

    def __init__(self, name, url):

        self.name = name
        self.url = url

        self.cap = cv2.VideoCapture(url)

        self.frame = None
        self.running = True

        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self.update,
            daemon=True
        )

        self.thread.start()

    def update(self):

        while self.running:

            ret, frame = self.cap.read()

            if ret:

                with self.lock:
                    self.frame = frame

            else:
                time.sleep(0.05)

    def get_frame(self):

        with self.lock:
            if self.frame is None:
                return None

            return self.frame.copy()

    def release(self):

        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# ==========================================
# CONNECT BOTH CAMERAS
# ==========================================

print("Connecting to cameras...")

cam1 = CameraStream("CAM-1", CAM1_URL)
cam2 = CameraStream("CAM-2", CAM2_URL)

time.sleep(2)

# ==========================================
# DETECT AGV ID 10
# ==========================================

def detect_agv(frame, camera_name):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    corners, ids, rejected = detector.detectMarkers(
        gray
    )

    detected = False

    if ids is not None:

        for i, marker_id in enumerate(ids.flatten()):

            if marker_id == TARGET_ID:

                detected = True

                pts = corners[i][0].astype(int)

                # Marker centre
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))

                # Draw marker boundary
                cv2.polylines(
                    frame,
                    [pts],
                    True,
                    (0, 255, 0),
                    3
                )

                # Draw centre
                cv2.circle(
                    frame,
                    (cx, cy),
                    7,
                    (0, 0, 255),
                    -1
                )

                # Display ID
                cv2.putText(
                    frame,
                    "AGV ID: 10",
                    (cx - 50, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                print(
                    f"{camera_name} DETECTED AGV "
                    f"| Center: ({cx}, {cy})"
                )

    # Camera status
    status = (
        "AGV DETECTED"
        if detected
        else "AGV NOT DETECTED"
    )

    color = (
        (0, 255, 0)
        if detected
        else (0, 0, 255)
    )

    cv2.putText(
        frame,
        f"{camera_name}: {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    return frame, detected


# ==========================================
# MAIN LOOP
# ==========================================

print("\nCOVERAGE-AI AGV TRACKING TEST")
print("Looking for ArUco ID 10...")
print("Press Q to exit.\n")

while True:

    frame1 = cam1.get_frame()
    frame2 = cam2.get_frame()

    detected1 = False
    detected2 = False

    if frame1 is not None:

        frame1, detected1 = detect_agv(
            frame1,
            "CAM-1"
        )

        cv2.imshow("COVERAGE-AI | CAM-1", frame1)

    if frame2 is not None:

        frame2, detected2 = detect_agv(
            frame2,
            "CAM-2"
        )

        cv2.imshow("COVERAGE-AI | CAM-2", frame2)

    # Combined detection status
    if detected1 and detected2:
        print("BOTH CAMERAS SEE THE AGV!")

    elif detected1:
        print("AGV VISIBLE IN CAM-1 ONLY")

    elif detected2:
        print("AGV VISIBLE IN CAM-2 ONLY")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
cam2.release()

cv2.destroyAllWindows()

print("COVERAGE-AI test stopped.")
