"""
Real-Time Facial Recognition Attendance System Using Machine Learning
======================================================================
Course: MCA Minor Project - Semester 3
Institution: Amity University Online
Language: Python 3 (Compatible with Python 3.9 - 3.13+)
Platform: Microsoft Visual Studio / Visual Studio Code / Windows / macOS
======================================================================
"""

import csv
import os
import sys
import time
from datetime import datetime

# ----------------------------------------------------------------------
# 1. PREREQUISITE VALIDATION
# ----------------------------------------------------------------------
MISSING_PACKAGES = []

try:
    import cv2
except ImportError:
    MISSING_PACKAGES.append("opencv-contrib-python")

try:
    import numpy as np
except ImportError:
    MISSING_PACKAGES.append("numpy")

if MISSING_PACKAGES:
    print("\n" + "=" * 70)
    print(" [!] REQUIRED PACKAGES MISSING")
    print("=" * 70)
    print("The following required Python libraries are not installed:")
    for pkg in MISSING_PACKAGES:
        print(f"   * {pkg}")
    print("\nPlease install them by opening Developer PowerShell and running:")
    print(f"   pip install {' '.join(MISSING_PACKAGES)}")
    print("=" * 70 + "\n")
    sys.exit(1)


# ----------------------------------------------------------------------
# 2. DIRECTORIES & CONFIGURATION SETUP
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ATTENDANCE_DIR = os.path.join(PROJECT_ROOT, "attendance")

for directory_path in [DATASET_DIR, MODEL_DIR, ATTENDANCE_DIR]:
    os.makedirs(directory_path, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "knn_face_model.npz")
STUDENTS_REGISTRY_FILE = os.path.join(DATASET_DIR, "students_registry.csv")

FACE_WIDTH = 64
FACE_HEIGHT = 64
SAMPLES_PER_STUDENT = 40
KNN_K_NEIGHBORS = 5
DISTANCE_THRESHOLD = 3200.0


# ----------------------------------------------------------------------
# 3. RESILIENT UNIVERSAL FACE DETECTOR
# ----------------------------------------------------------------------
class UniversalFaceDetector:
    def __init__(self):
        self.cascade = None
        self.mode = "NONE"
        self._init_detector()

    def _init_detector(self):
        cascade_ctor = None
        if hasattr(cv2, "CascadeClassifier"):
            cascade_ctor = cv2.CascadeClassifier
        elif hasattr(cv2, "xobjdetect") and hasattr(cv2.xobjdetect, "CascadeClassifier"):
            cascade_ctor = cv2.xobjdetect.CascadeClassifier
        elif hasattr(cv2, "objdetect") and hasattr(cv2.objdetect, "CascadeClassifier"):
            cascade_ctor = cv2.objdetect.CascadeClassifier

        if cascade_ctor is not None:
            search_paths = []
            if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                search_paths.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
            search_paths.extend([
                os.path.join(PROJECT_ROOT, "haarcascade_frontalface_default.xml"),
                "haarcascade_frontalface_default.xml"
            ])

            for xml_path in search_paths:
                if xml_path and os.path.exists(xml_path):
                    try:
                        c = cascade_ctor(xml_path)
                        if hasattr(c, "empty") and not c.empty():
                            self.cascade = c
                            self.mode = "HAAR_CASCADE"
                            return
                    except Exception:
                        continue

            try:
                if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
                    c = cascade_ctor(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    if hasattr(c, "empty") and not c.empty():
                        self.cascade = c
                        self.mode = "HAAR_CASCADE"
                        return
            except Exception:
                pass

        self.mode = "ADAPTIVE_ROI"

    def detect_faces(self, gray_frame):
        if self.mode == "HAAR_CASCADE" and self.cascade is not None:
            try:
                faces = self.cascade.detectMultiScale(
                    gray_frame,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(80, 80)
                )
                if len(faces) > 0:
                    return faces
            except Exception:
                pass

        h, w = gray_frame.shape
        box_size = int(min(h, w) * 0.55)
        x = (w - box_size) // 2
        y = (h - box_size) // 2
        return [(x, y, box_size, box_size)]


detector = UniversalFaceDetector()


# ----------------------------------------------------------------------
# 4. CAMERA ACCESS HELPER
# ----------------------------------------------------------------------
def get_working_camera():
    for idx in [0, 1]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    return None


# ----------------------------------------------------------------------
# 5. MACHINE LEARNING: K-NEAREST NEIGHBORS (KNN) CLASSIFIER
# ----------------------------------------------------------------------
class FaceRecognitionKNN:
    def __init__(self, k=5, threshold=3200.0):
        self.k = int(k)
        self.threshold = float(threshold)
        self.x_train = None
        self.y_train = None
        self.id_to_name = {}

    def fit(self, features, labels, name_mapping):
        self.x_train = np.array(features, dtype=np.float32)
        self.y_train = np.array(labels)
        self.id_to_name = dict(name_mapping)

    def predict(self, face_vector):
        if self.x_train is None or len(self.x_train) == 0:
            return None, "Model Not Trained", float("inf"), False

        vec = np.array(face_vector, dtype=np.float32)
        distances = np.linalg.norm(self.x_train - vec, axis=1)

        k_val = min(self.k, len(distances))
        k_indices = np.argsort(distances)[:k_val]
        k_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]

        avg_distance = float(np.mean(k_distances))
        unique_labels, counts = np.unique(k_labels, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]

        if avg_distance < self.threshold:
            name = self.id_to_name.get(majority_label, "Unknown")
            return majority_label, name, avg_distance, True
        return "UNKNOWN", "Unknown Person", avg_distance, False

    def save(self, filepath):
        np.savez_compressed(
            filepath,
            x_train=self.x_train,
            y_train=self.y_train,
            k=np.array([self.k], dtype=np.int32),
            threshold=np.array([self.threshold], dtype=np.float32),
            ids=list(self.id_to_name.keys()),
            names=list(self.id_to_name.values())
        )

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False
        try:
            data = np.load(filepath, allow_pickle=True)
            self.x_train = data["x_train"]
            self.y_train = data["y_train"]
            self.k = int(data["k"][0])
            self.threshold = float(data["threshold"][0])
            ids = list(data["ids"])
            names = list(data["names"])
            self.id_to_name = dict(zip(ids, names))
            return True
        except Exception:
            return False


# ----------------------------------------------------------------------
# 6. IMAGE PREPROCESSING & REGISTRY
# ----------------------------------------------------------------------
def preprocess_face(face_roi):
    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi
    resized = cv2.resize(gray, (FACE_WIDTH, FACE_HEIGHT), interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    return equalized.flatten().astype(np.float32)


def update_registry(student_id, student_name, department):
    file_exists = os.path.exists(STUDENTS_REGISTRY_FILE)
    with open(STUDENTS_REGISTRY_FILE, mode="a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(["StudentID", "StudentName", "Department", "EnrolledAt"])
        writer.writerow([student_id, student_name, department, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])


def load_registry_map():
    mapping = {}
    if not os.path.exists(STUDENTS_REGISTRY_FILE):
        return mapping
    with open(STUDENTS_REGISTRY_FILE, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)
        for row in rows[1:]:
            if len(row) >= 2:
                mapping[row[0].strip()] = row[1].strip()
    return mapping


# ----------------------------------------------------------------------
# 7. MODULE 1: STUDENT ENROLLMENT
# ----------------------------------------------------------------------
def register_new_student():
    print("\n" + "=" * 65)
    print(" [MODULE 1] STUDENT ENROLLMENT & DATASET CAPTURE")
    print("=" * 65)

    student_id = input("Enter Student ID / Roll No (e.g., AMU101): ").strip().upper()
    if not student_id:
        print("[!] Error: Student ID cannot be empty.")
        return

    student_name = input("Enter Student Full Name: ").strip()
    if not student_name:
        print("[!] Error: Student Name cannot be empty.")
        return

    department = input("Enter Department / Program (e.g., MCA / BCA / CMA): ").strip() or "General"
    student_folder = os.path.join(DATASET_DIR, student_id)
    os.makedirs(student_folder, exist_ok=True)

    print("\n[*] Initializing camera... Look directly into your webcam.")
    print("[*] Keep your face inside the bounding box.")
    print("[*] Press 'q' to stop early.\n")

    camera = get_working_camera()
    if camera is None:
        print("[!] Error: Could not access webcam. Please verify camera permissions.")
        return

    captured_count = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(gray_frame)
        display_frame = frame.copy()

        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = gray_frame[y : y + h, x : x + w]

            if captured_count < SAMPLES_PER_STUDENT:
                img_path = os.path.join(student_folder, f"{student_id}_{captured_count + 1}.jpg")
                resized_sample = cv2.resize(face_roi, (128, 128))
                cv2.imwrite(img_path, resized_sample)
                captured_count += 1
                time.sleep(0.08)

            cv2.putText(display_frame, f"Samples: {captured_count}/{SAMPLES_PER_STUDENT}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display_frame, f"Enrolling: {student_name} ({student_id})", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.imshow("Enrollment - Amity Facial Attendance System", display_frame)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (captured_count >= SAMPLES_PER_STUDENT):
            break

    camera.release()
    cv2.destroyAllWindows()

    if captured_count >= SAMPLES_PER_STUDENT:
        update_registry(student_id, student_name, department)
        print(f"\n[+] Success! Captured {captured_count} samples for {student_name}.")
        print("[+] Next, run Option [2] to train the machine learning model.")


# ----------------------------------------------------------------------
# 8. MODULE 2: MODEL TRAINING
# ----------------------------------------------------------------------
def train_machine_learning_model():
    print("\n" + "=" * 65)
    print(" [MODULE 2] TRAINING MACHINE LEARNING CLASSIFIER")
    print("=" * 65)

    student_subdirs = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d)) and not d.startswith(".")
    ]

    if not student_subdirs:
        print("[!] No student datasets found. Run Option [1] first.")
        return

    features, labels = [], []
    name_map = load_registry_map()

    for student_id in student_subdirs:
        student_folder = os.path.join(DATASET_DIR, student_id)
        img_files = [f for f in os.listdir(student_folder) if f.lower().endswith((".jpg", ".png"))]

        for img_name in img_files:
            img = cv2.imread(os.path.join(student_folder, img_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features.append(preprocess_face(img))
                labels.append(student_id)

    if not features:
        print("[!] Error: No valid face images found in dataset folders.")
        return

    print(f"[*] Training KNN classifier on {len(features)} feature vectors...")
    knn = FaceRecognitionKNN(k=KNN_K_NEIGHBORS, threshold=DISTANCE_THRESHOLD)
    knn.fit(features, labels, name_map)
    knn.save(MODEL_FILE)
    print(f"[+] Model trained and saved to: {MODEL_FILE}")


# ----------------------------------------------------------------------
# 9. MODULE 3: REAL-TIME ATTENDANCE LOGGING
# ----------------------------------------------------------------------
def mark_attendance_in_csv(student_id, student_name):
    today_str = datetime.now().strftime("%Y-%m-%d")
    csv_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today_str}.csv")
    file_exists = os.path.exists(csv_file)

    already_marked = False
    if file_exists:
        with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip() == student_id:
                    already_marked = True
                    break

    if not already_marked:
        current_time = datetime.now().strftime("%H:%M:%S")
        with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["StudentID", "StudentName", "Date", "Time", "Status"])
            writer.writerow([student_id, student_name, today_str, current_time, "Present"])
        print(f"\n[ATTENDANCE MARKED] {student_name} ({student_id}) at {current_time}")
        return True, current_time
    return False, "Already Marked"


def run_real_time_attendance():
    print("\n" + "=" * 65)
    print(" [MODULE 3] REAL-TIME ATTENDANCE MARKING (LIVE WEBCAM)")
    print("=" * 65)

    if not os.path.exists(MODEL_FILE):
        print("[!] Model file not found. Run Option [1] and Option [2] first.")
        return

    knn = FaceRecognitionKNN()
    if not knn.load(MODEL_FILE):
        print("[!] Failed to load trained model.")
        return

    camera = get_working_camera()
    if camera is None:
        print("[!] Error: Could not access webcam.")
        return

    print("[*] Camera feed running. Press 'q' to stop.\n")
    feedback_banner = {}

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detect_faces(gray_frame)
        display_frame = frame.copy()

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y : y + h, x : x + w]
            vector = preprocess_face(face_roi)
            student_id, student_name, distance, is_known = knn.predict(vector)

            if is_known:
                box_color = (0, 255, 0)
                label_text = f"{student_name} ({student_id})"
                newly_marked, time_str = mark_attendance_in_csv(student_id, student_name)
                if newly_marked:
                    feedback_banner[student_id] = (f"Marked Present @ {time_str}", time.time())
            else:
                box_color = (0, 0, 255)
                label_text = "Unknown Person"

            cv2.rectangle(display_frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.rectangle(display_frame, (x, y - 35), (x + w, y), box_color, cv2.FILLED)
            cv2.putText(display_frame, label_text, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            if is_known and (student_id in feedback_banner):
                msg, timestamp = feedback_banner[student_id]
                if time.time() - timestamp < 4.0:
                    cv2.putText(display_frame, msg, (x, y + h + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        now_str = datetime.now().strftime("%A, %d %b %Y | %H:%M:%S")
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (45, 45, 45), cv2.FILLED)
        cv2.putText(display_frame, f"Amity Attendance Live | {now_str}", (15, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Real-Time Face Recognition Attendance", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# 10. MODULE 4: VIEW ATTENDANCE LOGS
# ----------------------------------------------------------------------
def view_attendance_records():
    print("\n" + "=" * 65)
    print(" [MODULE 4] TODAY'S ATTENDANCE RECORDS")
    print("=" * 65)

    today_str = datetime.now().strftime("%Y-%m-%d")
    csv_file = os.path.join(ATTENDANCE_DIR, f"attendance_{today_str}.csv")

    if not os.path.exists(csv_file):
        print(f"[*] No attendance records logged yet today ({today_str}).")
        return

    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        print("[*] Attendance file is empty.")
        return

    print(f"\nDate: {today_str} | File: {os.path.basename(csv_file)}")
    print("-" * 65)
    print(f"{'Student ID':<15} {'Student Name':<25} {'Time':<12} {'Status':<10}")
    print("-" * 65)
    for row in rows[1:]:
        if len(row) >= 5:
            s_id, s_name, _, s_time, s_status = row[:5]
            print(f"{s_id:<15} {s_name:<25} {s_time:<12} {s_status:<10}")
    print("-" * 65)
    print(f"Total Present Today: {len(rows) - 1}\n")


# ----------------------------------------------------------------------
# 11. MODULE 5: VIEW STUDENT DIRECTORY
# ----------------------------------------------------------------------
def view_student_directory():
    print("\n" + "=" * 65)
    print(" [MODULE 5] REGISTERED STUDENTS DIRECTORY")
    print("=" * 65)

    if not os.path.exists(STUDENTS_REGISTRY_FILE):
        print("[*] No students enrolled yet.")
        return

    with open(STUDENTS_REGISTRY_FILE, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        print("[*] Registry is empty.")
        return

    print("-" * 75)
    print(f"{'Student ID':<15} {'Student Name':<25} {'Department':<15} {'Samples':<10}")
    print("-" * 75)
    for row in rows[1:]:
        if len(row) >= 3:
            s_id, s_name, s_dept = row[0], row[1], row[2]
            sample_folder = os.path.join(DATASET_DIR, s_id)
            sample_count = len(os.listdir(sample_folder)) if os.path.exists(sample_folder) else 0
            print(f"{s_id:<15} {s_name:<25} {s_dept:<15} {sample_count:<10}")
    print("-" * 75)
    print(f"Total Enrolled Students: {len(rows) - 1}\n")


# ----------------------------------------------------------------------
# 12. MAIN MENU CONTROLLER
# ----------------------------------------------------------------------
def display_banner():
    print("""
======================================================================
  AMITY UNIVERSITY ONLINE - CMA MINOR PROJECT (SEMESTER 3)
  Real-Time Facial Recognition Attendance System Using Machine Learning
======================================================================
  [1] Register New Student (Capture Face Dataset via Webcam)
  [2] Train Machine Learning Model (K-Nearest Neighbors Algorithm)
  [3] Start Real-Time Attendance Marking (Webcam & Automatic CSV Log)
  [4] View Today's Attendance Records
  [5] View Registered Students Directory
  [6] Exit System
======================================================================
""")


def main():
    while True:
        display_banner()
        choice = input("Select an option [1-6]: ").strip()

        if choice == "1":
            register_new_student()
        elif choice == "2":
            train_machine_learning_model()
        elif choice == "3":
            run_real_time_attendance()
        elif choice == "4":
            view_attendance_records()
        elif choice == "5":
            view_student_directory()
        elif choice == "6":
            print("\nExiting Face Recognition Attendance System. Goodbye!")
            sys.exit(0)
        else:
            print("\n[!] Invalid selection. Please enter a number between 1 and 6.")

        input("\nPress Enter to return to main menu...")


if __name__ == "__main__":
    main()
