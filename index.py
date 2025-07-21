import mediapipe as mp
import time
import json
import pyttsx3
import cv2
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import os

# ========== Environment Setup ==========
output_dir = os.environ.get("OUTPUT_DIR", ".")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# ========== Speech Setup ==========
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# ========== Helper Functions ==========
def dist(p1, p2): return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5
def brow_diff(lm): return abs(lm[70].y - lm[63].y) + abs(lm[300].y - lm[293].y)
def mouth_corner_diff(lm): return dist(lm[61], lm[291])
def lip_pucker(lm): return dist(lm[13], lm[14])
def eye_openness(lm): return (dist(lm[159], lm[145]) + dist(lm[386], lm[374])) / 2
def nose_lift(lm): return abs(lm[2].y - lm[98].y)
def nostril_flare(lm): return dist(lm[35], lm[265])
def chin_raise(lm): return abs(lm[152].y - lm[200].y)
def mouth_corner_drop(lm): return abs(lm[61].y - lm[146].y) + abs(lm[291].y - lm[375].y)

# ========== Expression Definitions ==========
expressions = [
    ("Raising eyebrows", "Frontalis", lambda lm: lm[105].y < lm[159].y and lm[334].y < lm[386].y, [105, 159, 334, 386], lambda lm: (lm[159].y - lm[105].y + lm[386].y - lm[334].y)/2, 0.005),
    ("Frowning", "Corrugator supercilii / Procerus", lambda lm: brow_diff(lm) < 0.02, [70, 63, 300, 293], brow_diff, 0.02),
    ("Smiling", "Zygomaticus major / minor", lambda lm: mouth_corner_diff(lm) > 0.03, [61, 291], mouth_corner_diff, 0.03),
    ("Pouting", "Orbicularis oris", lambda lm: lip_pucker(lm) > 0.015, [13, 14], lip_pucker, 0.015),
    ("Squinting", "Orbicularis oculi", lambda lm: eye_openness(lm) < 0.01, [159, 145, 386, 374], eye_openness, 0.01),
    ("Showing disgust", "Levator labii superioris", lambda lm: nose_lift(lm) > 0.01, [2, 98], nose_lift, 0.01),
    ("Flaring nostrils", "Nasalis", lambda lm: nostril_flare(lm) > 0.015, [35, 265], nostril_flare, 0.015),
    ("Puckering chin", "Mentalis", lambda lm: chin_raise(lm) > 0.01, [152, 200], chin_raise, 0.01),
    ("Depressing mouth corners", "Depressor anguli oris", lambda lm: mouth_corner_drop(lm) > 0.01, [61, 146, 291, 375], mouth_corner_drop, 0.01),
]

# ========== Patient Info ==========
patient_name = input("Enter patient name: ").strip()
patient_age = input("Enter patient age: ")
patient_gender = input("Enter patients gender (M/F): ").strip().upper()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ========== MediaPipe Setup ==========
mp_face_mesh = mp.solutions.face_mesh # type: ignore
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils # type: ignore

# ========== Main Loop ==========
cap = cv2.VideoCapture(0)
results_data = []
expression_duration = 10
speak("Starting facial paralysis detection. Please follow the instructions on the screen.") 
i = 0
while i < len(expressions):
    expr_text, muscle, detect_fn, highlight_indices, measure_fn, threshold = expressions[i]
    speak(f"Please perform: {expr_text}")
    expr_start = time.time()
    activated_count = 0
    total_measure = 0.0
    frames = 0
    key = None

    while key != ord('s') and time.time() - expr_start < expression_duration:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            try:
                val = measure_fn(landmarks)
                total_measure += val
                if detect_fn(landmarks):
                    activated_count += 1
            except Exception as e:
                print(f"Error in detection function: {e}")
            frames += 1

            for idx in highlight_indices:
                h, w, _ = frame.shape
                cx, cy = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        remaining = expression_duration - int(time.time() - expr_start)
        cv2.putText(frame, f"{expr_text} ({muscle}) - {remaining}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Press S to skip, R to repeat", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.imshow("Facial Analyzer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            break
        if key == ord('r'):
            expr_start = time.time()
            activated_count = 0
            total_measure = 0.0
            frames = 0

    if key != ord('s'):
        avg_measure = total_measure / max(frames, 1)
        status = "Paralyzed" if avg_measure < threshold else "Active"
        percent = round((activated_count / max(1, frames)) * 100, 2)
        results_data.append({
            "expression": expr_text,
            "muscle": muscle,
            "activation_percent": percent,
            "status": status,
            "measured_value": round(avg_measure, 5),
            "expected_threshold": threshold
        })
    i += 1

cap.release()
cv2.destroyAllWindows()

json_path = os.path.join(output_dir, f"{patient_name}_paralysis_report.json")
pdf_path = os.path.join(output_dir, f"{patient_name}_paralysis_report.pdf")

# ========== Save JSON ==========
with open(json_path, "w") as f:
    json.dump(results_data, f, indent=4)

# ========== Save PDF ==========
pdf = canvas.Canvas(pdf_path, pagesize=letter)
width, height = letter
pdf.setFont("Helvetica-Bold", 16)
pdf.drawString(100, height - 50, "Facial Paralysis Detection Report")
pdf.setFont("Helvetica", 12)
pdf.drawString(50, height - 80, f"Patient: {patient_name}")
pdf.drawString(50, height - 100, f"Age: {patient_age}")
pdf.drawString(300, height - 100, f"Gender: {patient_gender}")
y = height - 130
pdf.drawString(300, height - 80, f"Date: {timestamp}")
y = height - 110

for entry in results_data:
    pdf.drawString(50, y, f"{entry['expression']} ({entry['muscle']}): Activation = {entry['activation_percent']}% - {entry['status']}")
    y -= 20

if y < 300:
    pdf.showPage()
    y = height - 50
pdf.save()

print("\n✅ PDF and JSON reports generated successfully.")