
import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import winsound
import time

# ==========================================
# CONFIGURATION
# ==========================================
BG = "#071321"
PANEL = "#101F30"
CARD = "#14283D"
BORDER = "#243B52"
TEXT = "#F1F5F9"
MUTED = "#9CB4CC"
GREEN = "#00E676"
RED = "#FF5252"
YELLOW = "#FFD740"
ORANGE = "#FFA726"

EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.60
CLOSED_EYE_FRAMES = 15
YAWN_FRAMES = 12
HEAD_TURN_THRESHOLD = 0.15
BEEP_INTERVAL = 2.0

closed_eye_count = 0
yawn_count = 0
last_beep = 0.0

sound_lock = threading.Lock()
cap = None
camera_running = False
frame_job = None
photo_reference = None

# ==========================================
# MEDIAPIPE FACE MESH
# ==========================================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

NOSE_TIP = 1
FACE_LEFT = 234
FACE_RIGHT = 454


# ==========================================
# WARNING SOUND
# ==========================================
def play_warning_sound():
    if not sound_lock.acquire(blocking=False):
        return

    def sound_worker():
        try:
            winsound.Beep(1200, 700)
        except (RuntimeError, OSError):
            pass
        finally:
            sound_lock.release()

    threading.Thread(
        target=sound_worker,
        daemon=True
    ).start()


# ==========================================
# EYE ASPECT RATIO
# ==========================================
def calculate_ear(landmarks, indexes, width, height):
    points = []

    for index in indexes:
        point = landmarks[index]
        points.append(
            np.array([point.x * width, point.y * height])
        )

    vertical1 = np.linalg.norm(points[1] - points[5])
    vertical2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


# ==========================================
# MOUTH ASPECT RATIO
# ==========================================
def calculate_mar(landmarks, width, height):
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    vertical = np.linalg.norm(
        np.array([top.x * width, top.y * height]) -
        np.array([bottom.x * width, bottom.y * height])
    )

    horizontal = np.linalg.norm(
        np.array([left.x * width, left.y * height]) -
        np.array([right.x * width, right.y * height])
    )

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


# ==========================================
# HEAD DIRECTION
# ==========================================
def detect_head_turn(landmarks):
    nose_x = landmarks[NOSE_TIP].x
    left_x = landmarks[FACE_LEFT].x
    right_x = landmarks[FACE_RIGHT].x

    face_width = abs(right_x - left_x)

    if face_width == 0:
        return False

    face_center = (left_x + right_x) / 2.0
    deviation = abs(nose_x - face_center) / face_width

    return deviation > HEAD_TURN_THRESHOLD


# ==========================================
# DRAW FACE, EYES AND MOUTH
# ==========================================
def draw_face_overlays(frame, landmarks, width, height):
    # Face rectangle
    xs = [int(p.x * width) for p in landmarks]
    ys = [int(p.y * height) for p in landmarks]

    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(width - 1, max(xs))
    y2 = min(height - 1, max(ys))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 100),
        2
    )

    # Connected eye outlines
    eye_connections = [
        mp_face_mesh.FACEMESH_LEFT_EYE,
        mp_face_mesh.FACEMESH_RIGHT_EYE
    ]

    for connections in eye_connections:
        for start, end in connections:
            p1 = landmarks[start]
            p2 = landmarks[end]

            pt1 = (int(p1.x * width), int(p1.y * height))
            pt2 = (int(p2.x * width), int(p2.y * height))

            cv2.line(
                frame, pt1, pt2,
                (0, 255, 100), 2, cv2.LINE_AA
            )

    # Connected mouth outline
    for start, end in mp_face_mesh.FACEMESH_LIPS:
        p1 = landmarks[start]
        p2 = landmarks[end]

        pt1 = (int(p1.x * width), int(p1.y * height))
        pt2 = (int(p2.x * width), int(p2.y * height))

        cv2.line(
            frame, pt1, pt2,
            (0, 165, 255), 2, cv2.LINE_AA
        )


# ==========================================
# UI HELPER FUNCTIONS
# ==========================================
def make_label(parent, text, size=11, color=TEXT, bold=False):
    return tk.Label(
        parent,
        text=text,
        font=("Segoe UI", size, "bold" if bold else "normal"),
        bg=parent.cget("bg"),
        fg=color,
        anchor="w"
    )


def create_metric(parent, title, initial, color):
    card = tk.Frame(
        parent,
        bg=CARD,
        highlightbackground=BORDER,
        highlightthickness=1
    )

    tk.Label(
        card,
        text=title,
        font=("Segoe UI", 10),
        bg=CARD,
        fg=MUTED
    ).pack(anchor="w", padx=12, pady=(9, 2))

    value = tk.Label(
        card,
        text=initial,
        font=("Segoe UI", 16, "bold"),
        bg=CARD,
        fg=color
    )
    value.pack(anchor="w", padx=12, pady=(0, 9))

    return card, value


def create_button(parent, text, color, command):
    return tk.Button(
        parent,
        text=text,
        command=command,
        font=("Segoe UI", 11, "bold"),
        bg=color,
        fg="white",
        activebackground=color,
        activeforeground="white",
        relief="flat",
        cursor="hand2",
        padx=8,
        pady=11
    )


# ==========================================
# MAIN DASHBOARD WINDOW
# ==========================================
root = tk.Tk()
root.title("Driver Monitoring System")
root.geometry("1350x850")
root.minsize(1050, 680)
root.configure(bg=BG)

# Header
header = tk.Frame(root, bg=PANEL, height=88)
header.pack(fill="x")
header.pack_propagate(False)

tk.Label(
    header,
    text="◉",
    font=("Segoe UI", 34, "bold"),
    bg=PANEL,
    fg=GREEN
).pack(side="left", padx=(22, 12))

title_box = tk.Frame(header, bg=PANEL)
title_box.pack(side="left", pady=14)

tk.Label(
    title_box,
    text="Driver Monitoring System",
    font=("Segoe UI", 21, "bold"),
    bg=PANEL,
    fg=TEXT
).pack(anchor="w")

tk.Label(
    title_box,
    text="Drowsiness  |  Yawning  |  Head Movement Detection",
    font=("Segoe UI", 10),
    bg=PANEL,
    fg=MUTED
).pack(anchor="w")

clock_label = tk.Label(
    header,
    text="",
    font=("Segoe UI", 11, "bold"),
    bg=PANEL,
    fg=MUTED
)
clock_label.pack(side="right", padx=22)


def update_clock():
    if root.winfo_exists():
        clock_label.config(
            text=time.strftime("%d %b %Y   %I:%M:%S %p")
        )
        root.after(1000, update_clock)


# Main layout
main = tk.Frame(root, bg=BG)
main.pack(fill="both", expand=True, padx=14, pady=12)

sidebar = tk.Frame(main, bg=BG, width=235)
sidebar.pack(side="left", fill="y", padx=(0, 12))
sidebar.pack_propagate(False)

content = tk.Frame(main, bg=BG)
content.pack(side="left", fill="both", expand=True)

right_panel = tk.Frame(main, bg=BG, width=210)
right_panel.pack(side="right", fill="y", padx=(12, 0))
right_panel.pack_propagate(False)

# Left sidebar
make_label(
    sidebar, "CAMERA CONTROLS", 11, MUTED, True
).pack(anchor="w", pady=(2, 9))

sound_enabled = tk.BooleanVar(value=True)

start_button = create_button(
    sidebar, "▶   Start Camera", "#008F55",
    lambda: start_camera()
)
start_button.pack(fill="x", pady=(0, 7))

stop_button = create_button(
    sidebar, "■   Stop Camera", "#23415F",
    lambda: stop_camera()
)
stop_button.pack(fill="x", pady=(0, 7))

tk.Checkbutton(
    sidebar,
    text="Enable warning sound",
    variable=sound_enabled,
    font=("Segoe UI", 10),
    bg=BG,
    fg=TEXT,
    activebackground=BG,
    activeforeground=TEXT,
    selectcolor=CARD
).pack(anchor="w", pady=(2, 17))

make_label(
    sidebar, "LIVE METRICS", 12, TEXT, True
).pack(anchor="w", pady=(0, 9))

ear_card, ear_value = create_metric(
    sidebar, "Eye Aspect Ratio (EAR)", "--", YELLOW
)
ear_card.pack(fill="x", pady=(0, 7))

mar_card, mar_value = create_metric(
    sidebar, "Mouth Aspect Ratio (MAR)", "--", YELLOW
)
mar_card.pack(fill="x", pady=(0, 7))

head_card, head_value = create_metric(
    sidebar, "Head Direction", "Unknown", MUTED
)
head_card.pack(fill="x", pady=(0, 7))

face_card, face_value = create_metric(
    sidebar, "Face Status", "Not detected", MUTED
)
face_card.pack(fill="x", pady=(0, 7))

# Camera panel
camera_header = tk.Frame(content, bg=BG)
camera_header.pack(fill="x", pady=(0, 8))

make_label(
    camera_header, "LIVE CAMERA FEED", 12, TEXT, True
).pack(side="left")

camera_status = make_label(
    camera_header, "● CAMERA OFF", 10, RED, True
)
camera_status.pack(side="right")

video_panel = tk.Frame(
    content,
    bg="#020811",
    highlightbackground=BORDER,
    highlightthickness=1
)
video_panel.pack(fill="both", expand=True)

video_label = tk.Label(
    video_panel,
    text="Camera is stopped\n\nClick Start Camera to begin monitoring",
    font=("Segoe UI", 15),
    bg="#020811",
    fg=MUTED,
    justify="center"
)
video_label.pack(fill="both", expand=True)

# Warning status panel
status_panel = tk.Frame(
    content,
    bg=PANEL,
    highlightbackground=BORDER,
    highlightthickness=1
)
status_panel.pack(fill="x", pady=(10, 0))

status_title = tk.Label(
    status_panel,
    text="●  SYSTEM READY",
    font=("Segoe UI", 12, "bold"),
    bg=PANEL,
    fg=GREEN
)
status_title.pack(anchor="w", padx=14, pady=(10, 2))

status_detail = tk.Label(
    status_panel,
    text="Start the camera to begin driver monitoring.",
    font=("Segoe UI", 10),
    bg=PANEL,
    fg=MUTED,
    anchor="w"
)
status_detail.pack(anchor="w", padx=14, pady=(0, 10))

# Right-side detection status cards
make_label(
    right_panel, "DETECTION STATUS", 11, MUTED, True
).pack(anchor="w", pady=(2, 9))

system_card, system_value = create_metric(
    right_panel, "Overall Status", "Ready", GREEN
)
system_card.pack(fill="x", pady=(0, 8))

eyes_card, eyes_value = create_metric(
    right_panel, "Eyes", "Unknown", MUTED
)
eyes_card.pack(fill="x", pady=(0, 8))

mouth_card, mouth_value = create_metric(
    right_panel, "Mouth", "Normal", GREEN
)
mouth_card.pack(fill="x", pady=(0, 8))

direction_card, direction_value = create_metric(
    right_panel, "Head Direction", "Unknown", MUTED
)
direction_card.pack(fill="x", pady=(0, 8))

detection_card, detection_value = create_metric(
    right_panel, "Face Detection", "Not detected", MUTED
)
detection_card.pack(fill="x", pady=(0, 8))

sound_card, sound_value = create_metric(
    right_panel, "Warning Sound", "ON", GREEN
)
sound_card.pack(fill="x", pady=(0, 8))

# Footer
footer = tk.Frame(root, bg=PANEL)
footer.pack(fill="x", side="bottom")

tk.Label(
    footer,
    text="Real-time monitoring using MediaPipe  |  Press Q to quit",
    font=("Segoe UI", 9),
    bg=PANEL,
    fg=MUTED
).pack(side="left", padx=14, pady=9)


# ==========================================
# START CAMERA
# ==========================================
def start_camera():
    global cap, camera_running
    global closed_eye_count, yawn_count, last_beep

    if camera_running:
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap.release()
        cap = None
        messagebox.showerror(
            "Camera Error",
            "Unable to open webcam.\nCheck camera permissions."
        )
        return

    closed_eye_count = 0
    yawn_count = 0
    last_beep = 0.0
    camera_running = True

    camera_status.config(text="● CAMERA ON", fg=GREEN)
    status_title.config(text="● MONITORING", fg=GREEN)
    status_detail.config(
        text="Analyzing eyes, mouth and head direction."
    )
    system_value.config(text="Monitoring", fg=GREEN)
    start_button.config(state="disabled")

    update_frame()


# ==========================================
# STOP CAMERA
# ==========================================
def stop_camera():
    global cap, camera_running, frame_job
    global closed_eye_count, yawn_count

    camera_running = False

    if frame_job is not None:
        try:
            root.after_cancel(frame_job)
        except tk.TclError:
            pass
        frame_job = None

    if cap is not None:
        cap.release()
        cap = None

    closed_eye_count = 0
    yawn_count = 0

    camera_status.config(text="● CAMERA OFF", fg=RED)
    status_title.config(text="● CAMERA STOPPED", fg=MUTED)
    status_detail.config(
        text="Click Start Camera to resume monitoring."
    )
    system_value.config(text="Stopped", fg=MUTED)

    video_label.config(
        image="",
        text="Camera is stopped\n\nClick Start Camera to begin monitoring"
    )
    video_label.image = None

    reset_values = [
        (ear_value, "--", YELLOW),
        (mar_value, "--", YELLOW),
        (head_value, "Unknown", MUTED),
        (face_value, "Not detected", MUTED),
        (eyes_value, "Unknown", MUTED),
        (mouth_value, "Unknown", MUTED),
        (direction_value, "Unknown", MUTED),
        (detection_value, "Not detected", MUTED)
    ]

    for widget, text, color in reset_values:
        widget.config(text=text, fg=color)

    start_button.config(state="normal")


# ==========================================
# PROCESS WEBCAM FRAMES
# ==========================================
def update_frame():
    global frame_job, photo_reference
    global closed_eye_count, yawn_count, last_beep

    if not camera_running or cap is None:
        return

    success, frame = cap.read()

    if not success:
        status_title.config(text="● CAMERA READ ERROR", fg=RED)
        frame_job = root.after(100, update_frame)
        return

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    alerts = []
    ear = 0.0
    mar = 0.0
    head_turned = False
    face_detected = bool(results.multi_face_landmarks)

    if face_detected:
        landmarks = results.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(
            landmarks, LEFT_EYE, width, height
        )
        right_ear = calculate_ear(
            landmarks, RIGHT_EYE, width, height
        )
        ear = (left_ear + right_ear) / 2.0

        mar = calculate_mar(landmarks, width, height)
        head_turned = detect_head_turn(landmarks)

        # Eye closure warning
        if ear < EAR_THRESHOLD:
            closed_eye_count += 1
        else:
            closed_eye_count = 0

        if closed_eye_count >= CLOSED_EYE_FRAMES:
            alerts.append("DROWSINESS WARNING")

        # Yawn warning
        if mar > MAR_THRESHOLD:
            yawn_count += 1
        else:
            yawn_count = 0

        if yawn_count >= YAWN_FRAMES:
            alerts.append("POSSIBLE YAWN")

        # Head-turn warning
        if head_turned:
            alerts.append("HEAD TURN WARNING")

        draw_face_overlays(
            frame, landmarks, width, height
        )

    else:
        closed_eye_count = 0
        yawn_count = 0
        alerts.append("FACE NOT DETECTED")

    # Update EAR and MAR cards
    ear_value.config(
        text=f"{ear:.2f}" if face_detected else "--",
        fg=RED if face_detected and ear < EAR_THRESHOLD else GREEN
    )

    mar_value.config(
        text=f"{mar:.2f}" if face_detected else "--",
        fg=ORANGE if face_detected and mar > MAR_THRESHOLD else GREEN
    )

    # Update head direction
    head_text = (
        ("Turned" if head_turned else "Forward")
        if face_detected else "Unknown"
    )
    head_color = (
        RED if head_turned
        else GREEN if face_detected
        else MUTED
    )

    head_value.config(text=head_text, fg=head_color)
    direction_value.config(text=head_text, fg=head_color)

    # Update face detection
    face_text = "Detected" if face_detected else "Not detected"
    face_color = GREEN if face_detected else RED

    face_value.config(text=face_text, fg=face_color)
    detection_value.config(text=face_text, fg=face_color)

    # Update eye status
    eyes_text = (
        "Closed"
        if face_detected and ear < EAR_THRESHOLD
        else "Open"
    )

    eyes_value.config(
        text=eyes_text if face_detected else "Unknown",
        fg=(
            RED if eyes_text == "Closed"
            else GREEN if face_detected
            else MUTED
        )
    )

    # Update mouth status
    mouth_text = (
        "Yawning"
        if face_detected and mar > MAR_THRESHOLD
        else "Normal"
    )

    mouth_value.config(
        text=mouth_text if face_detected else "Unknown",
        fg=RED if mouth_text == "Yawning" else GREEN
    )

    # Draw warning banner and play sound
    if alerts:
        overlay = " | ".join(alerts)

        cv2.rectangle(
            frame, (0, 0), (width, 48),
            (0, 0, 150), -1
        )

        cv2.putText(
            frame,
            overlay,
            (12, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        status_title.config(
            text="● WARNING ACTIVE", fg=RED
        )
        status_detail.config(text=overlay)
        system_value.config(text="Warning", fg=RED)

        now = time.monotonic()

        if (
            sound_enabled.get()
            and now - last_beep >= BEEP_INTERVAL
        ):
            play_warning_sound()
            last_beep = now

    else:
        status_title.config(
            text="● MONITORING - NORMAL", fg=GREEN
        )
        status_detail.config(
            text="No current warning detected."
        )
        system_value.config(text="Monitoring", fg=GREEN)
        last_beep = 0.0

    sound_value.config(
        text="ON" if sound_enabled.get() else "OFF",
        fg=GREEN if sound_enabled.get() else MUTED
    )

    # Resize webcam feed to fit the dashboard
    panel_width = max(video_panel.winfo_width(), 320)
    panel_height = max(video_panel.winfo_height(), 240)

    frame = cv2.resize(
        frame,
        (panel_width, panel_height),
        interpolation=cv2.INTER_AREA
    )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    photo_reference = ImageTk.PhotoImage(
        Image.fromarray(frame)
    )

    video_label.config(
        image=photo_reference,
        text=""
    )
    video_label.image = photo_reference

    frame_job = root.after(15, update_frame)


# ==========================================
# EXIT HANDLING
# ==========================================
def on_key(event):
    if event.char.lower() == "q":
        on_close()


def on_close():
    stop_camera()

    try:
        face_mesh.close()
    except Exception:
        pass

    root.destroy()


root.bind("<Key>", on_key)
root.protocol("WM_DELETE_WINDOW", on_close)

update_clock()
root.mainloop()
