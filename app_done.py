import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import threading
import time
from playsound import playsound


# =========================
# SESSION STATE
# =========================
if "alarm_on" not in st.session_state:
    st.session_state.alarm_on = False

if "last_alarm_time" not in st.session_state:
    st.session_state.last_alarm_time = 0


# =========================
# ALARM LOCK (ANTI OVERLAP)
# =========================
alarm_lock = threading.Lock()


# =========================
# FUNGSI ALARM
# =========================
def play_alarm():
    if alarm_lock.locked():
        return

    with alarm_lock:
        playsound("kasih-paham-bos-dj.wav")


# =========================
# LOAD MODEL
# =========================
model = YOLO("best (1).pt")


# =========================
# UI TITLE
# =========================
st.title("🚗 Driver Drowsiness Detection")


# =========================
# SIDEBAR
# =========================
confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.25, 0.05
)


# =========================
# FILE UPLOAD MODE
# =========================
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    results = model.predict(image_np, conf=confidence)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detection Result")


# =========================
# CLASS NAMES
# =========================
class_names = {
0: "Kepala Menunduk",
1: "Menguap",
2: "Tidur",
3: "Kepala Mendongak",
4: "Miring Kanan Mengantuk",
5: "Miring Kanan Sadar",
6: "Sadar",
7: "Miring Kiri Mengantuk",
8: "Miring Kiri Sadar"
}


# =========================
# WEBCAM MODE
# =========================
st.subheader("Webcam Detection")

start = st.button("Start Camera")

frame_window = st.image([])

if start:

    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            st.write("Kamera tidak terdeteksi")
            break

        results = model(frame, conf=confidence)

        detected_yawn = False

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy
            classes = result.boxes.cls

            for box, cls in zip(boxes, classes):

                class_name = class_names[int(cls)]

                if class_name == "Menguap":
                    detected_yawn = True

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

                cv2.putText(
                    frame,
                    class_name,
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,0,255),
                    2
                )

        # =========================
        # LOGIC ALARM
        # =========================
        current_time = time.time()

        if detected_yawn:
            if (current_time - st.session_state.last_alarm_time) > 3:
                st.session_state.last_alarm_time = current_time
                threading.Thread(target=play_alarm).start()

        frame_window.image(frame, channels="BGR")

    cap.release()