import threading
import time
import av
import cv2
import numpy as np
from PIL import Image
from playsound import playsound
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO

# =========================
# SESSION STATE
# =========================
if "last_alarm_time" not in st.session_state:
  st.session_state.last_alarm_time = 0

# =========================
# ALARM LOCK (ANTI OVERLAP)
# =========================
alarm_lock = threading.Lock()


def play_alarm():
  if alarm_lock.locked():
    return
  with alarm_lock:
    try:
      playsound("kasih-paham-bos-dj.wav")
    except Exception:
      pass  # Menghindari crash jika audio device server cloud tidak tersedia


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
  return YOLO("best (1).pt")


model = load_model()

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
    8: "Miring Kiri Sadar",
}

# =========================
# UI TITLE & SIDEBAR
# =========================
st.title("🚗 Driver Drowsiness Detection")

confidence = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.25, 0.05
)

# =========================
# FILE UPLOAD MODE
# =========================
uploaded_file = st.file_uploader(
    "Upload Image", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  image_np = np.array(image)
  results = model.predict(image_np, conf=confidence)
  annotated_frame = results[0].plot()
  st.image(annotated_frame, caption="Detection Result")

st.divider()
st.subheader("📷 Live Webcam Detection")


# =========================
# WEBCAM CALLBACK FUNCTION
# =========================
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
  img = frame.to_ndarray(format="bgr24")

  results = model(img, conf=confidence)
  detected_yawn = False

  for result in results:
    if result.boxes is None:
      continue

    boxes = result.boxes.xyxy
    classes = result.boxes.cls

    for box, cls in zip(boxes, classes):
      class_name = class_names.get(int(cls), "Unknown")

      if class_name == "Menguap":
        detected_yawn = True

      x1, y1, x2, y2 = map(int, box)
      cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
      cv2.putText(
          img,
          class_name,
          (x1, y1 - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.8,
          (0, 0, 255),
          2,
      )

  current_time = time.time()
  if detected_yawn:
    if (current_time - st.session_state.last_alarm_time) > 3:
      st.session_state.last_alarm_time = current_time
      threading.Thread(target=play_alarm).start()

  return av.VideoFrame.from_ndarray(img, format="bgr24")


# =========================
# STREAMLIT WEBRTC COMPONENT
# =========================
webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)