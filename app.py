import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Color Wand Tracker v4", layout="wide")

# ---------------- SESSION STATE ----------------
if "snapshot" not in st.session_state:
    st.session_state.snapshot = None

if "recording" not in st.session_state:
    st.session_state.recording = False


# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:left;'>🎨 Color Wand Tracker System (v4)</h1>

<p style='text-align:left; opacity:0.75; font-size:15px;'>
Real-time computer vision tracking + recording dashboard
</p>

<hr>
""", unsafe_allow_html=True)


# ---------------- VIDEO PROCESSOR ----------------
class ColorTracker(VideoProcessorBase):
    def __init__(self):
        self.trail = []
        self.color = "Red"
        self.prev_time = time.time()
        self.fps = 0
        self.activity = 0
        self.detected = False
        self.last_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))

        # ---------------- FPS ----------------
        current_time = time.time()
        self.fps = 1 / max(current_time - self.prev_time, 0.001)
        self.prev_time = current_time

        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        color = self.color

        # ---------------- COLOR DETECTION ----------------
        if color == "Red":
            lower1 = np.array([0, 120, 70])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 120, 70])
            upper2 = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
            draw_color = (0, 0, 255)

        elif color == "Green":
            lower = np.array([40, 50, 50])
            upper = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            draw_color = (0, 255, 0)

        else:
            lower = np.array([90, 50, 50])
            upper = np.array([140, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            draw_color = (255, 0, 0)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.detected = False

        if contours:
            biggest = max(contours, key=cv2.contourArea)

            if cv2.contourArea(biggest) > 120:
                self.detected = True

                ((x, y), radius) = cv2.minEnclosingCircle(biggest)
                center = (int(x), int(y))

                self.trail.append(center)

                if len(self.trail) > 60:
                    self.trail.pop(0)

                self.activity = min(self.activity + 2, 100)

                for i in range(1, len(self.trail)):
                    cv2.line(img, self.trail[i - 1], self.trail[i], draw_color, 2)

                cv2.circle(img, center, 8, draw_color, -1)

        if not self.detected:
            self.activity = max(self.activity - 1, 0)

        # ---------------- HUD ----------------
        cv2.putText(img, f"FPS: {int(self.fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(img, f"Activity: {self.activity}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        self.last_frame = img.copy()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ---------------- LAYOUT ----------------
left, right = st.columns([3, 1])


# ---------------- LEFT SIDE ----------------
with left:
    st.markdown("### 📹 Live Camera Feed")

    ctx = webrtc_streamer(
        key="tracker_v4",
        video_processor_factory=ColorTracker,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True
    )

    # ---------------- TUTORIAL ----------------
    st.markdown("### 📘 Tutorial")
    st.write("""
    1. Select a tracking color (Red, Green, Blue)  
    2. Hold or move an object in front of the camera  
    3. System detects and draws motion trail  
    4. FPS + activity are shown in real time  
    5. Use controls to reset or record  
    """)

    # ---------------- ABOUT ----------------
    st.markdown("### ℹ️ About This System")
    st.write("""
This system demonstrates real-time color-based object tracking using computer vision techniques.
It detects selected colors and visualizes motion paths using OpenCV and WebRTC streaming.
    """)

    # ---------------- LIMITATIONS ----------------
    st.markdown("### ⚠️ Limitations & Performance Notes")
    st.warning("""
• Lighting conditions heavily affect detection  
• Camera quality affects accuracy  
• Background colors may interfere  
• Fast motion reduces tracking stability  
    """)

    st.info("""
✔ Best results:
- Use plain background  
- Use strong contrasting colors  
- Ensure good lighting  
- Avoid shadows or flickering light  
    """)


# ---------------- RIGHT SIDE ----------------
with right:
    st.markdown("### 🎛 Controls")

    color = st.radio("Select Color", ["Red", "Green", "Blue"])

    st.markdown("### 📊 System Status")

    cam_status = "❌ Not Connected"
    track_status = "❌ Not Tracking"

    if ctx.video_processor:
        ctx.video_processor.color = color
        cam_status = "✔ Camera Connected"

        if ctx.video_processor.detected:
            track_status = "✔ Tracking Object"
        else:
            track_status = "⚠ Waiting for Object"

    st.write("Camera:", cam_status)
    st.write("Tracking:", track_status)

    if "✔" in cam_status:
        st.success(cam_status)
    else:
        st.error(cam_status)

    if "✔ Tracking" in track_status:
        st.success(track_status)
    else:
        st.warning(track_status)


    # ---------------- SNAPSHOT ----------------
    st.markdown("### 📸 Snapshot")

    if st.button("Capture Frame"):
        if ctx.video_processor and ctx.video_processor.last_frame is not None:
            st.session_state.snapshot = ctx.video_processor.last_frame
            st.success("Snapshot Captured!")

    if st.session_state.snapshot is not None:
        st.image(st.session_state.snapshot, caption="Last Frame")


    # ---------------- VIDEO RECORDING ----------------
    st.markdown("### 🎥 Video Recording")

    if st.button("Start/Stop Recording"):
        st.session_state.recording = not st.session_state.recording

    if st.session_state.recording:
        st.warning("🔴 Recording Active (simulation mode)")

        # NOTE:
        # Real video saving needs extra pipeline (we can add next upgrade)


    # ---------------- RESET ----------------
    st.markdown("### 🧹 Reset")

    if st.button("Clear Trail"):
        if ctx.video_processor:
            ctx.video_processor.trail = []
            ctx.video_processor.activity = 0