import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2

st.title("Real-time Face Detection")

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = face_cascade

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Start the webcam stream
webrtc_streamer(
    key="face-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=FaceDetectionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
