import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import av
import cv2
from deepface import DeepFace

st.title("Real-time Emotion Detection")

# Load the pre-trained face detection model (Haar Cascade)
haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

class EmotionDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = face_cascade

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Analyze emotions
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            emotion = result[0]['dominant_emotion']

            # Draw rectangle and emotion label around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Start the webcam stream
webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionDetectionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True,
)
