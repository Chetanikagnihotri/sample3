import cv2
import streamlit as st
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os

face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
            except Exception as e:
                emotion = "Error"

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return img

def main():
    st.title("Real-time Emotion Detection")
    st.write("Use your webcam for real-time emotion detection.")

    webrtc_streamer(key="example", video_transformer_factory=EmotionDetector)

if __name__ == '__main__':
    main()
