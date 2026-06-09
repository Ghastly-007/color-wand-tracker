import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Webcam Test")

webrtc_streamer(key="camera")