import streamlit as st

st.title("Color Wand Tracker")

color = st.radio(
    "Select Tracking Color",
    ["Red", "Green", "Blue"]
)

st.write(f"Currently tracking: {color}")