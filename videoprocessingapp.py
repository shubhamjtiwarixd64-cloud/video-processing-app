import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.title("🎥 Advanced Video Processing App (Streamlit Compatible)")

uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save uploaded file temporarily
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())
    temp_input.close()

    st.video(temp_input.name)
    st.write("Processing video... please wait.")

    # Output file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    cap = cv2.VideoCapture(temp_input.name)

    # Get video properties
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_output.name, fourcc, FPS, (W, H))

    # Background subtractor
    back_sub = cv2.createBackgroundSubtractorMOG2()

    progress = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ========== Apply Effects ==========

        # Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Mirror
        mirror = cv2.flip(frame, 1)

        # Edge detect
        edges = cv2.Canny(gray, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Motion detect
        motion_mask = back_sub.apply(frame)
        motion_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)

        # Rotate
        rotate_180 = cv2.rotate(frame, cv2.ROTATE_180)

        # Combine 2x3 grid for output
        top_row = cv2.hconcat([frame, gray_bgr, mirror])
        bottom_row = cv2.hconcat([edges_bgr, motion_bgr, rotate_180])
        combined = cv2.vconcat([top_row, bottom_row])

        combined_resized = cv2.resize(combined, (W, H))

        out.write(combined_resized)

        count += 1
        progress.progress(min(count / total_frames, 1.0))

    cap.release()
    out.release()

    st.success("Video processing completed!")

    # Download Button
    with open(temp_output.name, "rb") as f:
        st.download_button(
            label="⬇️ Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4",
        )
