#!/usr/bin/env python3

"""
Video Processing App (Google Colab Compatible)

Features:
- Upload video in Colab
- Convert to grayscale
- Rotate 180 degrees
- Mirror horizontal / vertical
- Simple object detection (no deep learning)
- Splits (50:50 and 70:30 vertical & horizontal)
- Create grid tiles
- Save processed video

AUTO-INSTALLS OpenCV + moviepy if not available.
"""

import os
import sys

# -------------------------
# AUTO-INSTALL DEPENDENCIES
# -------------------------
try:
    import cv2
except:
    print("Installing OpenCV...")
    os.system("pip install opencv-python-headless")
    import cv2

try:
    from google.colab import files
    IN_COLAB = True
except:
    IN_COLAB = False

import numpy as np


# -------------------------
# FILE UPLOAD (Colab Only)
# -------------------------
def upload_video():
    print("UPLOAD VIDEO FILE:")
    uploaded = files.upload()
    filename = next(iter(uploaded))
    print("Uploaded:", filename)
    return filename


# -------------------------
# SIMPLE VIDEO TRANSFORMS
# -------------------------
def to_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def rotate_180(frame):
    return cv2.rotate(frame, cv2.ROTATE_180)

def mirror_h(frame):
    return cv2.flip(frame, 1)

def mirror_v(frame):
    return cv2.flip(frame, 0)

def detect_objects(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = frame.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h > 200:  # ignore tiny objects
            cv2.rectangle(out, (x, y), (x+w, y+h), (0,255,0), 2)

    return out


# -------------------------
# SPLITTING FUNCTIONS
# -------------------------
def split_vertical(frame, ratio):
    h, w = frame.shape[:2]
    cut = int(w * ratio)
    return frame[:, :cut], frame[:, cut:]

def split_horizontal(frame, ratio):
    h, w = frame.shape[:2]
    cut = int(h * ratio)
    return frame[:cut, :], frame[cut:, :]


# -------------------------
# GRID SPLITTING
# -------------------------
def save_grid(frame, rows, cols, outdir):
    h, w = frame.shape[:2]
    gh = h // rows
    gw = w // cols
    os.makedirs(outdir, exist_ok=True)

    count = 1
    for r in range(rows):
        for c in range(cols):
            part = frame[r*gh:(r+1)*gh, c*gw:(c+1)*gw]
            cv2.imwrite(f"{outdir}/grid_{count}.jpg", part)
            count += 1


# -------------------------
# PROCESS VIDEO PIPELINE
# -------------------------
def process_video(videopath):
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print("Error opening video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter("processed_video.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    print("\nChoose processing options:")
    print("1: Grayscale")
    print("2: Rotate 180°")
    print("3: Mirror Horizontal")
    print("4: Mirror Vertical")
    print("5: Object Detection (No DL)")
    print("6: Create Grid (4x4)")
    print("7: Split 50:50 (V/H)")
    print("8: Split 70:30 (V/H)")
    print("0: Continue without options")

    choice = input("Enter choices separated by comma (e.g., 1,3,5): ").split(',')

    do_gray = '1' in choice
    do_rot = '2' in choice
    do_mh  = '3' in choice
    do_mv  = '4' in choice
    do_obj = '5' in choice
    do_grid = '6' in choice
    do_v50 = '7' in choice
    do_v70 = '8' in choice

    frame_count = 0

    print("\nProcessing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if do_gray: frame = to_gray(frame)
        if do_rot:  frame = rotate_180(frame)
        if do_mh:   frame = mirror_h(frame)
        if do_mv:   frame = mirror_v(frame)
        if do_obj:  frame = detect_objects(frame)

        # Save grid only for first frame
        if frame_count == 0 and do_grid:
            save_grid(frame, 4, 4, "grid_output")

        # Save splits only for first frame
        if frame_count == 0:
            if do_v50:
                a,b = split_vertical(frame, 0.5)
                cv2.imwrite("split_v_50_left.jpg", a)
                cv2.imwrite("split_v_50_right.jpg", b)

                a,b = split_horizontal(frame, 0.5)
                cv2.imwrite("split_h_50_top.jpg", a)
                cv2.imwrite("split_h_50_bottom.jpg", b)

            if do_v70:
                a,b = split_vertical(frame, 0.7)
                cv2.imwrite("split_v_70_left.jpg", a)
                cv2.imwrite("split_v_70_right.jpg", b)

                a,b = split_horizontal(frame, 0.7)
                cv2.imwrite("split_h_70_top.jpg", a)
                cv2.imwrite("split_h_70_bottom.jpg", b)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print("\nProcessing complete!")
    print("Saved as: processed_video.mp4")


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("=== VIDEO PROCESSOR APP ===")

    if IN_COLAB:
        videopath = upload_video()
    else:
        videopath = input("Enter video file path: ").strip()

    process_video(videopath)

    if IN_COLAB:
        print("\nDownload processed video:")
        from google.colab import files
        files.download("processed_video.mp4")


   
