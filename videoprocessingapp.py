# ===============================================
#      ADVANCED VIDEO PROCESSING APP
#  (NO NEURAL NETWORKS — PURE CV + Filters)
# ===============================================

import cv2
import numpy as np

# ----------------------------
# 1. Load video
# ----------------------------
video_path = "input.mp4"   # ← Change your video here
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Video properties
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)

# Output writer
out = cv2.VideoWriter(
    "advanced_processed_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (W, H)
)

print("Processing... Please wait.")

# Background subtractor for motion detection
back_sub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ----------------------------
    # 2. Apply all filters
    # ----------------------------

    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Rotate 180°
    rotate_180 = cv2.rotate(frame, cv2.ROTATE_180)

    # Mirror
    mirror_h = cv2.flip(frame, 1)

    # Edge Detection
    edges = cv2.Canny(gray, 100, 200)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Motion Detection
    fg_mask = back_sub.apply(frame)
    motion = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

    # Object Detection (Contours)
    thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_frame = frame.copy()
    cv2.drawContours(contour_frame, contours, -1, (0, 255, 0), 2)

    # Sharpen Filter
    kernel_sharp = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
    sharpen = cv2.filter2D(frame, -1, kernel_sharp)

    # Blur Filter
    blur = cv2.GaussianBlur(frame, (15, 15), 0)

    # Cartoon Effect
    smooth = cv2.bilateralFilter(frame, 9, 200, 200)
    cartoon_edges = cv2.Canny(smooth, 50, 150)
    cartoon_edges = cv2.cvtColor(cartoon_edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(smooth, cartoon_edges)

    # ----------------------------
    # 3. Combine 6 frames into 2×3 grid
    # ----------------------------
    top_row = cv2.hconcat([frame, gray_color, mirror_h])
    bottom_row = cv2.hconcat([edges_color, motion, contour_frame])

    combined = cv2.vconcat([top_row, bottom_row])

    # Resize back to original video size
    final_frame = cv2.resize(combined, (W, H))

    # Write output
    out.write(final_frame)

# ----------------------------
# 4. Release everything
# ----------------------------
cap.release()
out.release()

print("\n🎉 Advanced video created: advanced_processed_video.mp4")
