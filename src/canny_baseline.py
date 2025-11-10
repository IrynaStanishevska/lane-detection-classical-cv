import cv2
import numpy as np

def run_canny_baseline(video_path, output_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open:", video_path)
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        out.write(edges_bgr)

        i += 1
        if max_frames and i >= max_frames:
            break

    cap.release()
    out.release()
    print("Saved Canny baseline to:", output_path)
