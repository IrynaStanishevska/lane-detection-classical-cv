# lane-detection-classical-cv
Classical lane detection with Canny + Hough, evaluated on simple vs challenging road conditions.

## Project structure

- `lane_detection.ipynb` — main notebook:
  - data loading (Kaggle videos),
  - Canny baseline visualization,
  - running `LaneDetector` on simple & hard videos,
  - frame-level visualization.
- `src/lane_detector.py` — implementation of the proposed lane detection pipeline.
- `src/canny_baseline.py` — simple Canny-based baseline on video frames.
- `requirements.txt` — project dependencies.
- `figures/` — example result frames.

## Baseline vs Proposed

**Baseline (`src/canny_baseline.py`):**
- Grayscale + Canny edges for each frame.
- Shows all edges; no robust lane separation.

**Proposed method (`src/lane_detector.py`):**
- Grayscale + Gaussian blur
- Canny edges
- Road region of interest (trapezoid)
- Probabilistic HoughLinesP
- Filtering by slope and position (left/right)
- Temporal smoothing of lane lines
- Lane area highlighting ("Lane" between left and right line)

## Example results

Simple daytime highway:

![Simple lane detection](figures/lane_simple_160.png)

Challenging night / urban:

![Hard lane detection](figures/lane_hard_200.png)

- On the simple video, the proposed method produces clear and stable lane markings.
- On the challenging video, detections are often unstable or incorrect,
  which demonstrates the limitations of classical CV methods in real-world conditions.

## How to run

1. Install dependencies:

```bash
pip install -r requirements.txt
