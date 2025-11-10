import cv2
import numpy as np


class LaneDetector:
    """
    Classical lane line detector based on:
    - grayscale + Gaussian blur
    - Canny edges
    - region of interest (road trapezoid)
    - Probabilistic HoughLinesP
    - line filtering by slope and position
    - temporal smoothing
    - optional lane area highlighting between left/right lines
    """

    def __init__(self):
        self.prev_left = None
        self.prev_right = None

    def region_of_interest(self, img):
        """
        Apply a trapezoid mask to keep only the road area.
        """
        h, w = img.shape[:2]
        polygon = np.array([[
            (int(0.05 * w), h),
            (int(0.45 * w), int(0.6 * h)),
            (int(0.55 * w), int(0.6 * h)),
            (int(0.95 * w), h)
        ]], dtype=np.int32)

        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(img, mask)

    def _separate_lines(self, lines, img_shape):
        """
        Split raw Hough segments into left and right lane candidates
        based on slope and position relative to image center.
        """
        if lines is None:
            return [], []
        h, w = img_shape[:2]
        center_x = w / 2
        left, right = [], []

        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)

            # reject almost horizontal segments
            if abs(slope) < 0.3:
                continue

            # negative slope on the left side
            if slope < 0 and x1 < center_x and x2 < center_x:
                left.append((x1, y1, x2, y2))
            # positive slope on the right side
            elif slope > 0 and x1 > center_x and x2 > center_x:
                right.append((x1, y1, x2, y2))

        return left, right

    def _make_line(self, points, img_shape):
        """
        Fit a single line (x1, y1, x2, y2) through a set of segments using linear regression.
        """
        if not points:
            return None

        h, w = img_shape[:2]
        xs, ys = [], []
        for x1, y1, x2, y2 in points:
            xs += [x1, x2]
            ys += [y1, y2]

        xs = np.array(xs)
        ys = np.array(ys)
        if len(xs) < 2:
            return None

        m, b = np.polyfit(xs, ys, 1)
        if m == 0:
            return None

        # compute points at the bottom of the image and some height above
        y1 = h
        y2 = int(h * 0.6)
        x1 = int((y1 - b) / m)
        x2 = int((y2 - b) / m)
        return (x1, y1, x2, y2)

    def _smooth(self, prev, new, alpha=0.7):
        """
        Exponential smoothing of line coordinates over time.
        """
        if prev is None or new is None:
            return new
        return tuple(int(alpha * p + (1 - alpha) * n) for p, n in zip(prev, new))

    def detect(self, frame):
        """
        Run lane detection on a single BGR frame and return visualization.
        """
        # 1. Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2. ROI + HoughLinesP
        roi = self.region_of_interest(edges)
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=40,
            maxLineGap=100
        )

        # 3. Grouping & smoothing
        left_pts, right_pts = self._separate_lines(lines, frame.shape)
        left_line = self._make_line(left_pts, frame.shape) if left_pts else None
        right_line = self._make_line(right_pts, frame.shape) if right_pts else None

        left_line = self._smooth(self.prev_left, left_line)
        right_line = self._smooth(self.prev_right, right_line)
        self.prev_left = left_line
        self.prev_right = right_line

        # 4. Draw lane lines
        line_img = np.zeros_like(frame)

        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 6)

        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 6)

        out = cv2.addWeighted(frame, 0.8, line_img, 1.0, 0)

        # 5. Highlight lane area if both lines are available
        if left_line is not None and right_line is not None:
            x1_l, y1_l, x2_l, y2_l = left_line
            x1_r, y1_r, x2_r, y2_r = right_line

            lane_overlay = np.zeros_like(frame)

            # polygon: bottom-left, top-left, top-right, bottom-right
            poly = np.array([[
                (x1_l, y1_l),
                (x2_l, y2_l),
                (x2_r, y2_r),
                (x1_r, y1_r)
            ]], dtype=np.int32)

            cv2.fillPoly(lane_overlay, poly, (0, 0, 255))
            out = cv2.addWeighted(out, 0.7, lane_overlay, 0.3, 0)

            # "Lane" label approximately in the middle
            cx = int((x1_l + x1_r) / 2)
            cy = int((y1_l + y1_r) / 2)
            cv2.putText(
                out,
                "Lane",
                (cx - 40, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return out


def run_lane_detector(input_path, output_path, max_frames=None):
    """
    Run LaneDetector on a video file and save the visualization.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Cannot open:", input_path)
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    detector = LaneDetector()

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame = detector.detect(frame)
        out.write(out_frame)

        i += 1
        if max_frames and i >= max_frames:
            break

    cap.release()
    out.release()
    print(f"Saved lane detection to: {output_path}")
