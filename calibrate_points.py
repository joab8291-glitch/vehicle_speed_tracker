"""
calibrate_points.py

Grabs one frame from your video source (file, RTSP CCTV stream, or webcam),
lets you click 4 points per road region, and prints/saves the exact
SRC_ROAD_L / SRC_ROAD_R arrays to paste into vehicle_speed_tracker.py.

CLICK ORDER MATTERS for each region (this must match how the tracker script
reads the points) - click in this order:
  1. far-left   (furthest from camera, left side of that carriageway)
  2. far-right  (furthest from camera, right side of that carriageway)
  3. near-right (closest to camera, right side)
  4. near-left  (closest to camera, left side)

In other words: trace the road edges from far to near, going right along the
far edge then back left along the near edge (a trapezoid, clockwise or
counter-clockwise depending on your camera angle - just be consistent).

Usage:
  python calibrate_points.py --source video.mp4
  python calibrate_points.py --source "rtsp://admin:pass@192.168.1.64:554/..." --regions 2
  python calibrate_points.py --source 0 --regions 1 --frame-skip 30

Controls (in the window):
  Left-click       - add a point
  u                 - undo last point
  r                 - reset current region
  n / Enter         - confirm this region's 4 points and move to the next
  s                 - save all completed regions to file and print code
  q / Esc           - quit without saving
"""

import argparse
import json
import sys
import time

import cv2
import numpy as np

REGION_NAMES_DEFAULT = ["SRC_ROAD_L", "SRC_ROAD_R"]
POINT_LABELS = ["far-left", "far-right", "near-right", "near-left"]
MAX_DISPLAY_DIM = 1280  # downscale big frames so they fit on screen; clicks are rescaled back up


def parse_source(raw_source):
    if raw_source.isdigit():
        return int(raw_source)
    return raw_source


def grab_frame(source, frame_skip):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source '{source}'.")
    frame = None
    for i in range(frame_skip + 1):
        ret, frame = cap.read()
        if not ret:
            if frame is not None:
                break  # ran out of frames but we already have one
            raise RuntimeError("Could not read a frame from the source. "
                                "Try a smaller --frame-skip, or check the source is reachable.")
    cap.release()
    return frame


class Calibrator:
    def __init__(self, frame, region_names):
        self.orig_frame = frame
        self.orig_h, self.orig_w = frame.shape[:2]
        self.scale = 1.0
        if max(self.orig_w, self.orig_h) > MAX_DISPLAY_DIM:
            self.scale = MAX_DISPLAY_DIM / max(self.orig_w, self.orig_h)
        self.disp_w = int(self.orig_w * self.scale)
        self.disp_h = int(self.orig_h * self.scale)
        self.base_display = cv2.resize(frame, (self.disp_w, self.disp_h))

        self.region_names = region_names
        self.region_idx = 0
        self.completed_regions = {}  # name -> list of 4 (orig-scale) points
        self.current_points = []     # display-scale points for the region in progress

        self.window = "Calibration - click 4 points per region"
        cv2.namedWindow(self.window)
        cv2.setMouseCallback(self.window, self._on_click)

    def _on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.current_points) < 4:
            self.current_points.append((x, y))

    def _current_region_name(self):
        if self.region_idx < len(self.region_names):
            return self.region_names[self.region_idx]
        return f"REGION_{self.region_idx + 1}"

    def _to_orig_scale(self, pt):
        return [round(pt[0] / self.scale), round(pt[1] / self.scale)]

    def _render(self):
        img = self.base_display.copy()

        # Draw already-completed regions in green
        for name, pts in self.completed_regions.items():
            disp_pts = [(int(p[0] * self.scale), int(p[1] * self.scale)) for p in pts]
            for i, p in enumerate(disp_pts):
                cv2.circle(img, p, 5, (0, 220, 0), -1)
            cv2.polylines(img, [np.array(disp_pts, dtype=np.int32)], True, (0, 220, 0), 2)

        # Draw current in-progress region in yellow, with labels
        for i, p in enumerate(self.current_points):
            cv2.circle(img, p, 6, (0, 220, 255), -1)
            label = f"{i+1}:{POINT_LABELS[i]}"
            cv2.putText(img, label, (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        if len(self.current_points) >= 2:
            cv2.polylines(img, [np.array(self.current_points, dtype=np.int32)],
                           False, (0, 220, 255), 2)

        # HUD
        region_name = self._current_region_name()
        next_label = (POINT_LABELS[len(self.current_points)]
                      if len(self.current_points) < 4
                      else "region complete - press n/Enter")
        hud_lines = [
            f"Region {self.region_idx + 1}/{len(self.region_names)}: {region_name}",
            f"Next click: {next_label} ({len(self.current_points)}/4 points)",
            "u=undo r=reset region n/Enter=confirm region s=save+print q/Esc=quit",
        ]
        for i, line in enumerate(hud_lines):
            y = 24 + i * 22
            cv2.rectangle(img, (8, y - 16), (8 + 9 * len(line), y + 6), (20, 20, 20), -1)
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_DUPLEX, 0.5,
                        (230, 230, 230), 1, cv2.LINE_AA)
        return img

    def _confirm_region(self):
        if len(self.current_points) != 4:
            print(f"Need exactly 4 points before confirming (have {len(self.current_points)}).")
            return
        name = self._current_region_name()
        self.completed_regions[name] = [self._to_orig_scale(p) for p in self.current_points]
        print(f"Region '{name}' captured: {self.completed_regions[name]}")
        self.current_points = []
        self.region_idx += 1
        if self.region_idx >= len(self.region_names):
            print("All regions captured. Press 's' to save, or keep clicking to add more "
                  "(will be named REGION_N) / 'q' to quit.")

    def _save_and_print(self, out_path):
        if not self.completed_regions:
            print("Nothing captured yet - click 4 points first.")
            return
        with open(out_path, "w") as f:
            json.dump(self.completed_regions, f, indent=2)
        print(f"\nSaved raw points -> {out_path}\n")
        print("Paste this into vehicle_speed_tracker.py, replacing the existing SRC_ROAD_* blocks:\n")
        for name, pts in self.completed_regions.items():
            print(f"{name} = np.float32([")
            for label, pt in zip(POINT_LABELS, pts):
                print(f"    [{pt[0]}, {pt[1]}],  # {label}")
            print("])\n")

    def run(self, out_path):
        while True:
            cv2.imshow(self.window, self._render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                print("Quit without saving.")
                break
            elif key == ord("u"):
                if self.current_points:
                    self.current_points.pop()
            elif key == ord("r"):
                self.current_points = []
            elif key in (ord("n"), 13):  # n or Enter
                self._confirm_region()
            elif key == ord("s"):
                if self.current_points and len(self.current_points) == 4:
                    self._confirm_region()
                self._save_and_print(out_path)
                break
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Click-based calibration tool for road region points.")
    p.add_argument("--source", required=True, help="Video file, RTSP/HTTP stream URL, or webcam index.")
    p.add_argument("--regions", type=int, default=2,
                    help="How many road regions to capture (e.g. 2 for a left+right carriageway). Default: 2")
    p.add_argument("--frame-skip", type=int, default=15,
                    help="Skip this many frames before grabbing the calibration frame "
                         "(avoids a black/warm-up frame on some streams). Default: 15")
    p.add_argument("--output", default="calibration_points.json",
                    help="Where to save the raw clicked points as JSON.")
    args = p.parse_args()

    source = parse_source(args.source)
    print(f"Grabbing a frame from '{args.source}' (skipping {args.frame_skip} frames)...")
    frame = grab_frame(source, args.frame_skip)
    print(f"Got frame: {frame.shape[1]}x{frame.shape[0]}")

    region_names = REGION_NAMES_DEFAULT[:args.regions] if args.regions <= 2 else \
        [f"SRC_ROAD_{i+1}" for i in range(args.regions)]

    calibrator = Calibrator(frame, region_names)
    calibrator.run(args.output)


if __name__ == "__main__":
    main()
