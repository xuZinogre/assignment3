#!/usr/bin/env python3
"""Assignment 3-2: detect objects in an image using Jetson SSD-MobileNet-v2.

Example:
    python3 assin3_detection.py --image=image1.jpg --threshold=0.3 --output=result1_03.jpg

Run on a Jetson device with jetson-inference and jetson-utils installed.
"""

import argparse
import sys

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput


def main():
    parser = argparse.ArgumentParser(description="Object detection on a still image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0 to 1)")
    parser.add_argument("--output", default="result.jpg", help="Path to annotated output image")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    net = detectNet("ssd-mobilenet-v2", threshold=args.threshold)
    camera = videoSource(args.image)
    output = videoOutput(args.output)

    try:
        img = camera.Capture()
        if img is None:
            raise RuntimeError("Could not load the input image: " + args.image)

        detections = net.Detect(img)
        print("Input image:", args.image)
        print("Threshold:", args.threshold)
        print("Number of detected objects:", len(detections))

        for idx, det in enumerate(detections, start=1):
            print("\nObject", idx)
            print("ClassID:", det.ClassID)
            print("Class:", net.GetClassDesc(det.ClassID))
            print("Confidence:", det.Confidence)
            print("Left:", det.Left)
            print("Top:", det.Top)
            print("Right:", det.Right)
            print("Bottom:", det.Bottom)
            print("Width:", det.Width)
            print("Height:", det.Height)
            print("Area:", det.Area)
            print("Center:", det.Center)

        output.Render(img)
        print("\nSaved annotated image to:", args.output)
    finally:
        output.Close()
        camera.Close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc, file=sys.stderr)
        sys.exit(1)
