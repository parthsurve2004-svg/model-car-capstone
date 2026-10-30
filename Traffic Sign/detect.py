"""Motor-free sign inference for an image, video, or webcam.

Uses only the sign predictor; it never imports the vehicle control runtime.
Video sampling follows media timestamps at 4Hz, rather than decode-loop speed.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from car.perception.sign_model import resolve_sign_model
from car.perception.traffic_signs import TrafficSignDetector, sign_geometry


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def annotate(frame, detections):
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 40), 2)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(annotated, label, (max(0, x1), max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 180, 40), 1)
    return annotated


def run(source, *, model=None, output='sign-detections', confidence=.6, rate_hz=4.,
        max_frames=None, detector_factory=TrafficSignDetector):
    """Write annotated output and JSON. max_frames bounds webcam/video runs."""
    selected = resolve_sign_model(model)
    if selected is None:
        raise FileNotFoundError('Bundled traffic-sign weights are missing; supply --model PATH')
    if max_frames is not None and max_frames <= 0:
        raise ValueError('max_frames must be positive')
    source_text = str(source)
    webcam = source_text.isdecimal()
    path = None if webcam else Path(source_text).expanduser().resolve()
    if path is not None and not path.is_file():
        raise FileNotFoundError(f'Source does not exist: {path}')
    image = path is not None and path.suffix.lower() in IMAGE_EXTENSIONS
    media_time = [0.]
    clock = time.monotonic if webcam else lambda: media_time[0]
    detector = detector_factory(str(selected), rate_hz=rate_hz, confidence=confidence, clock=clock)
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    capture = writer = None
    rows = []
    previous = []
    index = 0
    try:
        if image:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f'Image cannot be decoded: {path}')
            frames = [(frame, 0.)]
            fps = 1.
        else:
            capture = cv2.VideoCapture(int(source_text) if webcam else str(path))
            if not capture.isOpened():
                raise ValueError(f'Video/camera cannot be opened: {source}')
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not 0 < fps < 1000:
                fps = 30.
            frames = None
        while True:
            if image:
                if index:
                    break
                frame, timestamp = frames[0]
            else:
                okay, frame = capture.read()
                if not okay:
                    break
                timestamp = time.monotonic() if webcam else index / fps
            media_time[0] = timestamp
            detected = detector.poll(frame)
            fresh = detected is not None
            if fresh:
                previous = detected
            detections = [{**item, **sign_geometry(item, frame.shape)} for item in previous]
            rows.append({'frame': index, 'timestamp_seconds': timestamp,
                         'fresh_prediction': fresh, 'detections': detections})
            annotated = annotate(frame, previous)
            if image:
                target = destination / 'annotated.jpg'
                if not cv2.imwrite(str(target), annotated):
                    raise RuntimeError('Could not write annotated image')
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(destination / 'annotated.mp4'),
                                             cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                             (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Could not open annotated video writer')
                writer.write(annotated)
            index += 1
            if max_frames is not None and index >= max_frames:
                break
        if not rows:
            raise ValueError('Source contained no decodable frames')
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
    result = {'model': str(selected), 'source': source_text,
              'prediction_rate_limit_hz': rate_hz, 'frames': rows}
    (destination / 'detections.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Image/video path, or webcam index such as 0')
    parser.add_argument('--model', help='Existing local checkpoint/export; otherwise use bundled weights')
    parser.add_argument('--output', default='sign-detections', help='Directory for annotated media and JSON')
    parser.add_argument('--conf', type=float, default=.6)
    parser.add_argument('--rate-hz', type=float, default=4.)
    parser.add_argument('--max-frames', type=int, help='Optional frame limit (especially for webcam)')
    args = parser.parse_args()
    try:
        result = run(args.source, model=args.model, output=args.output,
                     confidence=args.conf, rate_hz=args.rate_hz, max_frames=args.max_frames)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + '\n')
    fresh = sum(row['fresh_prediction'] for row in result['frames'])
    print(f"Weights: {result['model']}")
    print(f"Processed {len(result['frames'])} frames; {fresh} fresh sign predictions")
    print(f"Annotated media and detections.json: {Path(args.output).resolve()}")


if __name__ == '__main__':
    main()
