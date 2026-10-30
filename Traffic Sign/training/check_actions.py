"""Check real trained sign predictions through the car policy without hardware.

Input images are generated printable cards. Predictions use actual model weights;
only time, obstacle/sonar inputs and the motor interface are simulated. A passing
report does not establish recognition on photographs or physical vehicle motion.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from car.main import TrafficSignRuntime
from car.perception.sign_model import resolve_sign_model
from car.perception.traffic_signs import TrafficSignDetector


class VirtualClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RecordingMotor:
    """In-memory motor commands; no serial ports or GPIO are opened."""
    def __init__(self):
        self.latest_state = 'IDLE'
        self.commands = []

    def stop(self):
        self.commands.append({'command': 'STOP'})
        return True

    def pi_stop(self):
        self.commands.append({'command': 'PI_STOP'})
        self.latest_state = 'PI_STOPPED'
        return True

    def turn(self, angle, speed):
        self.commands.append({'command': 'TURN', 'angle': angle, 'speed': speed})
        self.latest_state = 'TURNING'
        return True


def check(model, cards, confidence=0.6):
    selected = resolve_sign_model(model)
    if selected is None:
        raise FileNotFoundError('No bundled model found; supply --model PATH')
    cards = Path(cards).expanduser().resolve()
    cases = []
    for sign in ('stop', 'no_entry', 'turn_left', 'turn_right'):
        case = {'expected_class': sign, 'image': str(cards / (sign + '.png')),
                'observations': [], 'commands': [], 'passed': False}
        try:
            image = cv2.imread(case['image'])
            if image is None:
                raise FileNotFoundError('Cannot decode card: ' + case['image'])
            clock = VirtualClock()
            detector = TrafficSignDetector(str(selected), confidence=confidence, clock=clock)
            runtime = TrafficSignRuntime(detector, clock=clock)
            motor = RecordingMotor()
            for index in range(3):
                clock.now = index * 0.3
                action = runtime.step(
                    image, motor, camera_valid=True, obstacle_detections=[],
                    obstacle_time=clock(), sonar={'left': 100.0, 'right': 100.0},
                    sonar_time=clock(), pi_latched=False, motors_enabled=True,
                    mode='autonomous',
                )
                case['observations'].append({
                    'time_seconds': clock(), 'predictions': list(runtime.sign_detections),
                    'status': runtime.status, 'decision_override': action,
                })
            case['commands'] = motor.commands
            case['policy'] = runtime.snapshot()
            recognized = all(any(item['class_name'] == sign for item in row['predictions'])
                             for row in case['observations'])
            if sign == 'stop':
                acted = (any(cmd['command'] == 'STOP' for cmd in motor.commands)
                         and 'STOP sign detected' in runtime.status
                         and runtime.policy.stop_remaining > 0)
            elif sign == 'no_entry':
                acted = (any(cmd['command'] == 'STOP' for cmd in motor.commands)
                         and runtime.policy.no_entry
                         and 'No Entry detected' in runtime.status)
            else:
                angle = -90 if sign == 'turn_left' else 90
                acted = ({'command': 'TURN', 'angle': angle, 'speed': 210} in motor.commands
                         and 'taking ' + sign.removeprefix('turn_') in runtime.status)
            case['passed'] = bool(recognized and acted)
        except Exception as error:
            case['error'] = f'{type(error).__name__}: {error}'
        cases.append(case)
    return {
        'model': str(selected), 'confidence_threshold': confidence,
        'scope': ('Actual trained-model inference on generated printable cards followed by '
                  'the vehicle policy with mocked motor commands and simulated clear sensors. '
                  'This is not a physical-camera or motor test.'),
        'passed': all(case['passed'] for case in cases), 'cases': cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='Local checkpoint/NCNN export; default: bundled model')
    parser.add_argument('--cards', type=Path, default=PACKAGE_ROOT / 'printable_cards')
    parser.add_argument('--output', type=Path, help='Optional JSON report destination')
    parser.add_argument('--conf', type=float, default=0.6)
    args = parser.parse_args()
    try:
        result = check(args.model, args.cards, confidence=args.conf)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + '\n')
    encoded = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
