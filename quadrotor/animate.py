"""Animate the saved paper figure, retaining all six drone poses at the end."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--figure', type=Path)
    parser.add_argument('--output', type=Path, default=HERE / 'figures/simulation.gif')
    parser.add_argument('--mp4', type=Path)
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--hold-seconds', type=float, default=5.)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    figure = (args.figure or output.parent / 'figure_paper.pdf').resolve()
    if not figure.exists():
        parser.error('The final paper figure is missing. Run make_fig.py first.')
    command = [sys.executable, str(HERE / 'plotting/animate_paper.py'),
               '--case', str(args.data.resolve()), '--figure', str(figure),
               '--output', str(output), '--mp4', str((args.mp4 or output.with_suffix('.mp4')).resolve()),
               '--stride', '1', '--fps', str(args.fps), '--hold-seconds', str(args.hold_seconds)]
    if args.check_only:
        command.append('--check-only')
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
