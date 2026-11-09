"""Run all three controllers and check the adaptive result."""
from pathlib import Path
import argparse
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'data')
    args = parser.parse_args()
    output = args.output.resolve()

    for mode in ['adaptive', 'vanilla', 'nominal']:
        subprocess.run([sys.executable, str(HERE/'main_sim.py'), '--mode', mode,
                        '--output', str(output)], check=True)

    saved = output/'adaptive/continuous_candidate.npz'
    checks = output/'verification'
    commands = [
        ['verify_candidate.py', str(saved), '--output', str(checks)],
        ['verify_dense.py', '--input', str(saved), '--output', str(checks)],
        ['check_backup.py', '--output', str(checks/'smooth_backup_certificate.json')],
    ]
    for script, *arguments in commands:
        subprocess.run([sys.executable, str(HERE/'verification'/script), *arguments], check=True)
    print(f'Saved simulation data and checks to {output}')


if __name__ == '__main__':
    main()
