"""Run all three controllers and save the simulation data."""
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

    print(f'Saved simulation data to {output}')


if __name__ == '__main__':
    main()
