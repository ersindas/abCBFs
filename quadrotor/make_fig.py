"""Generate the paper figure from saved simulation data."""
import argparse
from pathlib import Path

from plotting.render_paper import render


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=here / 'data')
    parser.add_argument('--output', type=Path, default=here / 'figures')
    args = parser.parse_args()
    render(args.data, args.output)
    print(f'Saved {args.output / "figure_paper.pdf"}')
    print(f'Saved {args.output / "figure_paper.png"}')


if __name__ == '__main__':
    main()
