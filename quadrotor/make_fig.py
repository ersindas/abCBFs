"""Generate the paper figure and certified parameter bounds from saved data."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PLOTTING = HERE / 'plotting'


def run(script, *arguments):
    subprocess.run([sys.executable, str(PLOTTING / script),
                    *map(str, arguments)], check=True, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--output', type=Path, default=HERE / 'figures')
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache/matplotlib'))
    with tempfile.TemporaryDirectory(prefix='.paper-', dir=output) as temporary:
        build = Path(temporary)
        tubes = build / 'tube_visualization'
        run('tube_geometry.py', '--case', data, '--output', tubes)
        run('render_paper.py', '--case', data, '--output', build)
        run('audit_tubes.py', '--folder', tubes)
        run('add_mass_tick.py', build / 'figure_paper_tubes.pdf', build / 'with_tick.pdf')
        run('shorten_zero_ticks.py', build / 'with_tick.pdf', build / 'final.pdf')
        run('parameter_bounds.py', '--data', data / 'adaptive/continuous_candidate.npz',
            '--output', build / 'certified_parameter_bounds.pdf')
        for extension in ['pdf', 'png']:
            shutil.copyfile(build / ('final.' + extension), output / ('figure_paper.' + extension))
        for name in ['figure_verification.json', 'certified_parameter_bounds.pdf',
                     'certified_parameter_bounds.png', 'certified_parameter_bounds.json',
                     'certified_parameter_bounds_data.npz']:
            shutil.copyfile(build / name, output / name)
        destination = output / 'tube_visualization'
        destination.mkdir(exist_ok=True)
        for path in tubes.iterdir():
            if path.suffix in ['.json', '.npz']:
                shutil.copyfile(path, destination / path.name)
    print(f'Saved {output / "figure_paper.pdf"}')
    print(f'Saved {output / "certified_parameter_bounds.pdf"}')


if __name__ == '__main__':
    main()
