"""Plot saved certified parameter intervals from the current adaptive run."""
import argparse
import hashlib
import json
from pathlib import Path

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    with np.load(args.data, allow_pickle=False) as data:
        if 'dense_joint' in data:
            time = data['dense_t'].copy()
            estimates = data['dense_joint'][:, 48:54].copy()
            rho = np.sqrt(data['dense_joint'][:, 54:60] ** 2 + float(data['rho_floor']) ** 2)
        else:
            time, estimates, rho = (data[k].copy() for k in ('t', 'theta_hat', 'rho'))
        truth, prior_low, prior_high = (data[k].copy() for k in ('theta_true', 'theta_low', 'theta_high'))
    use = (time >= 0.) & (time <= .3 + 1e-12)
    time, estimates, rho = time[use], estimates[use], rho[use]
    assert len(time) >= 31 and estimates.shape == rho.shape == (len(time), 6)
    assert np.isfinite(estimates).all() and np.isfinite(rho).all() and (rho >= 0.).all()
    excess = np.abs(estimates - truth) - rho
    assert excess.max() <= 1e-10
    lower, upper = estimates - rho, estimates + rho

    # Matplotlib's bundled Computer Modern math fonts keep this reproducible
    # without an external TeX installation or a partial TeX package download.
    plt.rcParams.update({'text.usetex': False, 'mathtext.fontset': 'cm',
                         'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.sans-serif': ['DejaVu Sans'],
                         'font.monospace': ['Courier'],
                         'font.size': 13, 'axes.titlesize': 16,
                         'axes.labelsize': 14, 'xtick.labelsize': 13,
                         'ytick.labelsize': 13, 'axes.linewidth': .75,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
                         'xtick.major.width': .8, 'ytick.major.width': .8})
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.9), sharex=True)
    fig.subplots_adjust(left=.09, right=.985, top=.93, bottom=.18, wspace=.34, hspace=.4)
    names = [r'$c_x$', r'$c_z$', r'$\delta_g$', r'$\delta_m$', r'$\delta_J$', r'$\delta_\ell$']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    ticks = [[-.06, 0., .06], [-.06, 0., .06], [-.15, 0., .15],
             [-.4, -.25, 0., .25], [-.006, 0., .006], [-.003, 0., .003]]
    limits = []
    for i, ax in enumerate(axes.flat):
        color = colors[i]
        ax.fill_between(time, lower[:, i], upper[:, i], color=color, alpha=.16, linewidth=0.)
        ax.plot(time, lower[:, i], color=color, linewidth=.65)
        ax.plot(time, upper[:, i], color=color, linewidth=.65)
        ax.axhline(truth[i], color='#333333', linewidth=1.25, linestyle=(0, (3.5, 2.2)))
        ax.plot(time, estimates[:, i], color=color, linewidth=1.9)
        ax.set_title(names[i], pad=5.)
        ax.set_xlim(0., .3)
        low, high = float(lower[:, i].min()), float(upper[:, i].max())
        margin = .08 * (high - low)
        ax.set_ylim(low - margin, high + margin)
        ax.set_yticks(ticks[i])
        digits = 3 if i >= 4 else 2
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _, digits=digits: '$' + format(value, f'.{digits}f') + '$'))
        ax.set_xticks([0., .1, .2, .3])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: '$' + format(value, '.1f') + '$'))
        if i >= 3:
            ax.set_xlabel(r'time (s)', labelpad=3.)
        limits.append(list(ax.get_ylim()))
    handles = [Line2D([], [], color='#444444', linewidth=1.9, label=r'estimate $\hat{\theta}_i(t)$'),
               Line2D([], [], color='#333333', linewidth=1.25, linestyle=(0, (3.5, 2.2)), label=r'true value $\theta_i$'),
               Patch(facecolor='#dddddd', edgecolor='#777777', linewidth=.4, label=r'certified interval $\hat{\theta}_i(t) \pm \rho_i(t)$')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .015),
               ncol=3, frameon=False, fontsize=13, handlelength=2., columnspacing=1.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={'Title': 'Certified parameter intervals for the current adaptive bCBF simulation'})
    plt.close(fig)
    doc = fitz.open(args.output)
    doc[0].get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False).save(args.output.with_suffix('.png'))
    assert abs(doc[0].rect.width - 619.2) < .01 and abs(doc[0].rect.height - 352.8) < .01
    archive = args.output.parent / 'certified_parameter_bounds_data.npz'
    np.savez_compressed(archive, t=time, theta_hat=estimates, rho=rho,
                        theta_true=truth, theta_low=prior_low, theta_high=prior_high)
    report = {'source': str(args.data.resolve()),
              'text_renderer': 'Matplotlib mathtext (Computer Modern math, DejaVu Serif text); external LaTeX is not used',
              'source_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'parameters': ['cx', 'cz', 'delta_g', 'delta_m', 'delta_J', 'delta_l'],
              'true_values': truth.tolist(), 'initial_estimate': estimates[0].tolist(),
              'initial_rho': rho[0].tolist(), 'prior_lower': prior_low.tolist(),
              'prior_upper': prior_high.tolist(), 'samples': len(time),
              'time_range': [float(time[0]), float(time[-1])],
              'plotted_interval': 'theta_hat minus rho to theta_hat plus rho; no prior clipping',
              'maximum_containment_excess': float(excess.max()),
              'vertical_limits': limits, 'all_interval_endpoints_visible': True,
              'simulation_rerun': False}
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
