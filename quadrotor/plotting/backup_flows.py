"""Save backup trajectories and radii used by the paper figure."""

import argparse
from pathlib import Path
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictor import StructuredBackupPredictor


def backup_rhs(x, parameters):
    th = x[:, 2]
    om = x[:, 5]
    cx, cz, dg, dm, dJ, dl = parameters.T
    M = np.clip(4 * (th - np.deg2rad(25)) + 2 * om, -20, 20)
    out = np.empty_like(x)
    out[:, :3] = x[:, 3:]
    out[:, 3] = -cx * x[:, 3] + 20 * (1 + dm) * np.sin(th)
    out[:, 4] = -cz * x[:, 4] - 9.81 - dg + 20 * (1 + dm) * np.cos(th)
    out[:, 5] = -(4 + dJ) * M + 20 * dl
    return out


def save_backup_flows(data_dir):
    data_dir = Path(data_dir)
    with np.load(data_dir / 'adaptive.npz') as data:
        states = data['x']
        times = data['t']
        estimates = data['theta_hat']
        truth = data['theta_true']
        predictor = StructuredBackupPredictor(
            data['P'], dtau=.005, n_steps=200,
            theta_low=data['theta_low'], theta_high=data['theta_high'],
        )
        radii = np.asarray(jax.vmap(predictor.evaluate)(states, estimates, data['rho'])['radii'])

    count = len(states)
    x = np.vstack([states, states])
    parameters = np.vstack([np.tile(truth, (count, 1)), estimates])
    dt = .0005
    true_flows = []
    estimated_flows = []
    for step in range(2001):
        if step % 10 == 0:
            true_flows.append(x[:count].copy())
            estimated_flows.append(x[count:].copy())
        if step < 2000:
            k1 = backup_rhs(x, parameters)
            k2 = backup_rhs(x + dt * k1 / 2, parameters)
            k3 = backup_rhs(x + dt * k2 / 2, parameters)
            k4 = backup_rhs(x + dt * k3, parameters)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    output = data_dir / 'backup_flows.npz'
    np.savez_compressed(
        output, t=times, taus=np.linspace(0, 1, 201),
        estimated_flows=np.stack(estimated_flows, axis=1),
        true_flows=np.stack(true_flows, axis=1), fine_radii=radii,
    )
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    args = parser.parse_args()
    print(save_backup_flows(args.data))
