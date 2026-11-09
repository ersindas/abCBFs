"""Position-tube polygons from linearly interpolated centers and radii."""
import numpy as np


def rectangle_corners(center, radii):
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if center.shape != (2,) or radii.shape != (2,) or np.any(radii < 0):
        raise ValueError('Expected a 2D center and nonnegative half widths.')
    return center + np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]) * radii


def convex_hull(points):
    """Monotone-chain hull; handles the zero-radius initial cross section."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    low = []
    for p in pts:
        while len(low) >= 2 and cross(low[-2], low[-1], p) <= 0:
            low.pop()
        low.append(p)
    up = []
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.asarray(low[:-1] + up[:-1])


def swept_polygons(centers, radii):
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.shape != radii.shape or centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError('centers and radii must both have shape (N,2).')
    if np.any(radii < 0):
        raise ValueError('Negative radii are invalid.')
    boxes = [rectangle_corners(p, r) for p, r in zip(centers, radii)]
    return [convex_hull(np.vstack([a, b])) for a, b in zip(boxes[:-1], boxes[1:])]
