"""Audit spatial separation of displayed tubes; changes no simulation data."""
from pathlib import Path
import json
import argparse
import numpy as np
from shapely.geometry import LineString,box
from tube_geometry import tube_union

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--folder', type=Path, required=True)
folder=parser.parse_args().folder.resolve()
d=np.load(folder/'tube_preview_data.npz')
times=d['times'];centers=d['centers'];radii=d['radii'];true=d['true_positions']
tubes=[tube_union(p,r) for p,r in zip(centers,radii)]
lines=[LineString(p) for p in centers]
true_lines=[LineString(p) for p in true]
figure_audit=json.loads((folder.parent/'figure_verification.json').read_text())
legend_bounds=figure_audit['legend_updated_bbox_data']
legend=box(*legend_bounds)
rows=[]
for i in range(len(times)):
    for j in range(i+1,len(times)):
        intersection=tubes[i].intersection(tubes[j])
        rows.append(dict(times=[float(times[i]),float(times[j])],
            tubes_intersect=bool(tubes[i].intersects(tubes[j])),
            intersection_area_m2=float(intersection.area),
            distance_m=float(tubes[i].distance(tubes[j])),
            estimated_lines_intersect=bool(lines[i].intersects(lines[j])),
            true_lines_intersect=bool(true_lines[i].intersects(true_lines[j])),
            estimated_vs_other_true_lines_intersect=bool(
                lines[i].intersects(true_lines[j]) or
                lines[j].intersects(true_lines[i]))))
result=dict(display_times_s=times.tolist(),prediction_horizon_s=float(d['taus'][-1]),
    all_tubes_disjoint=all(not row['tubes_intersect'] for row in rows),
    minimum_separation_m=min(row['distance_m'] for row in rows),
    pairwise=rows,
    legend_rectangle=legend_bounds,
    legend_intersection=[dict(time=float(t),intersects=bool(g.intersects(legend)))
                         for t,g in zip(times,tubes)],
    combined_bounds_m=[(centers-radii).min(axis=(0,1)).tolist(),
                       (centers+radii).max(axis=(0,1)).tolist()],
    changed='This audit selects display snapshots from the separately simulated nonzero initial attitude case. It makes no changes to that simulation.',
    unchanged='Stored simulations, estimates, uncertainty bounds, true flows, one-second horizons, tube widths, and axis directions.',
    geometry='Tube is the union of convex hulls of adjacent rectangle cross sections. Exact for the piecewise-linear numerical rendering. No geometric smoothing or radius modification.')
(folder/'pairwise_separation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
