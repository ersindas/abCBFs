"""Position tubes from already saved backup predictions and radii.

The analytical cross section at tau is the axis-aligned rectangle
  p_hat(tau) + [-r_px(tau),r_px(tau)] x [-r_pz(tau),r_pz(tau)].
The analytical projected tube is their union for 0 <= tau <= T.

The polygon rendering below is exact for piecewise-linear interpolation of
the supplied centers and radii. It is a numerical rendering of the continuous
tube, not a separate certified between-node enclosure. No decorative spline
smoothing or manually chosen corridor width is used.
"""
import numpy as np


def rectangle_corners(center, radii):
    center=np.asarray(center,dtype=float)
    radii=np.asarray(radii,dtype=float)
    if center.shape!=(2,) or radii.shape!=(2,) or np.any(radii<0):
        raise ValueError('Expected a 2D center and nonnegative half widths.')
    return center+np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])*radii


def convex_hull(points):
    """Monotone-chain hull; handles the zero-radius initial cross section."""
    pts=sorted(set(map(tuple,np.asarray(points,dtype=float))))
    if len(pts)<=1:
        return np.asarray(pts,dtype=float).reshape(-1,2)
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    low=[]
    for p in pts:
        while len(low)>=2 and cross(low[-2],low[-1],p)<=0:
            low.pop()
        low.append(p)
    up=[]
    for p in reversed(pts):
        while len(up)>=2 and cross(up[-2],up[-1],p)<=0:
            up.pop()
        up.append(p)
    return np.asarray(low[:-1]+up[:-1])


def swept_polygons(centers,radii):
    centers=np.asarray(centers,dtype=float)
    radii=np.asarray(radii,dtype=float)
    if centers.shape!=radii.shape or centers.ndim!=2 or centers.shape[1]!=2:
        raise ValueError('centers and radii must both have shape (N,2).')
    if np.any(radii<0):
        raise ValueError('Negative radii are invalid.')
    boxes=[rectangle_corners(p,r) for p,r in zip(centers,radii)]
    return [convex_hull(np.vstack([a,b])) for a,b in zip(boxes[:-1],boxes[1:])]


def tube_union(centers,radii):
    """Single polygon union for uniform-opacity plotting; requires shapely."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    return unary_union([Polygon(p) for p in swept_polygons(centers,radii)
                        if len(p)>=3])


def pointwise_containment(true_positions,centers,radii,tolerance=0.):
    error=np.abs(np.asarray(true_positions)-np.asarray(centers))
    radii=np.asarray(radii)
    excess=error-radii
    ratios=np.divide(error,radii,out=np.zeros_like(error),where=radii>1e-14)
    return dict(maximum_excess_per_axis=excess.max(axis=0).tolist(),
                maximum_error_ratio_per_axis=ratios.max(axis=0).tolist(),
                contained=bool(np.all(excess<=tolerance)))


if __name__=='__main__':
    import json
    import hashlib
    from pathlib import Path
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--case', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--first-snapshot', type=float, default=.08)
    args=ap.parse_args()
    root=args.case.resolve()
    source=root/'adaptive/continuous_candidate.npz'
    independent=root/'verification/independent_backup_flows.npz'
    d=np.load(source);f=np.load(independent)
    if not np.array_equal(d['t'],f['t']):
        raise ValueError('Verification and trajectory update times differ.')
    if not (np.all(d['theta_true']>=d['theta_low']) and
            np.all(d['theta_true']<=d['theta_high'])):
        raise ValueError('The true parameter is outside the stored prior.')
    times=np.array([args.first_snapshot,.3,.5,1.1,1.93])
    coarse_tau=np.arange(d['predicted_states'].shape[1])*float(d['prediction_dt'])
    fi=np.searchsorted(f['taus'],coarse_tau)
    if not np.allclose(f['taus'][fi],coarse_tau,atol=1e-12):
        raise ValueError('Independent prediction samples do not align.')
    tau=f['taus']
    summaries=[];centers=[];radii=[];trues=[]
    for t in times:
        i=int(np.argmin(abs(d['t']-t)))
        assert abs(d['t'][i]-t)<1e-12
        p=f['estimated_flows'][i,:,:2]
        r=f['fine_radii'][i,:,3:5]
        truth=f['true_flows'][i,:,:2]
        check=pointwise_containment(truth,p,r)
        coarse_check=pointwise_containment(f['true_flows'][i,fi,:2],
            d['predicted_states'][i,:,:2],d['predicted_radii'][i,:,3:5])
        summaries.append(dict(time=float(t),
            maximum_half_widths_m=r.max(axis=0).tolist(),
            maximum_full_widths_m=(2*r.max(axis=0)).tolist(),
            terminal_half_widths_m=r[-1].tolist(),
            bounds_m=[(p-r).min(axis=0).tolist(),(p+r).max(axis=0).tolist()],
            rho=d['rho'][i].tolist(),
            coarse_prediction_comparison=coarse_check,**check))
        centers.append(p);radii.append(r);trues.append(truth)
    result=dict(source=str(source),independent_source=str(independent),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        independent_sha256=hashlib.sha256(independent.read_bytes()).hexdigest(),
        theta_true=d['theta_true'].tolist(),
        initial_state=d['x'][0].tolist(),
        initial_pitch_degrees=float(np.rad2deg(d['x'][0,2])),
        initial_angular_rate_rad_s=float(d['x'][0,5]),
        parameter_low=d['theta_low'].tolist(),parameter_high=d['theta_high'].tolist(),
        all_selected_fine_flows_contained=all(s['contained'] for s in summaries),
        combined_bounds_m=[np.min([s['bounds_m'][0] for s in summaries],axis=0).tolist(),
                           np.max([s['bounds_m'][1] for s in summaries],axis=0).tolist()],
        radius_order=['pitch_P','vx','vz','px','pz'],
        cross_section='Axis-aligned rectangle with half widths r_px and r_pz.',
        rendering='Union of adjacent cross-section convex hulls: exact for linearly interpolated centers and radii, numerical approximation of continuous tube.',
        snapshots=summaries,
        note='Uses saved data only. No controller, estimator, simulation, or uncertainty modification. Preview uses independent estimated and true flows with the saved fine radii at 0.005s prediction intervals. Comparing independent true flows with the original coarser prediction centers exposes tiny integration errors at very narrow early cross sections; these excesses are reported separately. Containment checks compare the stored true parameter flow with numerical bounds, not every parameter in the box.')
    output=args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/'tube_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_selected_fine_flows_contained']:
        raise SystemExit('Tube verification failed: a stored true flow leaves its position bounds. See tube_audit.json.')
    np.savez_compressed(output/'tube_preview_data.npz',times=times,taus=tau,
                        centers=np.asarray(centers),radii=np.asarray(radii),
                        true_positions=np.asarray(trues))
    print(json.dumps(result,indent=2))
