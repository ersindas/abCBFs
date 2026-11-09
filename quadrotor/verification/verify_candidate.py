"""Independent numerical audit of a saved candidate; never integrates its closed loop.

True and estimated backup dynamics are independently implemented below and
integrated with vectorized RK4 dt=.0005. Structured bounds are evaluated at
both .01 and .005 prediction steps. Finite dense checks are not a validated
ODE integration or a universal intersample certificate.
"""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictor import StructuredBackupPredictor
import jax
import jax.numpy as jnp
from verification_checks import (TOLERANCES, check, require_finite,
    parameter_checks, input_checks, save_summary, run_guarded)


def audit(input_path,output_dir):
 started=time.perf_counter();data=np.load(input_path);xs=data['x'];ts=data['t'];N=len(xs)
 require_finite(data, ['x','t','theta_hat','rho','theta_true','P','theta_low',
                      'theta_high','u','hbar','predicted_states','predicted_radii'])
 that=data['theta_hat'];rho=data['rho'];truth=data['theta_true'];P=data['P']
 if xs.shape != (N,6) or that.shape != (N,6) or rho.shape != (N,6) or data['u'].shape != (N,2):
  raise ValueError('State, parameter or input history has an incompatible shape')
 if N == 0 or np.any(np.diff(ts)<=0):
  raise ValueError('The saved time grid must be nonempty and strictly increasing')
 if not np.isclose(float(data['prediction_dt']),.01,rtol=0,atol=1e-12):
  raise ValueError('This experiment verifier requires the declared 0.01 s prediction grid')
 checks={}
 parameter_checks(checks,that,rho,truth,data['theta_low'],data['theta_high'])
 input_checks(checks,data['u'])
 pred01=StructuredBackupPredictor(P,dtau=.01,n_steps=100,theta_low=data['theta_low'],theta_high=data['theta_high'])
 pred005=StructuredBackupPredictor(P,dtau=.005,n_steps=200,theta_low=data['theta_low'],theta_high=data['theta_high'])
 coarse={k:np.asarray(v) for k,v in jax.vmap(pred01.evaluate)(xs,that,rho).items()}
 fine={k:np.asarray(v) for k,v in jax.vmap(pred005.evaluate)(xs,that,rho).items()}
 th_r=np.deg2rad(25);thmax=np.deg2rad(60)

 def rhs(x,pars):
  th=x[:,2];om=x[:,5];cx,cz,dg,dm,dJ,dl=pars.T
  M=np.clip(4*(th-th_r)+2*om,-20,20)
  out=np.empty_like(x);out[:,:3]=x[:,3:]
  out[:,3]=-cx*x[:,3]+20*(1+dm)*np.sin(th)
  out[:,4]=-cz*x[:,4]-9.81-dg+20*(1+dm)*np.cos(th)
  out[:,5]=-(4+dJ)*M+20*dl
  return out
 def safecomp(x):return np.stack([x[:,0]-1,x[:,1]-1,thmax**2-x[:,2]**2],axis=1)
 def backupcomp(x):
  y=x[:,[2,5]]-np.array([th_r,0]);V=np.einsum('bi,ij,bj->b',y,P,y)
  return np.stack([x[:,0]-2,x[:,1]-2,.05-V,x[:,3],x[:,4]],axis=1)
 def soft(v):return -logsumexp(-10*v,axis=1)/10
 # First N true flows, second N frozen-estimate flows.
 x=np.vstack([xs,xs]);pars=np.vstack([np.tile(truth,(N,1)),that]);dt=.0005;steps=2000
 true_store=[];est_store=[];safe_min=np.inf;safe_loc=None
 safe_comp_min=np.full(3,np.inf);moment_max=0.
 for k in range(steps+1):
  if k%10==0:
   true_store.append(x[:N].copy());est_store.append(x[N:].copy())
  hh=soft(safecomp(x[:N]));j=int(np.argmin(hh))
  if hh[j]<safe_min:safe_min=float(hh[j]);safe_loc=dict(start_time=float(ts[j]),prediction_time=k*dt,index=j)
  safe_comp_min=np.minimum(safe_comp_min,safecomp(x[:N]).min(axis=0))
  moment_max=max(moment_max,float(np.max(np.abs(4*(x[:,2]-th_r)+2*x[:,5]))))
  if k<steps:
   k1=rhs(x,pars);k2=rhs(x+dt*k1/2,pars);k3=rhs(x+dt*k2/2,pars);k4=rhs(x+dt*k3,pars)
   x=x+dt*(k1+2*k2+2*k3+k4)/6
 true_f=np.stack(true_store,axis=1);est_f=np.stack(est_store,axis=1)
 err=true_f-est_f;pe=err[:,:,[2,5]]
 actual_errors=np.stack([np.sqrt(np.maximum(0,np.einsum('nki,ij,nkj->nk',pe,P,pe))),
                         np.abs(err[:,:,3]),np.abs(err[:,:,4]),np.abs(err[:,:,0]),np.abs(err[:,:,1])],axis=-1)
 excess=actual_errors-fine['radii'];excessmax=excess.max(axis=(0,1))
 exloc=np.unravel_index(np.argmax(excess),excess.shape)
 hbt=soft(backupcomp(true_f[:,-1]));jb=int(np.argmin(hbt));bc=backupcomp(true_f[:,-1])
 hbest=soft(backupcomp(est_f[:,-1]));je=int(np.argmin(hbest))
 hbfine=fine['hbars'];hc=coarse['hbars'];hfmatch=np.concatenate([hbfine[:,:-1:2],hbfine[:,-1:]],axis=1)
 harg=np.unravel_index(np.argmin(hbfine),hbfine.shape)
 hcm=np.min(hc,axis=1);hfm=np.min(hbfine,axis=1)
 safe_tol=TOLERANCES['safety_absolute']
 bound_allowance=TOLERANCES['bound_absolute']+TOLERANCES['bound_relative']*fine['radii']
 check(checks,'true_backup_tube_safety',safe_min>=-safe_tol,
       minimum=float(safe_min),tolerance=safe_tol)
 check(checks,'true_backup_terminal_entry',hbt.min()>=-safe_tol,
       minimum=float(hbt.min()),tolerance=safe_tol)
 check(checks,'flow_error_bounds',np.all(excess<=bound_allowance),
       maximum_excess=float(excess.max()),absolute_tolerance=TOLERANCES['bound_absolute'],
       relative_tolerance=TOLERANCES['bound_relative'])
 true_running=soft(safecomp(true_f.reshape(-1,6))).reshape(N,201)
 barrier_excess=max(float((hbfine[:,:-1]-true_running).max()),float((hbfine[:,-1]-hbt).max()))
 check(checks,'tightened_barrier_is_lower_bound',barrier_excess<=TOLERANCES['bound_absolute'],
       maximum_excess=barrier_excess,tolerance=TOLERANCES['bound_absolute'])
 moment_limit=20.-TOLERANCES['moment_strict_margin']
 check(checks,'backup_moment_unsaturated',moment_max<moment_limit,
       maximum=moment_max,strict_limit=moment_limit)
 certified_moment=float(np.max(coarse['all_time_moment_bound']))
 check(checks,'all_time_moment_certificate',certified_moment<moment_limit,
       maximum=certified_moment,strict_limit=moment_limit)
 prediction_error=float(np.max(np.abs(fine['states']-est_f)))
 check(checks,'independent_prediction_agreement',prediction_error<=TOLERANCES['prediction_absolute'],
       maximum_error=prediction_error,tolerance=TOLERANCES['prediction_absolute'])
 for name,stored,recomputed in [('hbar',data['hbar'],hc),
     ('predicted_states',data['predicted_states'],coarse['states']),
     ('predicted_radii',data['predicted_radii'],coarse['radii'])]:
  if stored.shape != recomputed.shape:
   raise ValueError(f'{name} has an incompatible shape')
  difference=float(np.max(np.abs(stored-recomputed)))
  check(checks,'saved_'+name+'_consistent',difference<=TOLERANCES['saved_prediction_absolute'],
        maximum_error=difference,tolerance=TOLERANCES['saved_prediction_absolute'])
 check(checks,'finite_audit_results',all(np.all(np.isfinite(v)) for v in
       (true_f,est_f,hbfine,actual_errors,fine['radii'],coarse['hbars'])))
 summary=dict(input=str(input_path),count=N,independent_integrator='RK4',independent_dt=dt,
  true_min_safe=float(safe_min),true_min_safe_location=safe_loc,
  true_min_safe_components=safe_comp_min.tolist(),
  true_min_terminal_hb=float(hbt[jb]),true_min_terminal_start_time=float(ts[jb]),
  estimated_min_terminal_hb=float(hbest[je]),estimated_min_terminal_start_time=float(ts[je]),
  true_min_terminal_components=bc.min(axis=0).tolist(),
  min_terminal_component_start_times=[float(ts[int(np.argmin(bc[:,i]))]) for i in range(5)],
  maximum_true_or_estimated_backup_moment=moment_max,
  error_norm_order=['pitch_P','vx','vz','px','pz'],max_error_bound_excess=excessmax.tolist(),
  max_error_to_radius_ratio=np.max(actual_errors[:,1:]/np.maximum(fine['radii'][:,1:],1e-300),axis=(0,1)).tolist(),
  max_error_excess_location=dict(start_time=float(ts[exloc[0]]),prediction_time=float(exloc[1]*.005),component=int(exloc[2])),
  max_estimated_prediction_error_vs_independent_fine=np.max(np.abs(fine['states']-est_f),axis=(0,1)).tolist(),
  max_estimated_prediction_error_vs_independent_coarse=np.max(np.abs(coarse['states']-est_f[:,::2]),axis=(0,1)).tolist(),
  min_fine_hbar=float(hbfine.min()),min_fine_hbar_location=dict(start_time=float(ts[harg[0]]),constraint_index=int(harg[1])),
  min_fine_terminal=float(hbfine[:,-1].min()),
  min_fine_terminal_start_time=float(ts[int(np.argmin(hbfine[:,-1]))]),
  max_same_point_hbar_change_01_to_005=float(np.abs(hc-hfmatch).max()),
  max_minimum_hbar_drop_01_to_005=float((hcm-hfm).max()),
  max_terminal_hbar_change_01_to_005=float(np.abs(hc[:,-1]-hbfine[:,-1]).max()),
  saved_hbar_vs_recomputed_01=float(np.max(np.abs(data['hbar']-hc))) if 'hbar' in data else None,
  elapsed_seconds=time.perf_counter()-started,
  limitation='Dense numerical checks at stored closed-loop states; not a universal intersample or validated numerical integration guarantee.')
 output_dir.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(output_dir/'independent_backup_flows.npz',t=ts,taus=np.linspace(0,1,201),true_flows=true_f,estimated_flows=est_f,
   fine_hbars=hbfine,fine_radii=fine['radii'],actual_errors=actual_errors,terminal_components=bc)
 return save_summary(summary,checks,output_dir/'independent_verification.json')

def run(input_path,output_dir):
 return run_guarded(audit,input_path,output_dir,'independent_verification.json')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).with_name('candidate_verification'))
 a=p.parse_args();raise SystemExit(0 if run(a.input,a.output) else 1)
