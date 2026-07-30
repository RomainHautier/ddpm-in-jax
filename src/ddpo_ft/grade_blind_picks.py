"""STAGE 3 of the cross-regime adaptation loop: grade whatever the BLIND selection chose.

Stage 1 (transfer_grade_636): frozen config, no adaptation      -> the baseline
Stage 2 (blind_select_636):   GT-free config choice from LR only -> the decision
Stage 3 (this):               grade that choice at 640 triplets  -> the adapted result

Picks are parsed from stage 2's log, so nothing is re-decided here. Pairs whose pick is the frozen
K3 config are skipped — stage 1 already graded those. GT is touched only for scoring, after the fact.
"""
import os, sys, pickle, re
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, '.'); sys.path.insert(0, 'src/ddpo_ft')
import numpy as np, jax, jax.numpy as jnp
from diag_guided_residual import make_kchain_ddim_sampler
from ppo_claude import build_ddim_denoiser
from train_claude import build_base_ddpm
from viz_energy import local_hik_energy
from src.rewards import make_spectrum_fn, make_residual_loss
from src.physics_guidance import make_dx_func
from src.sequence_inference import build_triplets, grid_downsample_degrade, load_sequence
from eval_ddpo import eff_resolution
MEAN,SIG,N,HIK0 = 0.0,4.7988,256,32
TEST_SEQS, PER_SEQ, KC = list(range(4,20)), 40, 6
CFGS = {'K4[200,150,100,50]x110':([200,150,100,50],110), 'K3[150,100,50]x86':([150,100,50],86),
        'K2[100,75]x50':([100,75],50), 'K1[100]x20':([100],20),
        'K1[75]x20':([75],20), 'K1[50]x12':([50],12)}
MODELS = {
 'in-dist Re=1000': 'monitoring/ddpo_re1000_k2_s100-75_ddim50_ddiminit_hk10_emabase_ckpts/ddpo_re1000_iter0299.pkl',
 'Re=2000 model':   'monitoring/ddpo_re2000_dt32_ckpts/ddpo_re1000_iter0599.pkl',
 'Re=10000 model':  'monitoring/ddpo_re10000_dt32_ckpts/ddpo_re1000_iter0549.pkl'}

# ---- parse stage-2 picks ----
picks={}; cur_R=None; cur_m=None
for line in open('monitoring/ab_pdelocal/blind_select_636.log'):
    m=re.search(r'=== Re=(\d+) BLIND SELECTION', line)
    if m: cur_R=int(m.group(1)); continue
    m=re.match(r'  (\S.*):\s*$', line)
    if m and m.group(1) in MODELS: cur_m=m.group(1); continue
    m=re.search(r'=> BLIND PICK \(band .*\): (\S+)', line)
    if m and cur_R and cur_m: picks[(cur_R,cur_m)]=m.group(1)
todo=[(R,m,c) for (R,m),c in picks.items() if c!='K3[150,100,50]x86']
print(f"parsed {len(picks)} blind picks; {len(todo)} differ from the frozen config and need grading:",flush=True)
for R,m,c in todo: print(f"  Re={R:<6} {m:<18} -> {c}",flush=True)
if not todo: print("all picks were the frozen config — stage 1 already covers them",flush=True); sys.exit()

ddpm, base_params, _ = build_base_ddpm(); ab=ddpm.alpha_bar
spec_fn=make_spectrum_fn(N); ddim20=build_ddim_denoiser(ddpm.unet,ab,100,20)
_sa,_s1=float(jnp.sqrt(ab[100])),float(jnp.sqrt(1.0-ab[100]))
PARAMS={k:pickle.load(open(v,'rb'))['params'] for k,v in MODELS.items()}
kf=np.fft.fftfreq(N,1.0/N); KR=np.sqrt(kf[:,None]**2+kf[None,:]**2)
w_hi=0.5*(1+np.tanh((KR-float(KC))/2.0))
def batched(fn,x,seed,bs=16):
    k=jax.random.PRNGKey(seed); o=[]
    for i in range(0,len(x),bs): o.append(np.asarray(fn(jnp.asarray(x[i:i+bs]),jax.random.fold_in(k,i))))
    return np.concatenate(o)
cache={}
for R,mname,cname in sorted(todo):
    if R not in cache:
        gt=f'flow-data/generated/gen_fnons_re{R}_kf_1024to256_20seq.npy'
        xg,xl,sid=[],[],[]
        for s in TEST_SEQS:
            seq=load_sequence(gt,s)
            g=build_triplets(seq,MEAN,SIG); l=build_triplets(grid_downsample_degrade(seq,4),MEAN,SIG)
            i2=np.linspace(0,len(g)-1,PER_SEQ).astype(int)
            xg.append(g[i2]); xl.append(l[i2]); sid+=[s]*PER_SEQ
        xg=np.concatenate(xg); xl=np.concatenate(xl); sid=np.array(sid)
        rec=batched(lambda xb,kk: ddim20(base_params,_sa*xb+_s1*jax.random.normal(jax.random.fold_in(kk,1),xb.shape)),xl,500)
        cache[R]=(xg,rec,sid,np.asarray(spec_fn(jnp.asarray(xg))),
                  local_hik_energy(xg[...,1]*SIG,HIK0,6.0),
                  np.load(f'base_results/regime_stats_re{R}_obsfit_gen.npz')['spec_ref'],
                  jax.jit(make_residual_loss(n=N,re=float(R),std=SIG,mean=0.0)))
    xg,rec,sid,Eg_all,Ehg,A,resid_fn=cache[R]
    E_gt=Eg_all.mean(0)
    starts,steps=CFGS[cname]
    dx=make_dx_func(n=N,re=float(R),std=SIG,mean=MEAN)
    smp=make_kchain_ddim_sampler(ddpm.unet,ab,starts,steps,dx,3.0,temp=0.30)
    sa,s1=float(jnp.sqrt(ab[starts[0]])),float(jnp.sqrt(1.0-ab[starts[0]]))
    y=batched(lambda xb,kk: smp(PARAMS[mname],sa*xb+s1*jax.random.normal(jax.random.fold_in(kk,1),xb.shape),
                                jax.random.fold_in(kk,2)),rec,700)
    us=np.unique(sid); idxs={s:np.where(sid==s)[0] for s in us}; rng=np.random.default_rng(0)
    def rep(z,tag):
        Ez=np.asarray(spec_fn(jnp.asarray(z))); E=Ez.mean(0)
        ret=float(E[HIK0:96].sum()/E_gt[HIK0:96].sum()); lo=float(E[1:5].sum()/E_gt[1:5].sum())
        Eh=local_hik_energy(z[...,1]*SIG,HIK0,6.0); pl=float(np.corrcoef(Eh.ravel(),Ehg.ravel())[0,1])
        bs=[]
        for _ in range(400):
            m2=np.concatenate([idxs[s] for s in rng.choice(us,len(us),replace=True)])
            bs.append(Ez[m2][:,HIK0:96].sum(1).mean()/Eg_all[m2][:,HIK0:96].sum(1).mean())
        print(f"  Re={R} {mname} @ {cname} {tag:<8} ret={ret:.3f} +-{np.std(bs):.3f} lowk={lo:.3f} "
              f"place={pl:.3f} resid={float(np.asarray(resid_fn(jnp.asarray(z))).mean()):.1f} "
              f"k*={eff_resolution(E,E_gt)} | blind={float(E[10:96].sum()/A[10:96].sum()):.3f}",flush=True)
    rep(y,'ADAPTED')
    yh=np.real(np.fft.ifft2(np.fft.fft2(rec,axes=(1,2))*(1-w_hi[None,:,:,None])
              +np.fft.fft2(y,axes=(1,2))*w_hi[None,:,:,None],axes=(1,2))).astype(np.float32)
    rep(yh,'+hybrid')
print("\nGRADE PICKS COMPLETE",flush=True)
