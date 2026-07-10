# DDPO — open backlog

Tracked follow-ups from the DDPO physics-reward investigation. See `ddpo_findings.md` (narrative),
`ddpo_report.pdf` (paper), and the results artifact for the completed work.

---

## 1. Make the PDE residual go DOWN toward the GT floor (priority)

**Goal:** get the finetuned model's NS residual to move *toward* the GT residual floor, instead of the
current flat-to-slightly-up behavior.

**State (2026-07-09).** At grid-4× the base reconstruction is already near-physical (mean |resid| ~3.8
vs GT floor ~1.5). DDPO's spectral reward *adds* high-k energy, which sharpens filaments and *raises* the
residual slightly (base 3.8 → DDPO 3.9; K=3 → 4.2). The recon residual field is bright and **speckled
everywhere**, whereas GT's residual is dark/sparse/clean on a few coherent filaments
(`monitoring/ab_pdelocal/viz_residual_k1k3.png`). **PDE-residual placement** stays ~0.36 — below the ~0.6
"useful" line — at every resolution reached; it is a fine-scale derivative field that aligns with GT far
more slowly than energy does. The `pde_local` reward (worst-region residual) was a controlled **negative**.

**Why it matters.** Energy/spectrum is solved at grid-4× (retention +0.21, k* 31→95, placement 0.81, MSE
flat), but physics-consistency at the finest scales is the one unsolved axis.

**KEY FINDING (2026-07-09, `diag_residual_landscape`).** GT is the PDE-residual **minimum**: GT residual is
lowest at full resolution (1.31) and *rises* under any smoothing (up to 4.78) — smoothing breaks the NS
balance. So **reducing residual tends TOWARD GT, not toward an over-smoothed solution** (the earlier
over-smoothing worry is ruled out; the reward premise is sound). The recon's excess residual (3.65 vs GT
1.31) is **high-k speckle**: the residual power spectrum shows base/DDPO above GT *only* at k>32, and
low-passing k>64 drops recon residual to ~2.1. Interpretation: DDPO adds high-k *energy* with high-k
*residual* — right amount, wrong dynamics (phase). GT has high-k energy AND low high-k residual because its
fine structure is physical.

**The reward to build:** a **spectral-residual term** that penalizes residual POWER at k>32 (the speckle) —
driving the added structure toward NS-consistent/physical (toward GT) — WITHOUT smoothing away the high-k
*energy* (a naive smoothness prior kills the retention DDPO worked for and gets stuck at residual ~2.1).
The two must be separated: keep the enstrophy spectrum, kill the residual spectrum at high k.

**Other ideas (untested):**
- **Two-sided / stronger pde term** so the reward pushes residual *toward* the floor (current hinge only
  penalizes *above* the floor → tolerates the speckle). Now known safe since GT is the residual minimum.
- **Gentler multi-phase (K=2) / lower sampling temperature** — K=3 over-sharpens and raises residual.
- **Denser input** (>11% sampling) where PDE placement finally climbs — the information-limited fallback.

---

## 2. Multi-phase K=2 / lower-temp test

K=3 helps placement but over-amplifies in-distribution (harmful) and is useful only OOD (capacity-starved).
Test K=2 (S=[100,50]) or lower temperature to capture the placement/sharpness benefit without the amplitude
overshoot — turning the "K=3 backfires" caveat into a usable in-dist lever. (`eval_multiphase`,
`diag_multiphase_spatial` already parametrized.)

## 3. Conditioned-model ("learned head") comparison

All results use the **unconditional** base + SDEdit + DDPO. Compare against a model with a learned
conditioning head (`config_field_cond.yaml`, `check_conditioning_training.ipynb`) at grid-4×: does an
explicit conditioning path beat SDEdit for reconstruction, and does DDPO still add on top?

## 4. Pure zero-lower-regime extrapolation

The Re=2000 extrapolation uses anchors from {Re=500, 1000}. Push further: extrapolate to a regime with
*no* bracketing lower-Re data, or extrapolate the `align` reference (currently the Re=1000 value) too.

## 5. Loose ends
- Re=1000 grid-4× enstrophy-vs-k spectrum with the input curve (`eval_multiphase --re 1000 --grid_factor 4`)
  — only the Re=2000 one was generated.
- Resolution sweep at intermediate densities (2048, 3072 pts) to pin where placement crosses 0.6 more
  precisely (currently bracketed to ~3000).
