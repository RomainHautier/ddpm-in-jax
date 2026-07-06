#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1 / 1")
#set text(size: 10.5pt)
#set par(justify: true)
#set heading(numbering: "1.")
#show heading: set block(above: 1.4em, below: 0.8em)
#show math.equation.where(block: true): set block(above: 1em, below: 1em)
#show link: underline

#align(center)[
  #text(17pt, weight: "bold")[DDPO Reward Components — Mathematical Specification]

  #text(10pt)[`ddpm-jax` · physics-aware finetuning of the SR diffusion model \
  Implementation: `src/rewards.py` · Calibration: `base_results/reward_calibration.ipynb` · 6 July 2026]
]

#v(0.5em)
*Purpose.* Exact definitions of the four reward families as implemented, with every deliberate
modelling choice flagged for theoretical review. Line references point at the implementation so
math-vs-code can be checked directly. Calibration evidence (monotonicity ladders, hackability
probes, cross-$"Re"$ tables) lives in the notebook; this document is the theory.

= Setup and conventions

Every reward maps a *normalized vorticity triplet* to a per-sample scalar distance. With
$omega$ the physical vorticity on an $N times N$ grid ($N = 256$) over the torus $(0, 2pi)^2$:

$ x = (omega - mu) / sigma in RR^(N times N times 3), quad mu = 0, quad sigma = 4.7988 $

$sigma, mu$ are the model's training statistics (Re$=1000$, first-32-sequence split) and are used
for *all* regimes — the model only ever sees this normalization; the regime anchors (§2) absorb
the residual mismatch. Channels are consecutive frames $(omega_(t-1), omega_t, omega_(t+1))$.
Statistical rewards use only the middle frame $x_"mid"$; the PDE residual uses all three. Rewards
are evaluated on the final denoised sample $x_0$ only — DDPO needs scalars, not reward gradients.

*Spectral conventions* (identical to `lambda_sweep_re1000.spectrum_fn` and
`docs/energy_spectrum_metric.md`). Unnormalized 2-D DFT $hat(x)(k) = sum_x x(x) e^(-i k dot x)$
with integer wavenumbers $k = (k_x, k_y)$, and radial shell index $kappa(k) = "round"(|k|)$. The
*enstrophy (vorticity-power) spectrum* of the normalized middle frame is the shell sum

$ E(k) = sum_(kappa(k') = k) |hat(x)_"mid" (k')|^2, quad k = 0, dots, N\/2 - 1 = 127 $

Parseval in the numpy convention, $sum_k |hat(x)(k)|^2 = N^2 sum_x x(x)^2$, gives
$sum_k E(k) = N^4 dot ⟨ x_"mid"^2 ⟩$ (with $⟨ dot ⟩$ the spatial mean),
so $E(k)$ is the total normalized enstrophy decomposed by scale. The kinetic-energy spectrum is
the same data reweighted by $1\/(2k^2)$; the enstrophy convention is used because it weights
exactly the high-$k$ band where over-smoothing lives.

Each component below is a distance $d(x) >= 0$, lower is better; the combined DDPO reward
negates them (§7).

= Regime anchors #text(10pt)[(`compute_regime_stats`, rewards.py:61)]

All rewards are *reference-free with respect to paired ground truth*: they are anchored to
regime-level statistics computed once from any $M$ frames ${x_m}$ of the target regime (train
split, or a generated run at the target Re). This is what allows the same rewards to drive
finetuning toward regimes with no paired HR data, and prevents the reward from collapsing back
into the MSE objective whose high-$k$ blindness is the thing being fixed.

$ "spec"_"ref" (k) &= 1/M sum_(m=1)^M E_m (k) & quad &"(arithmetic mean — stored, not used)" \
  Lambda(k) := log"spec"_"ref" (k) &= 1/M sum_(m=1)^M ln E_m (k) & quad &"(geometric-mean anchor — used)" \
  cal(E)_"ref" &= 1/M sum_(m=1)^M ⟨ x_m^2 ⟩ & quad &"(total-enstrophy anchor)" \
  q_"ref" (j) &= 1/M sum_(m=1)^M F_m^(-1) (j / (Q-1)), & quad & j = 0, dots, Q-1, quad Q = 257 $

where $F_m^(-1)$ is frame $m$'s empirical quantile function (linear interpolation).

*Why the geometric mean for the spectral anchor* (first-pass calibration finding): per-shell
frame spectra are approximately log-normally distributed, so the arithmetic mean lies *above*
the typical frame's spectrum. With an arithmetic anchor, a slightly blurred frame scored
*better* than GT (blur $sigma_b = 0.5$ gave $d_"spec" = 0.09$ against a GT floor of $0.24$) —
the reward paid the model to smooth. The geometric mean is the minimizer of the expected squared
log-distance, so the typical frame sits at the floor. #text(fill: rgb("#b00"))[Challenge points:]
a per-shell median anchor, or per-shell variance weighting (Mahalanobis in log space).

= Spectrum matching — $d_"spec"$, $d_("spec,hi"k)$ #text(10pt)[(rewards.py:106)]

$ d_"spec" (x) = 1/(|B|) sum_(k in B) [ ln tilde(E)(k) - Lambda(k) ]^2, quad quad
  tilde(E)(k) = max( E(k), space epsilon e^(Lambda(k)) ), quad epsilon = 10^(-6) $

$ B = {1, dots, 95} "  for " d_"spec", quad quad
  B = {32, dots, 95} "  for " d_("spec,hi"k) $

- *Log space:* $E(k)$ falls $tilde 3$ decades across the band; a linear metric would be dominated
  by the first few shells and blind at high $k$ — reproducing exactly the failure mode of L2. In
  log space every decade of the enstrophy cascade counts equally: this is a per-scale *relative*
  energy error.
- *Band:* $k = 0$ excluded (mean mode, $approx 0$ on normalized data); $k >= 96$ excluded because
  GT energy $-> 0$ there and log-ratios become noise. The high-$k$ band starts at $k = 32$, well
  above the forcing scale $k_f = 4$ — the "deficit band" of the OOD analysis.
- *Relative floor $epsilon$:* a hard-zeroed shell contributes a bounded $(ln 10^(-6))^2 approx 191$
  instead of an epsilon-driven blow-up.
- *Blind spot (by construction, proven by the scramble probe):* phases. Any field with the
  reference spectrum — including spectrum-matched noise — sits at the GT floor. Matching marginal
  spectra is necessary, not sufficient.

Relation to the tracked OOD metric: hi-$k$ retention
$R_(k>=32) = (sum_(k>=32) E_"recon" (k)) \/ (sum_(k>=32) E_"GT" (k))$ is a GT-paired linear band
ratio; $d_("spec,hi"k)$ is its reference-free log-space counterpart. Calibration: rank
correlation within-input $approx -0.7$, model-level $-1.0$.

= Energy conservation — $d_"energy"$ #text(10pt)[(rewards.py:128)]

$ d_"energy" (x) = [ ln ⟨ x_"mid"^2 ⟩ - ln cal(E)_"ref" ]^2 $

Total normalized enstrophy against the regime mean, as a squared log-ratio (symmetric: over- and
under-energized cost the same; a factor $e^(plus.minus 1)$ costs 1). By Parseval this is the
$k$-integral of the spectrum, so it is scale-blind — it pins the overall level while $d_"spec"$
fixes the distribution across scales.

Note this is *enstrophy* conservation, not kinetic energy: the KE analogue would anchor
$sum_k E(k) \/ (2k^2)$, which is low-$k$ weighted. Enstrophy was chosen for loss-alignment and
high-$k$ sensitivity. #text(fill: rgb("#b00"))[Challenge point:] if the conserved-quantity
argument should be about KE in the inverse-cascade range, swap the weighting. Calibration: weak
sensor (recon/floor $1.8 times$ at Re$=1000$), monotone only on the noise ladder — kept as a
cheap guard, weight $0.25$.

= Vorticity distribution — $d_(W_1)$ #text(10pt)[(rewards.py:141)]

The 1-Wasserstein distance between the sample's pointwise vorticity marginal $F$ and the regime
reference $G$, in the quantile representation

$ W_1 (F, G) = integral_0^1 |F^(-1)(q) - G^(-1)(q)| dif q $

discretized at $Q = 257$ evenly spaced ranks:

$ d_(W_1) (x) = 1/Q sum_(j=0)^(Q-1) | x_"sorted" [r_j] - q_"ref" (j) |, quad quad
  r_j = "round"( j (N^2 - 1) / (Q - 1) ) $

with $x_"sorted"$ the ascending sort of $x_"mid"$'s $N^2$ pixels. (Sample side: nearest-rank
order statistics; reference side: interpolated quantiles — a negligible asymmetry at
$N^2 = 65536 >> Q$.) Units: normalized vorticity.

Sensitive to PDF shape including the tails (the extreme-vorticity filaments that over-smoothing
clips); blind to *where* values sit spatially, and to phases. Calibration: the weakest sensor —
recons sit at $0.8 times$ the GT floor (the models do not measurably distort the marginal PDF on
this task); kept as a guard (weight $0.25$), candidate to drop.
#text(fill: rgb("#b00"))[Challenge point:] $W_1$ on velocity increments or on $|nabla omega|$
would target intermittency directly, which the pointwise marginal barely sees.

= PDE residual — $d_"pde"$, $d_("pde,lr")$ #text(10pt)[(rewards.py:157; physics_guidance.py:38)]

De-normalize $w = sigma x + mu$ and evaluate the Kolmogorov-flow vorticity equation residual at
the middle frame (faithful port of the BaratiLab `voriticity_residual`, same operators as the
data generator):

$ cal(R)(w) = underbrace(partial_t omega, "central FD") +
  underbrace((u dot nabla) omega - 1/"Re" Delta omega, "spectral") +
  underbrace(0.1 omega, "drag") + underbrace(4 cos(4y), "forcing") $

$ partial_t omega approx (w_3 - w_1) / (2 Delta t), quad Delta t = 1\/32, quad quad
  u = (partial_y psi, -partial_x psi), quad Delta psi = -omega $

Time derivative: central finite difference across the triplet (the only FD term). All spatial
operators are spectral on $(0, 2pi)^2$ with integer wavenumbers; the velocity comes from the
spectral Poisson solve for the streamfunction; *no* 2/3 dealiasing (matching the reference
implementation). The last two terms are the `kf_2d` source terms moved to the left side.
#text(fill: rgb("#b00"))[Assumption to challenge:] the target regime shares this forcing
($k_f = 4$) and drag ($0.1 omega$); Re is the only parameter varied.

$ d_"pde" (x) = ⟨ cal(R)(w)^2 ⟩, quad quad
  d_("pde,lr") (x) = [ ln d_"pde" (x) - ln r_"ref" ]^2 $

*The GT floor $r_"ref"$ is not zero.* Ground-truth data itself has $d_"pde" = 1.56 \/ 17.46 \/
45.99$ at Re $= 500 \/ 1000 \/ 2000$ (measured), from $O(Delta t^2)$ temporal truncation plus
spectral truncation/aliasing of unresolved scales. Two calibration findings follow:

+ Raw $d_"pde"$ mildly *rewards blurring*: mild smoothing filters the GT's own truncation noise
  (residual drops $25 -> 17$ at blur $sigma_b approx 1$) — a reward-hacking direction.
+ DDPO therefore uses $d_("pde,lr")$: the optimum is "as NS-consistent as the data itself", and
  leaving the floor in *either* direction (dirtier, or suspiciously cleaner than the physics
  allows) is penalized.

This is the only component that sees *dynamics/phases*: under spectrum-matched noise (all phases
randomized, spectrum bit-exact) $d_("pde,lr")$ fires at $146 times$ its floor while all
statistical components sit at $1.0 times$. Known wart: $r_"ref"$ varies by sequence at Re$=2000$
($approx 32$–$46$) — the anchor is noisy there; flagged to fix before Re$=2000$ DDPO.

= Combined reward and DDPO usage #text(10pt)[(`make_ddpo_reward`, rewards.py:181)]

$ r(x) = - sum_i w_i (d_i (x)) / (s_i) $

- *Scales $s_i$:* the standard deviation of $d_i$ over the *step-0 reconstruction population* at
  that regime (what the frozen model produces before finetuning), from
  `reward_calibration.json`. Purely a units choice making the $w_i$ comparable knobs.
- *Starting weights* (to revise after first runs):
  $w_"spec" = 0.5$, $w_("spec,hi"k) = 1.0$, $w_"energy" = 0.25$, $w_(W_1) = 0.25$,
  $w_"pde" = 1.0$ (log-ratio mode).
- *Advantages must be per-input.* For $K$ samples $x^((1)), dots, x^((K))$ of the same
  conditioning input:

$ A^((j)) = (r(x^((j))) - macron(r)) / ("std"_j space r(x^((j)))), quad quad
  macron(r) = 1/K sum_j r(x^((j))) $

  This is not optional: all anchors are regime-level, so each input carries a systematic offset
  (its own GT's deviation from the regime mean). That offset is constant across samples of one
  input and cancels in $A$; pooled normalization leaves it as the dominant noise. Measured:
  pooled per-frame rank correlation with hi-$k$ retention $approx 0.1$; within-input $approx -0.7$.

= Summary of deliberate choices (attack surface)

#table(
  columns: (auto, auto, 1fr),
  inset: 6pt,
  stroke: 0.5pt + rgb("#999"),
  table.header([*choice*], [*alternative not taken*], [*rationale*]),
  [enstrophy spectrum], [KE spectrum ($times 1\/2k^2$)], [weights the deficit band; loss-aligned],
  [geometric-mean anchor $Lambda(k)$], [arithmetic mean], [typical frame $approx$ floor; kills the blur exploit],
  [squared log-ratio per shell], [linear ratio / L2 on spectra], [equal weight per cascade decade],
  [band $k in [1, 96)$], [full $[0, 128)$], [$k = 0$ trivial; $k >= 96$ GT-energy noise],
  [pde as log-ratio to GT floor], [raw residual $-> 0$], [floor $eq.not 0$; raw rewards smoothing],
  [$W_1$ on pointwise $omega$ marginal], [increment / $|nabla omega|$ PDFs], [simplicity; intermittency not covered],
  [$sigma = 4.7988$ for all regimes], [per-regime normalization], [model convention; anchors absorb it],
  [middle frame for statistics], [all 3 frames], [matches `spectrum_fn` / metric conventions],
  [per-input advantage baseline], [pooled batch normalization], [regime anchors $=>$ per-input offsets],
)

#v(0.5em)
#line(length: 100%, stroke: 0.5pt + rgb("#999"))
#text(9pt)[Companion files: `docs/ddpo_reward_math.md` (this content as repo markdown) ·
`docs/energy_spectrum_metric.md` (spectral background) · `base_results/reward_calibration.ipynb`
(evidence) · `base_results/reward_calibration.json` (scales, floors, weights).]
