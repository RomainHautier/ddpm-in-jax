# Cross regime transfer, section scaffold and draft prose

Prose below follows the house style. No dashes as punctuation, no colon constructions,
figure and table names refer to files in `docs/figs_overleaf/` and `docs/tables/`.

---

## Opening (adapt as the section introduction)

> Super resolution models are expensive to specialise. The base model of this thesis was
> trained at Re equal to 1000, and the preceding chapter showed that reward fine tuning
> can produce a specialist for any single regime on the ladder. In practice, however,
> observations arrive from whatever regime the flow happens to occupy, and training one
> specialist per regime multiplies the cost of the method by the size of the ladder. This
> section asks whether that multiplication is necessary. Concretely, it asks what a
> specialist has actually learned about its regime, and whether that answer permits a
> single fine tuned model to serve the entire ladder.

> Throughout this section the reward and the guidance machinery are held fixed, and the
> models under study are the plain specialist fine tunes of the previous chapter. Every
> intervention examined here acts at inference time. This constraint is deliberate. When
> the weights, the reward, and the guide are all frozen, any change in the reconstruction
> is attributable to the sampling procedure alone, and the mechanistic claims of this
> section inherit that cleanliness. The gated dial and the gated fine tune, which relax
> the constraint, are introduced at the end of the section as work in development.

---

## Part 1. The failure under constant inference compute

Topic sentences, in argument order.

1. > Carried across the ladder under identical inference, every specialist fails in the
   > same one signed and ordered way. Below its training regime it injects too much fine
   > scale energy, above it too little, and the size of the error grows with the log
   > distance between training and target regime.
2. > The failure is spectrally confined. The large scales are preserved to within a few
   > percent at every rung, while the wavenumber at which the surplus begins retreats
   > toward higher k as the target approaches the training regime.
3. > The sharpest statement of the failure is that under fixed inference each specialist
   > produces an essentially constant output. The reconstruction clouds of one model at
   > four different target regimes span less than two units along the Reynolds axis of
   > the spectral manifold, while the truths they should match span twenty three. The
   > model does not adapt to its input regime at all, and the transfer error at any rung
   > is simply the distance from one fixed destination to that rung's truth.

Assets. `regime_ratio_dial_lowxfer_low3.pdf`, `specialist_transfer_full.tex`,
`downward_4k8k.tex`, `overshoot_onset.tex`, `manifold_transfer_clouds.pdf`,
`transfer_grid.pdf` for the visual reader, `sample_row_re4000.pdf` for the in regime
reference.

## Part 2. What the specialists share

Method justification.

> The sampler interacts with the learned weights through a single quantity, the noise
> prediction epsilon. Whatever fine tuning changed is therefore fully contained in the
> difference between the specialist's prediction and the base model's prediction on
> identical inputs, and this difference can be measured directly.

Topic sentences.

1. > The corrections of the three specialists occupy the same locations and point in the
   > same direction. Their support overlaps by roughly one half, their cosine
   > similarities lie between 0.55 and 0.64, and nearly half of the strongest correction
   > is explained by the span of the two weaker ones.
2. > What distinguishes the specialists is magnitude and spectral reach. The correction
   > norms are ordered as one to 1.38 to 2.29 across the training regimes, and the
   > surplus of the higher regime models concentrates at wavenumbers above 32.
3. > Fine tuning has therefore re weighted a shared correction rather than learned
   > regime specific physics. The models appear to exploit the common large scale
   > structure of Kolmogorov flows, and differ in how strongly they energise its fine
   > scales. This is the observation that makes an inference time remedy plausible,
   > because a difference of degree can be dosed.

Assets. `eps_correction_maps.pdf`, `eps_correction_stats.tex`, `weight_drift.pdf`
(supporting, one sentence).

## Part 3. The mechanism of dose delivery

Method justification, two instruments.

> Two complementary instruments are used. The first records the fine band energy of the
> running clean estimate at every step of the sampling chain, and answers when the dose
> is delivered. The second embeds reconstructions in a low dimensional spectral manifold,
> built from a principal component analysis of per sample log spectra over the pooled
> ground truths of four regimes, and answers where the sample travels. The leading
> component of this embedding orders the regimes and carries 96 percent of the variance,
> so trajectories through the space can be read as movement along a Reynolds axis. The
> projection collapses spectral shape to a coordinate, so proximity in the embedding is a
> necessary rather than sufficient condition for spectral agreement, and every claim made
> on the manifold is paired with a claim made on the spectra themselves.

Topic sentences, as one causal chain.

1. > The dose is set at the first clean estimate of each pass. Relative to the
   > reconstruction the chain starts from, the first estimate multiplies the fine band
   > energy by a factor between 1.6 and 3.6 that is ordered by training regime, while the
   > remaining steps of the pass change it by at most a quarter and do so almost
   > identically for every model.
2. > A deterministic and a stochastic rendering of the same chain separate the model's
   > injection from the sampler's. The gap between the two traces is the energy
   > contributed by the noise of the stochastic sampler, and it accounts for roughly a
   > third of the delivered dose at Re equal to 1000.
3. > The magnitude of the learned correction itself is nearly constant through the
   > chain. The shrinking dose of later passes is therefore a property of the sampler
   > geometry, not of the model predicting less.
4. > On the manifold, renoising launches every model from the same far off manifold
   > point, because the added noise has a flat spectrum that impersonates fine scale
   > turbulence. The base model identifies the added energy as noise and removes it,
   > returning almost exactly to its starting point. Each specialist instead retains a
   > trained fraction of it as coherent structure, and the retained fraction is the dose.
5. > The displacement produced by successive passes collapses from fourteen units to
   > three to essentially zero as the renoise depth falls. The leverage over the dose
   > therefore lies in the renoise depth and the number of passes, and not in the number
   > of denoising steps within a pass.
6. > Finally, the starting point of the journey barely depends on the regime of the
   > input. The base reconstructions of all four regimes occupy one neighbourhood of the
   > manifold, so the travel required of the chain is set almost entirely by where the
   > target truth lives.

Assets, main text. `chain_energy_trace.pdf` and `chain_energy_trace_eta1.pdf` as a pair,
`chain_injection_stats.tex`, `eps_chain_norms.pdf`, `offmanifold_pass1.pdf`,
`manifold_pass_distances.tex`, `manifold_recon_starts.pdf`, `offmanifold_topview.pdf`
(or the single panel `offmanifold_topview_re1000.pdf` beside the energy trace).

Assets, appendix. `offmanifold_passes.pdf`, `offmanifold_travel_3d.pdf`,
`step_trajectory.pdf`, `chain_dose.pdf`.

## Part 4. Adaptive inference with fixed weights

Transition and method.

> If the dose is a monotone function of the renoise depth and the pass count, and the
> transfer error is a monotone function of the distance to the training regime, then a
> per regime schedule of the chain should land every rung. Chains are selected per regime
> on the ground truth statistics of the validation pool, by the same rule that nominates
> checkpoints, and each selected chain is spent on the test pool exactly once. A fully
> blind selection rule was also evaluated and is reported honestly. Its proxies inherit
> per regime offsets from the finite training split and reject working chains outside
> Re equal to 1000, so ground truth referenced selection is retained.

Topic sentences.

1. > Removing passes and then reducing the renoise depth walks the same sample back
   > across the manifold in controlled steps, and the shortened chains are exact prefixes
   > of the full one under a shared noise sequence.
2. > The selected rungs form an almost regular ladder. A single pass at depth 100 lands
   > Re equal to 1000, depth 120 lands 1500, 125 lands 2000, 150 lands 3000, and 160
   > lands 4000, with test retentions between 0.90 and 1.10 and in band shares up to 94
   > percent.
3. > The two halves of the ladder call for different tools. Below the training regime the
   > chain is the mechanism, because guidance cannot suppress an overdose approaching
   > four times the truth. At and above it the full chain is the correct depth and
   > guidance provides the remaining lift.
4. > The tapered dial composes with a tuned chain as a signed per sample corrector. It
   > pulls a small surplus down and pushes a small deficit up, at some cost in per sample
   > placement below the training regime.
5. > One model therefore reaches every rung of the ladder to within seven percent of the
   > true band energy. What no configuration changes is the highest octave, which
   > collapses beyond k of roughly 64 in every setting and marks the boundary between
   > what inference adaptation can and cannot recover.

Assets, main text. `offmanifold_dechain_endpoints.pdf`, `re8k_downward.tex`,
`offmanifold_bestconfig.pdf`, `bestconfig_ratio.pdf`.

Assets, appendix. `offmanifold_dechain.pdf`, `offmanifold_chainfix.pdf`,
`manifold_chain_clouds.pdf`, `chain_ladder_spectra.pdf`, `bestconfig_spectra.pdf`.

## Part 5. Toward adaptive guidance, the gate and the gated fine tune

Framing sentence.

> The preceding parts held the guidance fixed and adapted the schedule. The natural next
> step is to make the guidance itself adaptive, and this part presents that step as work
> in development rather than as a settled method.

Topic sentences.

1. > The per band gate steers each sample toward its own predicted energy level and
   > shuts off band by band as the target is reached. With the gate, every model lands at
   > and above its own training regime, and the gated Re equal to 8000 specialist covers
   > the entire upper half of the ladder with retentions between 0.95 and 1.04.
2. > On the manifold the gate moves the reconstruction cloud in the required direction,
   > along the Reynolds axis and back toward the arc of physical spectra that the
   > unguided cloud had left.
3. > The gate has two demonstrated limits. It cannot create energy a model does not
   > generate, so gated retention still decays above each model's home, and it cannot
   > fully suppress the overdose far below it.
4. > Moving the adaptive correction into the weights is the subject of the gated dose
   > fine tune, whose first results are provisional and which is left as the forward
   > direction of this work.

Assets, main text. The gate grid table (extract from `specialist_transfer_full.tex` or a
dedicated small table), `regime_ratio_dial_xfer.pdf`, `manifold_recon_clouds.pdf`.

Assets, appendix. `regime_ratio_dial_r8k.pdf`, `regime_ratio_dial_r4k.pdf`,
`regime_pertriplet_gate.pdf`, `regime_ratio_dial_gate.pdf`.

---

## Figure and table checklist

Main text, in order of appearance.

| # | asset | part | carries |
|---|---|---|---|
| F1 | regime_ratio_dial_lowxfer_low3.pdf | 1 | over energisation under constant compute |
| T1 | specialist_transfer_full.tex | 1 | the full grid, four specialists |
| T2 | overshoot_onset.tex | 1 | low k preserved, onset retreats |
| F2 | manifold_transfer_clouds.pdf | 1 | the attractor result |
| F3 | eps_correction_maps.pdf | 2 | shared support, different amplitude |
| T3 | eps_correction_stats.tex | 2 | cosines, overlap, magnitude ordering |
| F4 | chain_energy_trace.pdf + _eta1 | 3 | dose at first estimate, noise share |
| T4 | chain_injection_stats.tex | 3 | injection multipliers |
| F5 | eps_chain_norms.pdf | 3 | correction magnitude through the chain |
| F6 | offmanifold_pass1.pdf | 3 | shared launch, five walks back |
| F7 | offmanifold_topview_re1000.pdf | 3 | pairs with F4 |
| T5 | manifold_pass_distances.tex | 3 | moves shrink with t |
| F8 | manifold_recon_starts.pdf | 3 | start is regime independent |
| F9 | offmanifold_dechain_endpoints.pdf | 4 | the ladder walked back |
| T6 | re8k_downward.tex | 4 | chain rungs, test confirmed |
| F10 | offmanifold_bestconfig.pdf | 4 | best config per regime |
| F11 | bestconfig_ratio.pdf | 4 | the spectra of the best configs |
| T7 | downward_4k8k.tex | 1 or 4 | both specialists below home |
| F12 | regime_ratio_dial_xfer.pdf | 5 | gate against the two extremes |
| F13 | manifold_recon_clouds.pdf | 5 | gate returns to the arc |

Appendix. offmanifold_passes, offmanifold_dechain (trajectory version),
offmanifold_chainfix, manifold_chain_clouds, offmanifold_travel(_3d), step_trajectory,
chain_dose, chain_ladder_spectra, bestconfig_spectra, regime_ratio_dial_r4k / _r8k,
regime_pertriplet_gate, transfer_grid (or main text if a visual anchor is wanted early),
sample_row_re4000.

Unused by this section (belongs to the in distribution chapter or superseded).
offmanifold_travel.pdf (superseded by topview), manifold_pca.pdf, gain_vs_re family,
indist panels, re1000 family, deficit panels, ood_finetune_ladder.
