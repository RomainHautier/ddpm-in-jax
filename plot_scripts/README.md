# plot_scripts — clean, editable figure scripts

Self-contained: only `numpy`, `matplotlib`, and the sibling `style.py`. Figures are saved as **PDF**
by default (vector, thesis-ready); pass `--out something.png` for a raster version. Every script has a
`CONFIG` dict at the top you can edit directly; CLI flags override it. Run from anywhere:

    JAX_PLATFORMS=cpu /home/rhautier/venv-ddpm/bin/python plot_scripts/<script>.py [flags]

| script | data store | what it draws |
|---|---|---|
| `audit_spectra.py`  | `base_results/re1000_audit.npz` (single regime) | absolute spectrum + ratio to GT + per-band retention bars, for any rows |
| `audit_metrics.py`  | same | bars of ret / place / MSE / residual / low-k / any band, per-sample dots when present |
| `grid_panel.py`     | `base_results/steering_full_grid.npz` (9 regimes) | metrics vs Re, one row per model, one curve per strategy, seed bands |
| `grid_spectra.py`   | same | 3x3 per-regime spectra (ratio or absolute) for one model, placement inset |
| `style.py`          | — | palette, band shading, name maps, rc settings — edit once, restyle everything |

Store keys
- audit: `'{Re}|{row}||{field}'`, rows `LR`, `recon`, `{model}|{cfg}|{strategy}`; fields `ret place mse
  resid_ratio lowk kstar E Eb bp` and per-sample `ps_ret_paired ps_ret_ens ps_place ps_mse ps_resid psEb`.
- grid: `'{Re}|{model}|SG{strategy}||{field}'` (+ `|s701`/`|s702` seed repeats), `'{Re}|GT||E'`.
- strategies: none residual reward placement all3 rewardv2 all3v2; models: see `style.MODEL`.

Examples

    python plot_scripts/audit_spectra.py --rows "LR,recon,base0|K3|none,base0|K3|rewardv2" --panels abs,ratio
    python plot_scripts/audit_metrics.py --rows "recon,base0|K3|*,r1k-449|K3|none" --metrics ret,place,mse,resid,b2
    python plot_scripts/grid_panel.py --models re2k-149,st8k-599 --metrics b1632,b3264,place,mse
    python plot_scripts/grid_spectra.py --model rs8kkl-799 --absolute
