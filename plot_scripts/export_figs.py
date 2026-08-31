"""Regenerate every figure and copy it into docs/figs_overleaf/ under a stable name.

Each figure is produced by its own script in this folder - edit that script (they all carry a
CONFIG dict at the top and CLI overrides) and re-run this to refresh the PDFs.

  JAX_PLATFORMS=cpu python plot_scripts/export_figs.py [--no-regen]
"""
import os, re, sys, shutil, subprocess, argparse
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
PY = sys.executable
OUT = 'docs/figs_overleaf'
# (script, extra args, produced file, name in the paper)
FIGS = [
    ('indist_ladder.py',   [],                  'indist_ensemble.pdf',      're1000_ensemble_spectrum.pdf'),
    ('indist_ladder.py',   [],                  'indist_perframe.pdf',      're1000_pertriplet_dose_k32.pdf'),
    ('indist_ladder.py',   ['--band', '16,96'], 'indist_perframe_k16.pdf',  're1000_pertriplet_dose_k16.pdf'),
    ('placement_vs_k.py',  [],                  'placement_vs_k.pdf',       're1000_placement_vs_k.pdf'),
    ('total_energy.py',    [],                  'total_energy.pdf',         're1000_total_energy.pdf'),
    ('deficit_panel.py',   [],                  'deficit_panel.pdf',        're1000_deficit_panel.pdf'),
    ('training_curves.py', [],                  'training_curves_re1000.pdf', 're1000_training_curves.pdf'),
]
ap = argparse.ArgumentParser(); ap.add_argument('--no-regen', action='store_true')
a = ap.parse_args()
os.makedirs(OUT, exist_ok=True)
env = dict(os.environ, JAX_PLATFORMS='cpu')
done = set()
for script, args, produced, name in FIGS:
    key = (script, tuple(args))
    if not a.no_regen and key not in done:
        r = subprocess.run([PY, f'plot_scripts/{script}'] + args, capture_output=True, text=True, env=env)
        if r.returncode: print(f"  {script} {' '.join(args)} FAILED:\n{r.stderr[-400:]}"); continue
        done.add(key)
    src = f'plotting/figs/{produced}'
    if not os.path.exists(src): print(f"  missing {src}"); continue
    shutil.copy(src, f'{OUT}/{name}')
    d = open(src, 'rb').read()
    imgs = len(re.findall(rb'/Subtype\s*/Image', d))
    t3 = len(re.findall(rb'/Subtype\s*/Type3', d))
    tt = len(re.findall(rb'/FontFile2', d))
    flag = 'vector, TrueType' if (imgs == 0 and t3 == 0 and tt > 0) else \
           f'raster:{imgs} type3:{t3} truetype:{tt}'
    print(f"  {name:<36}{os.path.getsize(src)//1024:>5} KB   {flag}")
print(f"\n-> {OUT}/  (edit plot_scripts/*.py and re-run this to refresh)")
