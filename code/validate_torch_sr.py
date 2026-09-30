import argparse
import sys
from pathlib import Path
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='model.pt written by context_sr_explore_torch.py')
parser.add_argument('--source', default=None, help='Path to context_sr_explore_torch.py; defaults to same directory.')
args0 = parser.parse_args()

model_path = Path(args0.model).resolve()
source_path = Path(args0.source).resolve() if args0.source else Path(__file__).with_name('context_sr_explore_torch.py').resolve()
ck = torch.load(model_path, map_location='cpu', weights_only=False)
cfg = ck.get('args', {})

# Recreate the script's global configuration using the checkpoint config, but do not execute its Main section.
argv = [str(source_path)]
for key in ['Lx','Ly','t_steps','t_max','d_model','num_heads','num_layers','ctx_tokens','train_protocols','test_protocols','dtype','sr_reg','sr_rcond']:
    if key in cfg:
        flag = '--' + key.replace('_','-')
        argv += [flag, str(cfg[key])]
argv += ['--steps','0','--out',str(model_path.parent / '_validate_tmp')]
saved_argv = sys.argv
sys.argv = argv
src = source_path.read_text()
prefix = src.split('# ----------------------------\n# Main\n# ----------------------------')[0]
g = {'__file__': str(source_path), '__name__': '_context_sr_defs_'}
exec(compile(prefix, str(source_path), 'exec'), g)
sys.argv = saved_argv

model = g['ContextNQS']()
model.load_state_dict(ck['model'])
ctx = model.m0.detach().clone()
hx, hz = 1.27, 0.043
v, d = g['sr_mdot'](model, ctx, hx, hz)
psi = g['model_state'](model, ctx).astype(np.complex128)

ok = True
print('--- Infinitesimal direction check ---')
for eps in [1e-3, 3e-4, 1e-4]:
    target = g['evolve_step'](psi, hx, hz, dt=eps).astype(np.complex128)
    p0 = abs(np.vdot(target, psi))**2
    pp = abs(np.vdot(target, g['model_state'](model, ctx + eps*v)))**2
    pm = abs(np.vdot(target, g['model_state'](model, ctx - eps*v)))**2
    print('eps', eps, 'F0', p0, 'Fplus', pp, 'Fminus', pm)
    
    if eps >= 3e-4:
        ok &= (pp >= p0 - 2e-7) and (pp >= pm - 2e-7)

print('--- Control-affine check ---')
v0, _ = g['sr_mdot'](model, ctx, 0.0, 0.0)
vx1, _ = g['sr_mdot'](model, ctx, 1.0, 0.0)
vz1, _ = g['sr_mdot'](model, ctx, 0.0, 1.0)
vaff = v0 + hx*(vx1-v0) + hz*(vz1-v0)
rel = float(torch.linalg.vector_norm(v-vaff) / torch.linalg.vector_norm(v))
print('control_affine_relative_error', rel)
ok &= rel < 5e-5

print('--- Finite-difference tangent coverage check ---')
H = g['H_sparse'](hx, hz)
Hpsi = H @ psi
E = np.vdot(psi, Hpsi)
b = -1j*(Hpsi-E*psi)
last_err = None
for eps in [1e-3, 3e-4, 1e-4]:
    psip = g['model_state'](model, ctx + eps*v).astype(np.complex128)
    dpsi = (psip-psi)/eps
    dpsi = dpsi-psi*np.vdot(psi,dpsi)
    cov = 1-np.linalg.norm(dpsi-b)**2/np.linalg.norm(b)**2
    err = abs(cov - d['coverage'])
    print('eps', eps, 'finite_diff_coverage', cov, 'analytic_regularized_coverage', d['coverage'], 'error', err)
    last_err = err
ok &= last_err is not None and last_err < 5e-3

if not ok:
    raise SystemExit('VALIDATION FAILED')
print('ALL CHECKPOINT-LEVEL VALIDATIONS PASSED')
