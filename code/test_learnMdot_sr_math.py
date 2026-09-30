import ast
from pathlib import Path
import types
import numpy as np
import jax
import jax.numpy as jnp

src = (Path(__file__).resolve().parent / "learnMdot_sr_jax.py").read_text()
tree = ast.parse(src)
want = {'_sr_centered_arrays','_sr_sample_space_factors','_stable_linear_solve','_solve_sr_from_X'}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
mod = ast.Module(body=nodes, type_ignores=[])
ns = {'np': np, 'CLI_ARGS': types.SimpleNamespace(sr_rcond=1e-12)}
exec(compile(mod, '<extracted_sr>', 'exec'), ns)
center = ns['_sr_centered_arrays']
factors = ns['_sr_sample_space_factors']
solve = ns['_solve_sr_from_X']

# Test 1: synthetic push-through equivalence.
rng=np.random.default_rng(0)
for P,Ns in [(7,3),(20,5),(4,10)]:
    O=rng.normal(size=(Ns,P))+1j*rng.normal(size=(Ns,P))
    E=rng.normal(size=Ns)+1j*rng.normal(size=Ns)
    w=rng.random(Ns); w/=w.sum()
    Ot,Et,w,_=center(O,E,w)
    X,f=factors(Ot,Et,w)
    S=np.real(Ot.conj().T @ (w[:,None]*Ot))
    F=np.imag(Ot.conj().T @ (w*Et))
    assert np.allclose(S,X@X.T,rtol=1e-12,atol=1e-12)
    assert np.allclose(F,X@f,rtol=1e-12,atol=1e-12)
    lam=0.137
    vp,_,_,_=solve(X,f,'parameter',lam)
    vs,_,_,_=solve(X,f,'sample',lam)
    rel=np.linalg.norm(vp-vs)/max(np.linalg.norm(vp),1e-15)
    print('synthetic',P,Ns,'rel backend',rel)
    assert rel < 1e-10

# Test 2: JAX toy normalized complex wavefunction. Verify SR force sign against direct tangent least squares.
D=8; P=5
A=jnp.asarray(rng.normal(size=(D,P))*0.4)
B=jnp.asarray(rng.normal(size=(D,P))*0.4)
# Hermitian random H
C=rng.normal(size=(D,D))+1j*rng.normal(size=(D,D))
H=jnp.asarray((C+C.conj().T)/2)
m=jnp.asarray(rng.normal(size=P)*0.3)

def psi_fn(m):
    logits=A@m
    p=jax.nn.softmax(logits)
    phase=B@m
    return jnp.sqrt(p)*jnp.exp(1j*phase)

def logpsi_comp(m):
    psi=psi_fn(m)
    return jnp.stack([jnp.log(jnp.abs(psi)), jnp.angle(psi)], axis=-1)

psi=np.asarray(psi_fn(m),dtype=np.complex128)
# O=d log psi/dm = d log|psi| + i d phase. angle is smooth for this linear phase toy near current point.
jac=np.asarray(jax.jacrev(logpsi_comp)(m)) # D,2,P
O=jac[:,0,:]+1j*jac[:,1,:]
Hnp=np.asarray(H,dtype=np.complex128)
Hpsi=Hnp@psi
Eloc=Hpsi/psi
w=np.abs(psi)**2
Ot,Et,w,Emean=center(O,Eloc,w)
X,f=factors(Ot,Et,w)
S=np.real(Ot.conj().T@(w[:,None]*Ot))
F=np.imag(Ot.conj().T@(w*Et))
T=psi[:,None]*Ot
b=-1j*(Hpsi-Emean*psi)
Sd=np.real(T.conj().T@T)
Fd=np.real(T.conj().T@b)
print('toy metric rel',np.linalg.norm(S-Sd)/np.linalg.norm(Sd))
print('toy force rel',np.linalg.norm(F-Fd)/np.linalg.norm(Fd))
assert np.allclose(S,Sd,rtol=2e-5,atol=2e-6)
assert np.allclose(F,Fd,rtol=2e-5,atol=2e-6)

lam=1e-3*np.trace(S)/P
vp,_,_,_=solve(X,f,'parameter',lam)
vs,_,_,_=solve(X,f,'sample',lam)
print('toy backend rel',np.linalg.norm(vp-vs)/np.linalg.norm(vp))
assert np.allclose(vp,vs,rtol=1e-8,atol=1e-8)

# Test 3: finite-step direction sign: +v must beat -v for sufficiently small dt.
from scipy.linalg import expm
for dt in [1e-3,3e-4,1e-4]:
    target=np.asarray(expm(-1j*Hnp*dt)@psi)
    pp=np.asarray(psi_fn(m+jnp.asarray(dt*vp)))
    pm=np.asarray(psi_fn(m-jnp.asarray(dt*vp)))
    fp=abs(np.vdot(target,pp))**2
    fm=abs(np.vdot(target,pm))**2
    f0=abs(np.vdot(target,psi))**2
    print('dt',dt,'fplus',fp,'fzero',f0,'fminus',fm)
    assert fp + 1e-9 >= fm
print('ALL SR MATH TESTS PASSED')
