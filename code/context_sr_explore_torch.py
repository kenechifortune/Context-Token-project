import argparse, json, math, os, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
import torch
from torch import nn

# ----------------------------
# CLI
# ----------------------------
p = argparse.ArgumentParser()
p.add_argument('--Lx', type=int, default=3)
p.add_argument('--Ly', type=int, default=2)
p.add_argument('--t-steps', type=int, default=21)
p.add_argument('--t-max', type=float, default=0.5)
p.add_argument('--d-model', type=int, default=16)
p.add_argument('--num-heads', type=int, default=4)
p.add_argument('--num-layers', type=int, default=2)
p.add_argument('--ctx-tokens', type=int, default=2)
p.add_argument('--train-protocols', type=int, default=4)
p.add_argument('--test-protocols', type=int, default=3)
p.add_argument('--steps', type=int, default=1500)
p.add_argument('--fit-mode', choices=['trajectory','anchor'], default='trajectory')
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--smooth-weight', type=float, default=1e-3)
p.add_argument('--ctx-l2-weight', type=float, default=0.0)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--n-modes', type=int, default=3)
p.add_argument('--field-seed', type=int, default=1234)
p.add_argument('--test-field-seed', type=int, default=4321)
p.add_argument('--test-family', choices=['random','ood','stress'], default='random')
p.add_argument('--sr-reg', type=float, default=1e-5)
p.add_argument('--sr-rcond', type=float, default=1e-8)
p.add_argument('--integrator', choices=['euler','heun'], default='heun')
p.add_argument('--out', type=str, default='/mnt/data/context_sr_run')
p.add_argument('--save-model', action='store_true')
p.add_argument('--load-model', type=str, default=None)
p.add_argument('--dtype', choices=['float32','float64'], default='float64')
p.add_argument('--print-every', type=int, default=100)
p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
args = p.parse_args()

if args.device == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA was requested, but no GPU is available.')
device = torch.device(
    'cuda' if args.device == 'cuda'
    or (args.device == 'auto' and torch.cuda.is_available())
    else 'cpu'
)

torch.set_num_threads(min(9, os.cpu_count() or 1))
torch.manual_seed(args.seed)
np.random.seed(args.seed)
DT = args.t_max/(args.t_steps-1)
N = args.Lx*args.Ly
DIM = 2**N
JZZ = 1.0
DTYPE = torch.float64 if args.dtype=='float64' else torch.float32
CDTYPE = torch.complex128 if DTYPE==torch.float64 else torch.complex64
OUT = Path(args.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUT / 'plots'; PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Output directory: {OUT}')
print(f'Plot directory:    {PLOT_DIR}')

# ----------------------------
# Lattice and Hamiltonian
# ----------------------------
def bonds(Lx,Ly):
    out=[]
    def idx(x,y): return y*Lx+x
    for y in range(Ly):
        for x in range(Lx):
            if x+1<Lx: out.append((idx(x,y),idx(x+1,y)))
            if y+1<Ly: out.append((idx(x,y),idx(x,y+1)))
    return out
BONDS=bonds(args.Lx,args.Ly)
idx_np=np.arange(DIM,dtype=np.int64)
ZZ=np.zeros(DIM); ZSUM=np.zeros(DIM)
for i,j in BONDS:
    zi=1-2*((idx_np>>i)&1); zj=1-2*((idx_np>>j)&1); ZZ += zi*zj
for i in range(N): ZSUM += 1-2*((idx_np>>i)&1)
rows=[]; cols=[]
for i in range(N):
    rows.append(idx_np); cols.append(idx_np^(1<<i))
rows=np.concatenate(rows); cols=np.concatenate(cols)
XSUM=sp.coo_matrix((np.ones_like(rows,dtype=float),(rows,cols)),shape=(DIM,DIM)).tocsr()

def H_sparse(hx,hz):
    return sp.diags(-JZZ*ZZ-hz*ZSUM,format='csr') - hx*XSUM

def evolve_step(psi,hx,hz,dt=DT):
    out=expm_multiply((-1j*dt)*H_sparse(float(hx),float(hz)),np.asarray(psi,np.complex128))
    return out/np.linalg.norm(out)

def initial_state():
    return np.ones(DIM,dtype=np.complex128)/math.sqrt(DIM)

def trajectories(fields):
    all_states=[]
    for f in fields:
        psi=initial_state(); states=[psi.copy()]
        for t in range(args.t_steps-1):
            hx=.5*(f[t,0]+f[t+1,0]); hz=.5*(f[t,1]+f[t+1,1])
            psi=evolve_step(psi,hx,hz); states.append(psi.copy())
        all_states.append(np.stack(states))
    return np.stack(all_states)


def gen_stress_fields(n_protocols):
    t=np.linspace(0,args.t_max,args.t_steps)
    out=[]
    amps=[0.2,0.5,0.8,1.1,1.4,1.8]
    for i in range(n_protocols):
        A=amps[i%len(amps)]
        hx=1.0 + A*np.exp(-0.5*((t-0.5*args.t_max)/(0.10*args.t_max))**2)
        hz=0.04*np.ones_like(t)
        out.append(np.stack([hx,hz],axis=-1))
    return np.stack(out)

def gen_ood_fields(n_protocols):
    t=np.linspace(0,args.t_max,args.t_steps)
    out=[]
    # Deliberately qualitatively different from the Fourier-series training family.
    candidates=[]
    # Gaussian pulse in transverse field + weak longitudinal offset.
    hx=1.0 + 0.55*np.exp(-0.5*((t-0.52*args.t_max)/(0.12*args.t_max))**2)
    hz=0.035*np.ones_like(t)
    candidates.append(np.stack([hx,hz],axis=-1))
    # Smooth tanh ramp crossing a broad range.
    hx=0.75 + 0.50*(0.5*(1+np.tanh((t-0.48*args.t_max)/(0.08*args.t_max))))
    hz=0.04*np.tanh((t-0.50*args.t_max)/(0.11*args.t_max))
    candidates.append(np.stack([hx,hz],axis=-1))
    # Chirped drive, absent from the fixed-harmonic training family.
    tau=t/max(args.t_max,1e-12)
    hx=1.0 + 0.30*np.sin(2*np.pi*(0.7*tau+2.3*tau**2))
    hz=0.06*np.sin(2*np.pi*(0.4*tau+1.4*tau**2)+0.7)
    candidates.append(np.stack([hx,hz],axis=-1))
    # Narrower Gaussian + sign-changing longitudinal pulse.
    hx=1.0 - 0.45*np.exp(-0.5*((t-0.35*args.t_max)/(0.07*args.t_max))**2)
    hz=0.08*np.exp(-0.5*((t-0.70*args.t_max)/(0.10*args.t_max))**2)-0.03
    candidates.append(np.stack([hx,hz],axis=-1))
    for i in range(n_protocols): out.append(candidates[i%len(candidates)])
    return np.stack(out)

def gen_fields(seed,n_protocols):
    rg=np.random.default_rng(seed)
    t=np.linspace(0,args.t_max,args.t_steps)
    fields=[]
    for _ in range(n_protocols):
        hx0=rg.uniform(.9,1.1); hz0=rg.uniform(-.05,.05)
        hx=np.full_like(t,hx0); hz=np.full_like(t,hz0)
        for k in range(1,args.n_modes+1):
            # Smooth, bandwidth-controlled family.
            w=2*np.pi*k/args.t_max
            hx += rg.uniform(-.35,.35)/(k**1.5)*np.sin(w*t+rg.uniform(0,2*np.pi))
            hz += rg.uniform(-.08,.08)/(k**1.5)*np.sin(w*t+rg.uniform(0,2*np.pi))
        fields.append(np.stack([hx,hz],axis=-1))
    return np.stack(fields)

# Basis bitstrings, least-significant bit = site 0.
ALL_BITS=np.stack([((idx_np>>i)&1) for i in range(N)],axis=1).astype(np.int64)
ALL_BITS_T=torch.tensor(ALL_BITS,dtype=torch.long,device=device)

# ----------------------------
# Context-conditioned autoregressive transformer
# ----------------------------
class Block(nn.Module):
    def __init__(self,d,h):
        super().__init__()
        self.ln1=nn.LayerNorm(d,dtype=DTYPE); self.ln2=nn.LayerNorm(d,dtype=DTYPE); self.ln3=nn.LayerNorm(d,dtype=DTYPE)
        self.sa=nn.MultiheadAttention(d,h,batch_first=True,dtype=DTYPE)
        self.ca=nn.MultiheadAttention(d,h,batch_first=True,dtype=DTYPE)
        self.ff=nn.Sequential(nn.Linear(d,4*d,dtype=DTYPE),nn.GELU(),nn.Linear(4*d,d,dtype=DTYPE))
    def forward(self,x,ctx,mask):
        y=self.ln1(x); y,_=self.sa(y,y,y,attn_mask=mask,need_weights=False); x=x+y
        y=self.ln2(x); y,_=self.ca(y,ctx,ctx,need_weights=False); x=x+y
        y=self.ln3(x); x=x+self.ff(y)
        return x

class ContextNQS(nn.Module):
    def __init__(self):
        super().__init__(); d=args.d_model
        self.spin=nn.Embedding(2,d,dtype=DTYPE); self.pos=nn.Parameter(torch.randn(N,d,dtype=DTYPE)*.02)
        self.start=nn.Parameter(torch.randn(1,1,d,dtype=DTYPE)*.02)
        self.blocks=nn.ModuleList([Block(d,args.num_heads) for _ in range(args.num_layers)])
        self.lnf=nn.LayerNorm(d,dtype=DTYPE); self.amp=nn.Linear(d,2,dtype=DTYPE); self.phase=nn.Linear(d,2,dtype=DTYPE)
        nn.init.normal_(self.phase.weight,std=.02); nn.init.zeros_(self.phase.bias)
        # fixed M0 with row norm sqrt(d)
        m0=torch.randn(args.ctx_tokens,d,dtype=DTYPE)
        m0=m0/torch.linalg.vector_norm(m0,dim=-1,keepdim=True)*math.sqrt(d)
        self.register_buffer('m0',m0)
    def forward_logpsi(self,bits,ctx):
        B=bits.shape[0]; d=args.d_model
        if ctx.ndim==2: ctx=ctx.unsqueeze(0).expand(B,-1,-1)
        start=self.start.expand(B,-1,-1)+self.pos[0:1].unsqueeze(0)
        if N>1:
            prev=self.spin(bits[:,:-1])+self.pos[1:].unsqueeze(0)
            x=torch.cat([start,prev],dim=1)
        else: x=start
        # True means masked for torch MHA. Mask future j>i.
        mask=torch.triu(torch.ones(N,N,dtype=torch.bool,device=bits.device),diagonal=1)
        for b in self.blocks: x=b(x,ctx,mask)
        h=self.lnf(x)
        la=self.amp(h); ph=self.phase(h)
        logp=torch.log_softmax(la,dim=-1)
        oh=torch.nn.functional.one_hot(bits,2).to(DTYPE)
        logamp=.5*torch.sum(logp*oh,dim=(-1,-2))
        phase=torch.sum(ph*oh,dim=(-1,-2))
        return torch.complex(logamp,phase)

# ----------------------------
# Training utilities
# ----------------------------
def state_batch(model,ctx):
    # ctx: (Q,C,D), return (Q,DIM)
    Q=ctx.shape[0]
    bits=ALL_BITS_T.repeat(Q,1)
    ctx_big=ctx[:,None,:,:].expand(Q,DIM,args.ctx_tokens,args.d_model).reshape(Q*DIM,args.ctx_tokens,args.d_model)
    lp=model.forward_logpsi(bits,ctx_big).reshape(Q,DIM)
    psi=torch.exp(lp)
    # AR amplitude is normalized in theory; normalize to remove tiny roundoff/model bugs.
    psi=psi/torch.linalg.vector_norm(psi,dim=1,keepdim=True)
    return psi


def fit_anchor_only(model):
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    target=torch.tensor(initial_state(),dtype=CDTYPE,device=device)
    hist=[]
    best=(1e9,None)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        psi=state_batch(model,model.m0[None])[0]
        fid=torch.abs(torch.sum(torch.conj(target)*psi))**2
        loss=1-fid
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        lf=float(loss.detach()); hist.append(lf)
        if lf<best[0]: best=(lf,{k:v.detach().clone() for k,v in model.state_dict().items()})
        if step%args.print_every==0 or step==args.steps-1:
            print(f'anchor {step:5d} loss={lf:.4e} F={float(fid.detach()):.8f}')
    model.load_state_dict(best[1])
    contexts=model.m0.detach()[None,None].expand(args.train_protocols,args.t_steps,-1,-1).clone()
    return contexts,np.array(hist)

def fit_model(model,train_states):
    P=args.train_protocols; T=args.t_steps
    # Initialize a small smooth random walk in latent space around M0.
    ctx0=model.m0.detach().clone()
    increments=torch.randn(P,T-1,args.ctx_tokens,args.d_model,dtype=DTYPE)*0.01
    path=torch.cat([ctx0[None,None].expand(P,1,-1,-1), ctx0[None,None]+torch.cumsum(increments,dim=1)],dim=1)
    contexts=nn.Parameter(path)
    opt=torch.optim.Adam(list(model.parameters())+[contexts],lr=args.lr)
    target=torch.tensor(train_states,dtype=CDTYPE,device=device)
    best=(1e9,None,None)
    hist=[]
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        psi=state_batch(model,contexts.reshape(P*T,args.ctx_tokens,args.d_model)).reshape(P,T,DIM)
        ov=torch.sum(torch.conj(target)*psi,dim=-1)
        fids=torch.abs(ov)**2
        fit_loss=torch.mean(1-fids)
        smooth=torch.mean((contexts[:,1:]-contexts[:,:-1])**2)
        l2=torch.mean(contexts**2)
        loss=fit_loss+args.smooth_weight*smooth+args.ctx_l2_weight*l2
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+[contexts],1.0); opt.step()
        with torch.no_grad(): contexts[:,0].copy_(model.m0)
        lf=float(loss.detach()); hist.append(lf)
        if lf<best[0]: best=(lf,{k:v.detach().clone() for k,v in model.state_dict().items()},contexts.detach().clone())
        if step%args.print_every==0 or step==args.steps-1:
            print(f'fit {step:5d} loss={lf:.4e} meanF={float(fids.mean()):.6f} worstF={float(fids.min()):.6f} smooth={float(smooth):.3e}')
    model.load_state_dict(best[1]); return best[2],np.array(hist)

# ----------------------------
# Exact SR/TDVP in context coordinates
# ----------------------------
def model_state(model,ctx):
    with torch.no_grad():
        return state_batch(model,ctx[None])[0].cpu().numpy()

def logpsi_jacobian(model,ctx):
    # Returns logpsi (DIM,), O (DIM,P) complex, exact.
    bits=ALL_BITS_T


# ------------------------------------------------------------
# Training only
# ------------------------------------------------------------
print(
    f"N={N}, DIM={DIM}, context P={args.ctx_tokens * args.d_model}, "
    f"dtype={args.dtype}"
)

train_fields = gen_fields(args.field_seed, args.train_protocols)
train_states = trajectories(train_fields)

model = ContextNQS().to(device)

if args.load_model:
    checkpoint = torch.load(
        args.load_model,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    contexts = checkpoint.get("contexts")
    history = np.array([])
    print(f"Loaded checkpoint: {args.load_model}")
else:
    start_time = time.time()

    if args.fit_mode == "trajectory":
        contexts, history = fit_model(model, train_states)
    else:
        contexts, history = fit_anchor_only(model)

    print("Training seconds:", time.time() - start_time)

if args.save_model:
    checkpoint = {
        "model": model.state_dict(),
        "contexts": contexts.detach().cpu(),
        "args": vars(args),
    }
    torch.save(checkpoint, OUT / "model.pt")
    print(f"Saved model checkpoint: {OUT / 'model.pt'}")

np.save(OUT / "train_fields.npy", train_fields)
np.save(OUT / "fit_history.npy", history)

print("Training complete.")
print("No rollout, observables, or plots were calculated.")

