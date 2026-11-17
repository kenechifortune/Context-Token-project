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
p.add_argument('--context-batch-size', type=int, default=0,
               help='Microbatch the training contexts to bound exact-state memory; 0 uses one full batch.')
p.add_argument('--train-sample-count', type=int, default=0,
               help='Uniformly sample this many basis states per fit update; 0 uses the full basis.')
p.add_argument('--sr-sample-count', type=int, default=0,
               help='Uniformly sample this many basis states for SR updates; 0 uses the full basis.')
p.add_argument('--skip-rollout', action='store_true',
               help='Save training-fit results without running SR rollouts.')
p.add_argument('--fit-mode', choices=['trajectory','anchor','neighborhood_walk'], default='trajectory',
               help='trajectory fits time-dependent targets; anchor fits the initial state; neighborhood_walk fits a bounded context path around M0.')
p.add_argument('--fit-t-max', type=float, default=0.05,
               help='Pretraining horizon, capped at --t-max; independent of the SR rollout horizon.')
p.add_argument('--training-target-method', choices=['first_order_initial','exact_trajectory'],
               default='first_order_initial',
               help='Use first-order targets at the initial Hamiltonian or exact time-dependent trajectories.')
p.add_argument('--ctx-local-weight', type=float, default=1e-2,
               help='Weight of mean((M-M0)^2) during neighborhood_walk fitting.')
p.add_argument('--neighborhood-radius', type=float, default=0.5,
               help='Maximum Frobenius norm ||M-M0|| for each context in neighborhood_walk fitting.')
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
p.add_argument(
    "--hard-initial-anchor",
    action="store_true",
    help="Force psi(M0) to equal the uniform initial state",
)
p.add_argument('--print-every', type=int, default=100)
args = p.parse_args()
if args.t_steps < 2:
    p.error('--t-steps must be at least 2.')
if not math.isfinite(args.t_max) or args.t_max <= 0:
    p.error('--t-max must be finite and positive.')
if not math.isfinite(args.fit_t_max) or args.fit_t_max <= 0:
    p.error('--fit-t-max must be finite and positive.')
if (args.training_target_method == 'exact_trajectory'
        and min(args.fit_t_max,args.t_max) < args.t_max-1e-12):
    p.error('--training-target-method exact_trajectory requires --fit-t-max to cover --t-max.')
if not math.isfinite(args.ctx_local_weight) or args.ctx_local_weight < 0:
    p.error('--ctx-local-weight must be finite and nonnegative.')
if not math.isfinite(args.neighborhood_radius) or args.neighborhood_radius <= 0:
    p.error('--neighborhood-radius must be finite and positive.')
if args.context_batch_size < 0:
    p.error('--context-batch-size must be nonnegative.')
if args.train_sample_count < 0:
    p.error('--train-sample-count must be nonnegative.')
if args.sr_sample_count < 0:
    p.error('--sr-sample-count must be nonnegative.')

thread_limit=max(1,int(os.environ.get('TORCH_NUM_THREADS',min(9,os.cpu_count() or 1))))
torch.set_num_threads(thread_limit)
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


def first_order_training_states(fields, times):
    """Normalized (I - i*t*H(0))|psi0> targets, expanded about t=0.

    The derivative is evaluated once at each protocol's initial Hamiltonian.
    Each time is measured from the same initial state; no exact propagation
    or repeated Euler steps are used. For smooth time-dependent fields this
    is the first-order Taylor expansion of the Schrodinger solution.
    """
    times = np.asarray(times, dtype=np.float64)
    psi0 = initial_state()
    all_states = []
    for f in fields:
        hpsi0 = H_sparse(float(f[0, 0]), float(f[0, 1])) @ psi0
        states = psi0[None, :] - 1j * times[:, None] * hpsi0[None, :]
        states /= np.linalg.norm(states, axis=1, keepdims=True)
        all_states.append(states)
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
ALL_BITS_T=torch.tensor(ALL_BITS,dtype=torch.long)

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
    def _raw_heads(self, bits, ctx):
        B = bits.shape[0]
        d = args.d_model

        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(0).expand(B, -1, -1)

        start = self.start.expand(B, -1, -1) + self.pos[0:1].unsqueeze(0)

        if N > 1:
            prev = self.spin(bits[:, :-1]) + self.pos[1:].unsqueeze(0)
            x = torch.cat([start, prev], dim=1)
        else:
            x = start

        mask = torch.triu(
            torch.ones(
                N,
                N,
                dtype=torch.bool,
                device=bits.device,
            ),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, ctx, mask)

        h = self.lnf(x)

        raw_logits = self.amp(h)
        raw_phase = self.phase(h)

        return raw_logits, raw_phase


    def forward_logpsi(self, bits, ctx):
        B = bits.shape[0]

        raw_logits, raw_phase = self._raw_heads(bits, ctx)

        if args.hard_initial_anchor:
            m0 = self.m0.to(
                device=bits.device,
                dtype=raw_logits.dtype,
            )
            m0 = m0.unsqueeze(0).expand(B, -1, -1)

            initial_logits, initial_phase = self._raw_heads(bits, m0)

            # Force both outputs to vanish at ctx = M0.
            raw_logits = raw_logits - initial_logits
            raw_phase = raw_phase - initial_phase

        logp = torch.log_softmax(raw_logits, dim=-1)

        one_hot = torch.nn.functional.one_hot(
            bits,
            num_classes=2,
        ).to(logp.dtype)

        logamp = 0.5 * torch.sum(
            logp * one_hot,
            dim=(-1, -2),
        )

        phase = torch.sum(
            raw_phase * one_hot,
            dim=(-1, -2),
        )

        return torch.complex(logamp, phase)

# ----------------------------
# Training utilities
# ----------------------------
def state_batch(model,ctx,bits=None,normalize=True):
    # ctx: (Q,C,D), return (Q,number_of_basis_states)
    Q=ctx.shape[0]
    bits=ALL_BITS_T if bits is None else bits
    nbits=bits.shape[0]
    bits_big=bits.repeat(Q,1)
    ctx_big=ctx[:,None,:,:].expand(Q,nbits,args.ctx_tokens,args.d_model).reshape(Q*nbits,args.ctx_tokens,args.d_model)
    lp=model.forward_logpsi(bits_big,ctx_big).reshape(Q,nbits)
    psi=torch.exp(lp)
    # AR amplitude is normalized in theory; normalize to remove tiny roundoff/model bugs.
    if normalize:
        psi=psi/torch.linalg.vector_norm(psi,dim=1,keepdim=True)
    return psi


def state_batch_full(model,ctx):
    """Evaluate and normalize states over the full basis in bounded chunks."""
    chunk_size=64 if args.Ly>=4 and args.d_model>=32 else 128
    parts=[
        state_batch(model,ctx,bits=ALL_BITS_T[start:start+chunk_size],normalize=False)
        for start in range(0,DIM,chunk_size)
    ]
    psi=torch.cat(parts,dim=1)
    return psi/torch.linalg.vector_norm(psi,dim=1,keepdim=True)


def fit_anchor_only(model):
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    target=torch.tensor(initial_state(),dtype=CDTYPE)
    chunk_size=64 if args.Ly>=4 and args.d_model>=32 else 128
    hist=[]
    best=(1e9,None)
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        overlap=torch.zeros((),dtype=CDTYPE)
        norm2=torch.zeros((),dtype=DTYPE)
        with torch.no_grad():
            for start in range(0,DIM,chunk_size):
                end=min(start+chunk_size,DIM)
                bits=ALL_BITS_T[start:end]
                psi=torch.exp(model.forward_logpsi(bits,model.m0))
                overlap+=torch.sum(torch.conj(target[start:end])*psi)
                norm2+=torch.sum(torch.abs(psi)**2)
        fid=torch.abs(overlap)**2/norm2.clamp_min(torch.finfo(DTYPE).tiny)
        # Accumulate the exact gradient of 1-|<target|psi>|^2/||psi||^2
        # over basis chunks, avoiding a full-Hilbert autograd graph.
        for start in range(0,DIM,chunk_size):
            end=min(start+chunk_size,DIM)
            bits=ALL_BITS_T[start:end]
            psi=torch.exp(model.forward_logpsi(bits,model.m0))
            local_overlap=torch.sum(torch.conj(target[start:end])*psi)
            local_norm2=torch.sum(torch.abs(psi)**2)
            grad_piece=(
                -2*torch.real(torch.conj(overlap)*local_overlap)/norm2
                +torch.abs(overlap)**2*local_norm2/(norm2**2)
            )
            grad_piece.backward()
        loss=1-fid
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        lf=float(loss.detach()); hist.append(lf)
        if lf<best[0]: best=(lf,{k:v.detach().clone() for k,v in model.state_dict().items()})
        if step%args.print_every==0 or step==args.steps-1:
            print(f'anchor {step:5d} loss={lf:.4e} F={float(fid.detach()):.8f}')
    model.load_state_dict(best[1])
    contexts=model.m0.detach()[None,None].expand(args.train_protocols,args.t_steps,-1,-1).clone()
    return contexts,np.array(hist)

def project_context_neighborhood(contexts,center,radius):
    """Project each (..., C, D) context into ||M-center||_F <= radius."""
    delta=contexts-center
    distance=torch.linalg.vector_norm(delta,dim=(-2,-1),keepdim=True)
    scale=(radius/distance.clamp_min(torch.finfo(contexts.dtype).tiny)).clamp(max=1.0)
    return center+delta*scale


def fit_model(model,train_states,neighborhood=False):
    P=args.train_protocols; T=args.t_steps
    # Initialize a small smooth random walk in latent space around M0.
    ctx0=model.m0.detach().clone()
    increments=torch.randn(P,T-1,args.ctx_tokens,args.d_model,dtype=DTYPE)*0.01
    path=torch.cat([ctx0[None,None].expand(P,1,-1,-1), ctx0[None,None]+torch.cumsum(increments,dim=1)],dim=1)
    if neighborhood:
        path=project_context_neighborhood(path,ctx0,args.neighborhood_radius)
    contexts=nn.Parameter(path)
    opt=torch.optim.Adam(list(model.parameters())+[contexts],lr=args.lr)
    target=torch.tensor(train_states,dtype=CDTYPE)
    Q=P*T
    context_batch_size=args.context_batch_size or Q
    context_batch_size=min(context_batch_size,Q)
    target_flat=target.reshape(Q,DIM)
    sample_count=min(args.train_sample_count,DIM) if args.train_sample_count else DIM
    sample_scale=DIM/sample_count
    best=(1e9,None,None)
    hist=[]
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        if sample_count<DIM:
            sample_indices=np.random.choice(DIM,size=sample_count,replace=False)
            sample_index_tensor=torch.as_tensor(sample_indices,dtype=torch.long)
            bits=ALL_BITS_T[sample_index_tensor]
        else:
            sample_index_tensor=None
            bits=ALL_BITS_T
        fid_chunks=[]
        fit_loss_value=0.0
        for start in range(0,Q,context_batch_size):
            end=min(start+context_batch_size,Q)
            context_chunk=contexts.reshape(Q,args.ctx_tokens,args.d_model)[start:end]
            target_chunk=target_flat[start:end]
            if sample_index_tensor is not None:
                target_chunk=target_chunk[:,sample_index_tensor]
            psi=state_batch(model,context_chunk,bits=bits,normalize=(sample_index_tensor is None))
            ov=sample_scale*torch.sum(torch.conj(target_chunk)*psi,dim=-1)
            fids=torch.abs(ov)**2
            chunk_loss=torch.mean(1-fids)*((end-start)/Q)
            chunk_loss.backward()
            fit_loss_value+=float(chunk_loss.detach())
            fid_chunks.append(fids.detach())
        fids=torch.cat(fid_chunks)
        smooth=torch.mean((contexts[:,1:]-contexts[:,:-1])**2)
        l2=torch.mean(contexts**2)
        loss=args.smooth_weight*smooth+args.ctx_l2_weight*l2
        if neighborhood:
            local=torch.mean((contexts-ctx0)**2)
            loss=loss+args.ctx_local_weight*local
        lf=fit_loss_value+float(loss.detach()); hist.append(lf)
        # Keep the model and contexts that actually produced the measured loss.
        if lf<best[0]: best=(lf,{k:v.detach().clone() for k,v in model.state_dict().items()},contexts.detach().clone())
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters())+[contexts],1.0); opt.step()
        with torch.no_grad():
            if neighborhood:
                contexts.copy_(project_context_neighborhood(contexts,ctx0,args.neighborhood_radius))
            contexts[:,0].copy_(model.m0)
        if step%args.print_every==0 or step==args.steps-1:
            label='neighborhood_walk' if neighborhood else 'fit'
            local_info=f' local={float(local.detach()):.3e}' if neighborhood else ''
            print(f'{label} {step:5d} loss={lf:.4e} meanF={float(fids.mean()):.6f} worstF={float(fids.min()):.6f} smooth={float(smooth):.3e}{local_info}')
    model.load_state_dict(best[1]); return best[2],np.array(hist)


def fit_neighborhood_walk(model,train_states):
    """Fit training targets using a local penalty and a bounded context path."""
    return fit_model(model,train_states,neighborhood=True)

# ----------------------------
# Exact SR/TDVP in context coordinates
# ----------------------------
def model_state(model,ctx):
    with torch.no_grad():
        return state_batch_full(model,ctx[None])[0].cpu().numpy()

def logpsi_jacobian(model,ctx,bits=None):
    # Returns logpsi and O for the supplied configurations (full basis by default).
    bits=ALL_BITS_T if bits is None else bits
    state_count=bits.shape[0]
    x=ctx.detach().clone().requires_grad_(True)
    def fun(flat):
        c=flat.reshape(args.ctx_tokens,args.d_model)
        lp=model.forward_logpsi(bits,c)
        return torch.cat([lp.real,lp.imag],dim=0)
    flat=x.reshape(-1)
    # The context dimension is much smaller than the Hilbert-space output.
    vals=fun(flat)
    # Chunk the VJP batch for both sampled and full-basis SR updates. This
    # keeps peak allocator use bounded on CPU-only machines.
    if args.Ly >= 4 and args.d_model >= 24:
        jac_chunk_size=1
    elif args.d_model >= 40:
        jac_chunk_size=2
    else:
        jac_chunk_size=8
    J=torch.func.jacrev(fun,chunk_size=jac_chunk_size)(flat)
    lp=torch.complex(vals[:state_count],vals[state_count:])
    O=torch.complex(J[:state_count],J[state_count:])
    return lp.detach().cpu().numpy(),O.detach().cpu().numpy()

def sr_mdot(model,ctx,hx,hz,reg=None):
    sample_count=min(args.sr_sample_count,DIM) if args.sr_sample_count else DIM
    sampled=sample_count<DIM
    if sampled:
        sample_indices=np.random.choice(DIM,size=sample_count,replace=False)
        sample_tensor=torch.as_tensor(sample_indices,dtype=torch.long)
        sample_bits=ALL_BITS_T[sample_tensor]
        lp,O=logpsi_jacobian(model,ctx,sample_bits)
        psi=np.exp(lp)
        diagonal=-JZZ*ZZ[sample_indices]-hz*ZSUM[sample_indices]
        neighbor_indices=sample_indices[:,None]^(1<<np.arange(N,dtype=np.int64)[None,:])
        neighbor_bits=ALL_BITS_T[torch.as_tensor(neighbor_indices.reshape(-1),dtype=torch.long)]
        with torch.no_grad():
            neighbor_lp=model.forward_logpsi(neighbor_bits,ctx).reshape(sample_count,N)
        ratios=np.exp(neighbor_lp.cpu().numpy()-lp[:,None])
        El=diagonal-hx*ratios.sum(axis=1)
    else:
        lp,O=logpsi_jacobian(model,ctx)
        psi=np.exp(lp); psi=psi/np.linalg.norm(psi)
        H=H_sparse(hx,hz); Hpsi=H@psi
        denom=np.where(np.abs(psi)>1e-300,psi,1e-300+0j)
        El=Hpsi/denom
    w=np.abs(psi)**2; w=w/w.sum()
    Om=(w[:,None]*O).sum(axis=0); Em=(w*El).sum()
    Ot=O-Om; Et=El-Em
    S=np.real(Ot.conj().T@(w[:,None]*Ot))
    F=np.imag(Ot.conj().T@(w*Et))
    scale=np.trace(S)/max(S.shape[0],1); lam=(args.sr_reg if reg is None else reg)*max(scale,1e-14)
    # SVD-based stable solve: preserve diagnostics and suppress numerical null modes.
    ev,U=np.linalg.eigh((S+S.T)/2)
    cutoff=args.sr_rcond*max(ev.max(),1e-30)
    inv=np.where(ev>cutoff,1/(ev+lam),0.0)
    v=U@(inv*(U.T@F))
    # tangent residual with unregularized S
    varH=float(np.real((w*np.abs(Et)**2).sum()))
    res=max(varH-2*float(F@v)+float(v@S@v),0.0)
    coverage=1-res/max(varH,1e-30)
    rank=int(np.sum(ev>cutoff))
    # unregularized optimal coverage using pseudoinverse, useful as expressivity metric independent of lambda
    pinv=np.where(ev>cutoff,1/ev,0.0)
    vp=U@(pinv*(U.T@F))
    res_opt=max(varH-2*float(F@vp)+float(vp@S@vp),0.0)
    cov_opt=1-res_opt/max(varH,1e-30)
    return torch.tensor(v.reshape(args.ctx_tokens,args.d_model),dtype=DTYPE),dict(
        coverage=coverage,coverage_opt=cov_opt,rank=rank,varH=varH,lam=lam,
        cond=float(ev.max()/max(ev[ev>cutoff].min(),1e-30)) if rank else np.inf,
        mdot_rms=float(np.sqrt(np.mean(v*v))),energy=float(np.real(Em)))

def site_probabilities(psi):
    """Return marginal probabilities P(up), P(down) at each site."""
    psi=np.asarray(psi,dtype=np.complex128)
    psi=psi/np.linalg.norm(psi)
    probs=np.abs(psi)**2
    out=np.zeros((N,2),dtype=float)
    for i in range(N):
        bits=(idx_np>>i)&1
        out[i,0]=float(probs[bits==0].sum())
        out[i,1]=float(probs[bits==1].sum())
    return out

def save_probability_plots(result, exact_states, q):
    """Save 2D heatmaps and 3D surfaces of site probabilities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels=[r'$P_i(\uparrow)$',r'$P_i(\downarrow)$']
    var=result['site_probs']
    exact=np.stack([site_probabilities(psi) for psi in exact_states])
    times=np.linspace(0,args.t_max,args.t_steps)
    sites=np.arange(N)
    site_labels=[f'({i % args.Lx},{i // args.Lx})' for i in sites]
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,sharey=True)
    for k,label in enumerate(labels):
        for col,(data,title) in enumerate(((var,'Variational'),(exact,'Exact'))):
            im=axes[k,col].imshow(data[:,:,k].T,aspect='auto',origin='lower',
                                  extent=[times[0],times[-1],0,N-1],cmap='viridis',vmin=0,vmax=1)
            axes[k,col].set_title(f'{title} {label}')
            axes[k,col].set_xlabel('Time $t$'); axes[k,col].set_ylabel('Site $(x,y)$')
            axes[k,col].set_xticks(times); axes[k,col].set_yticks(sites)
            axes[k,col].set_yticklabels(site_labels)
            fig.colorbar(im,ax=axes[k,col],shrink=.85)
    fig.suptitle(f'Site probabilities, test protocol {q}',y=.995)
    fig.tight_layout()
    path2d=PLOT_DIR/f'site_probabilities_test_{q}.png'
    fig.savefig(path2d,dpi=180); plt.close(fig)

    time_grid,site_grid=np.meshgrid(times,sites,indexing='ij')
    fig=plt.figure(figsize=(14,8))
    for k,label in enumerate(labels):
        ax=fig.add_subplot(2,2,k+1,projection='3d')
        ax.plot_surface(time_grid,site_grid,var[:,:,k],cmap='viridis',vmin=0,vmax=1,linewidth=0)
        ax.set_title(f'Variational {label}'); ax.set_xlabel('Time $t$'); ax.set_ylabel('Site $i$'); ax.set_zlabel('Probability'); ax.set_zlim(0,1)
        ax=fig.add_subplot(2,2,k+3,projection='3d')
        ax.plot_surface(time_grid,site_grid,exact[:,:,k],cmap='viridis',vmin=0,vmax=1,linewidth=0)
        ax.set_title(f'Exact {label}'); ax.set_xlabel('Time $t$'); ax.set_ylabel('Site $i$'); ax.set_zlabel('Probability'); ax.set_zlim(0,1)
    fig.suptitle(f'3D site probabilities, test protocol {q}'); fig.tight_layout()
    path3d=PLOT_DIR/f'site_probabilities_3d_test_{q}.png'
    fig.savefig(path3d,dpi=180); plt.close(fig)
    print(f'Saved 2D plot: {path2d.resolve()}')
    print(f'Saved 3D plot: {path3d.resolve()}')

def rollout(model,fields,exact_states):
    ctx=model.m0.detach().clone(); Ms=[ctx.cpu().numpy()]; mdots=[]; diags=[]; gf=[]; localF=[]
    psi=model_state(model,ctx); gf.append(abs(np.vdot(exact_states[0],psi))**2)
    site_probs=[site_probabilities(psi)]
    for t in range(args.t_steps-1):
        hx=.5*(fields[t,0]+fields[t+1,0]); hz=.5*(fields[t,1]+fields[t+1,1])
        if args.integrator=='euler':
            v,d=sr_mdot(model,ctx,hx,hz); nextctx=ctx+DT*v
        else:
            v1,d=sr_mdot(model,ctx,hx,hz)
            pred=ctx+DT*v1
            # Hamiltonian at same midpoint; this is explicit Heun for autonomous-in-step tangent field.
            v2,d2=sr_mdot(model,pred,hx,hz)
            nextctx=ctx+.5*DT*(v1+v2)
            # average geometric diagnostics at endpoints
            for k in ['coverage','coverage_opt','mdot_rms']:
                d[k]=.5*(d[k]+d2[k])
            d['rank']=min(d['rank'],d2['rank']); d['cond']=max(d['cond'],d2['cond'])
            v=.5*(v1+v2)
        # Reuse the previous step's variational state instead of evaluating it
        # again; psi is exactly the model state for the current context.
        psi_t=psi; target=evolve_step(psi_t,hx,hz)
        psi_next=model_state(model,nextctx); localF.append(abs(np.vdot(target,psi_next))**2)
        ctx=nextctx.detach(); Ms.append(ctx.cpu().numpy()); mdots.append(v.detach().cpu().numpy()); diags.append(d)
        psi=psi_next; gf.append(abs(np.vdot(exact_states[t+1],psi))**2)
        site_probs.append(site_probabilities(psi))
    return dict(M=np.array(Ms),Mdot=np.array(mdots),global_fid=np.array(gf),local_fid=np.array(localF),
                site_probs=np.stack(site_probs),diags=diags)

# ----------------------------
# Main
# ----------------------------
print(f'N={N}, DIM={DIM}, context P={args.ctx_tokens*args.d_model}, dtype={args.dtype}')
train_fields=gen_fields(args.field_seed,args.train_protocols)
train_times=np.linspace(0.0,min(args.fit_t_max,args.t_max),args.t_steps)
training_target_method=args.training_target_method
test_fields=(gen_fields(args.test_field_seed,args.test_protocols) if args.test_family=='random' else (gen_ood_fields(args.test_protocols) if args.test_family=='ood' else gen_stress_fields(args.test_protocols)))
test_states=trajectories(test_fields) if not args.skip_rollout else np.empty((0,))
model=ContextNQS()
if args.load_model:
    ck=torch.load(args.load_model,map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); contexts=ck.get('contexts')
    hist=np.array([])
    # Retain a checkpoint's training provenance; never relabel an older exact-trained model.
    training_target_method=ck.get('training_target_method','legacy_unspecified')
    train_fields=np.asarray(ck.get('train_fields',train_fields))
    train_times=np.asarray(ck.get('train_times',[]),dtype=np.float64)
    train_fid=np.asarray(ck.get('train_fid',[]))
else:
    st=time.time();
    if args.fit_mode in ('trajectory','neighborhood_walk'):
        if args.training_target_method == 'exact_trajectory':
            train_states=trajectories(train_fields)
            print(f'Pretraining targets: exact time-dependent trajectories, 0 <= t <= {args.t_max:.6g}')
        else:
            train_states=first_order_training_states(train_fields,train_times)
            print(f'Pretraining targets: normalized first-order expansion, 0 <= t <= {train_times[-1]:.6g}')
        if args.fit_mode=='neighborhood_walk':
            print(f'Context neighborhood: radius={args.neighborhood_radius:.6g}, local weight={args.ctx_local_weight:.6g}')
            contexts,hist=fit_neighborhood_walk(model,train_states)
        else:
            contexts,hist=fit_model(model,train_states)
    else:
        training_target_method='anchor'
        train_times=np.zeros(args.t_steps)
        train_states=np.broadcast_to(initial_state(),(args.train_protocols,args.t_steps,DIM))
        contexts,hist=fit_anchor_only(model)
    print('fit seconds',time.time()-st)
    with torch.no_grad():
        contexts_flat=contexts.reshape(args.train_protocols*args.t_steps,args.ctx_tokens,args.d_model)
        eval_batch_size=args.context_batch_size or contexts_flat.shape[0]
        psi_parts=[
            state_batch_full(model,contexts_flat[start:start+eval_batch_size])
            for start in range(0,contexts_flat.shape[0],eval_batch_size)
        ]
        psi=torch.cat(psi_parts,dim=0).reshape(args.train_protocols,args.t_steps,DIM)
        tar=torch.tensor(train_states,dtype=CDTYPE); train_fid=(torch.abs(torch.sum(torch.conj(tar)*psi,dim=-1))**2).cpu().numpy()

# Cached training diagnostics also allow checkpoint-only rollouts on a different time grid.
oracle_mean=float(train_fid.mean()) if train_fid.size else None
oracle_worst=float(train_fid.min()) if train_fid.size else None
print('training target method',training_target_method)
print('training target fidelity mean/worst',oracle_mean,oracle_worst)
if args.save_model:
    torch.save({'model':model.state_dict(),'contexts':contexts,'args':vars(args),
                'training_target_method':training_target_method,'train_times':train_times,
                'train_fields':train_fields,'train_fid':train_fid},OUT/'model.pt')

if args.skip_rollout:
    summary=dict(config=vars(args),training_target_method=training_target_method,
                 training_times=train_times.tolist(),oracle_mean=oracle_mean,
                 oracle_worst=oracle_worst,tests=[],avg_test_mean_global=None,
                 avg_test_final_global=None,avg_coverage_opt=None)
    with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
    np.savez_compressed(
        OUT/'results.npz',
        train_fields=train_fields,
        train_times=train_times,
        training_target_method=np.asarray(training_target_method),
        train_fid=train_fid,
        fit_hist=hist,
    )
    print('Rollouts skipped; training-fit outputs saved.')
    raise SystemExit(0)

# Test SR rollouts.
rolls=[]
for q in range(args.test_protocols):
    print('rollout test',q)
    r=rollout(model,test_fields[q],test_states[q]); rolls.append(r)
    save_probability_plots(r,test_states[q],q)
    cov=np.array([d['coverage_opt'] for d in r['diags']]); rk=np.array([d['rank'] for d in r['diags']])
    print('  mean/final global F',r['global_fid'].mean(),r['global_fid'][-1],'mean local F',r['local_fid'].mean(),'cov opt',cov.mean(),cov.min(),'rank med',np.median(rk))

summ=[]
for q,r in enumerate(rolls):
    cov=np.array([d['coverage'] for d in r['diags']]); covo=np.array([d['coverage_opt'] for d in r['diags']]); rk=np.array([d['rank'] for d in r['diags']])
    summ.append(dict(test=q,mean_global_fid=float(r['global_fid'].mean()),final_global_fid=float(r['global_fid'][-1]),
                     mean_local_fid=float(r['local_fid'].mean()),worst_local_fid=float(r['local_fid'].min()),
                     mean_coverage=float(cov.mean()),mean_coverage_opt=float(covo.mean()),worst_coverage_opt=float(covo.min()),
                     median_rank=float(np.median(rk)),max_rank=int(rk.max())))
summary=dict(config=vars(args),training_target_method=training_target_method,
             training_times=train_times.tolist(),oracle_mean=oracle_mean,oracle_worst=oracle_worst,tests=summ,
             avg_test_mean_global=float(np.mean([s['mean_global_fid'] for s in summ])),avg_test_final_global=float(np.mean([s['final_global_fid'] for s in summ])),
             avg_coverage_opt=float(np.mean([s['mean_coverage_opt'] for s in summ])))
print(json.dumps(summary,indent=2))
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
save_data = {
    "train_fields": train_fields,
    "train_times": train_times,
    "training_target_method": np.asarray(training_target_method),
    "test_fields": test_fields,
    "train_fid": train_fid,
    "fit_hist": hist,
}

for i, r in enumerate(rolls):
    save_data[f"gfid_{i}"] = r["global_fid"]
    save_data[f"lfid_{i}"] = r["local_fid"]
    save_data[f"M_{i}"] = r["M"]
    save_data[f"Mdot_{i}"] = r["Mdot"]
    save_data[f"site_probs_{i}"] = r["site_probs"]
    save_data[f"coverage_{i}"] = np.array(
        [d["coverage_opt"] for d in r["diags"]]
    )
    save_data[f"rank_{i}"] = np.array(
        [d["rank"] for d in r["diags"]]
    )

np.savez_compressed(OUT / "results.npz", **save_data)
