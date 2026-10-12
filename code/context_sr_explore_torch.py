    fig.suptitle(f'Energy and entanglement, test protocol {q}')
    fig.tight_layout()
    path=PLOT_DIR/f'energy_entanglement_test_{q}.png'
    fig.savefig(path,dpi=180); plt.close(fig)
    print(f'Saved energy/entanglement plot: {path.resolve()}')

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
    log_derivatives=[]; local_energies=[]; sample_weights=[]; exact_log_derivatives=[]
    psi=model_state(model,ctx); gf.append(abs(np.vdot(exact_states[0],psi))**2)
    site_probs=[site_probabilities(psi)]
    energy=[energy_expectation(psi,fields[0,0],fields[0,1])]
    exact_energy=[energy_expectation(exact_states[0],fields[0,0],fields[0,1])]
    entanglement=[entanglement_entropy(psi)]
    exact_entanglement=[entanglement_entropy(exact_states[0])]
    for t in range(args.t_steps-1):
        hx=.5*(fields[t,0]+fields[t+1,0]); hz=.5*(fields[t,1]+fields[t+1,1])
        O_t,Eloc_t,w_t,Dexact_t=log_derivative_and_eloc(model,ctx,hx,hz)
        log_derivatives.append(O_t); local_energies.append(Eloc_t)
        sample_weights.append(w_t); exact_log_derivatives.append(Dexact_t)
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
        psi_t=model_state(model,ctx); target=evolve_step(psi_t,hx,hz)
        psi_next=model_state(model,nextctx); localF.append(abs(np.vdot(target,psi_next))**2)
        ctx=nextctx.detach(); Ms.append(ctx.cpu().numpy()); mdots.append(v.detach().cpu().numpy()); diags.append(d)
        psi=model_state(model,ctx); gf.append(abs(np.vdot(exact_states[t+1],psi))**2)
        site_probs.append(site_probabilities(psi))
        energy.append(energy_expectation(psi,fields[t+1,0],fields[t+1,1]))
        exact_energy.append(energy_expectation(exact_states[t+1],fields[t+1,0],fields[t+1,1]))
        entanglement.append(entanglement_entropy(psi))
        exact_entanglement.append(entanglement_entropy(exact_states[t+1]))
    return dict(M=np.array(Ms),Mdot=np.array(mdots),global_fid=np.array(gf),local_fid=np.array(localF),
                site_probs=np.stack(site_probs),log_derivatives=np.stack(log_derivatives),
                local_energies=np.stack(local_energies),sample_weights=np.stack(sample_weights),
                exact_log_derivatives=np.stack(exact_log_derivatives),
                energy=np.array(energy),exact_energy=np.array(exact_energy),
                entanglement=np.array(entanglement),exact_entanglement=np.array(exact_entanglement),
                diags=diags)

def save_tangent_plot(result,q):
    """Save centered exact/variational tangent diagnostics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    O=result['log_derivatives']; w=result['sample_weights']
    Dexact=result['exact_log_derivatives']; md=result['Mdot'].reshape(O.shape[0],-1)
    Omean=(w[:,:,None]*O).sum(axis=1); Ot=O-Omean[:,None,:]
    Dvar=np.einsum('tsp,tp->ts',Ot,md)
    residual=Dexact-Dvar
    def norm(x): return np.sqrt(np.sum(w*np.abs(x)**2,axis=1))
    ne,nv,nr=norm(Dexact),norm(Dvar),norm(residual)
    inner=np.sum(w*np.conj(Dexact)*Dvar,axis=1)
    align=np.real(inner)/(ne*nv+1e-30)
    times=np.linspace(0,args.t_max,len(ne),endpoint=False)
    fig,ax=plt.subplots(2,2,figsize=(11,7))
    ax[0,0].plot(times,ne,label='Exact tangent'); ax[0,0].plot(times,nv,label='Variational tangent')
    ax[0,0].set_title('Centered tangent magnitudes'); ax[0,0].set_xlabel('Time $t$'); ax[0,0].set_ylabel('Weighted RMS'); ax[0,0].legend(); ax[0,0].grid(True)
    ax[0,1].plot(times,nr,color='crimson'); ax[0,1].set_title('Tangent residual'); ax[0,1].set_xlabel('Time $t$'); ax[0,1].set_ylabel(r'$\|D_{exact}-D_{var}\|$'); ax[0,1].grid(True)
    ax[1,0].plot(times,align,color='purple'); ax[1,0].set_title('Tangent directional alignment'); ax[1,0].set_xlabel('Time $t$'); ax[1,0].set_ylabel('Alignment'); ax[1,0].set_ylim(-1,1); ax[1,0].grid(True)
    ax[1,1].plot(times,np.sqrt(np.sum(w*np.abs(result['local_energies']-np.sum(w*result['local_energies'],axis=1)[:,None])**2,axis=1)),color='darkorange')
    ax[1,1].set_title('Centered local-energy RMS'); ax[1,1].set_xlabel('Time $t$'); ax[1,1].set_ylabel(r'$\sqrt{\mathrm{Var}(E_{loc})}$'); ax[1,1].grid(True)
    fig.suptitle(f'TDVP tangent diagnostics, test protocol {q}'); fig.tight_layout()
    path=PLOT_DIR/f'tangent_diagnostics_test_{q}.png'; fig.savefig(path,dpi=180); plt.close(fig)
    print(f'Saved tangent diagnostics: {path.resolve()}')

# ----------------------------
# Main
# ----------------------------
print(f'N={N}, DIM={DIM}, context P={args.ctx_tokens*args.d_model}, dtype={args.dtype}')
train_fields=gen_fields(args.field_seed,args.train_protocols); train_states=trajectories(train_fields)
test_fields=(gen_fields(args.test_field_seed,args.test_protocols) if args.test_family=='random' else (gen_ood_fields(args.test_protocols) if args.test_family=='ood' else gen_stress_fields(args.test_protocols))); test_states=trajectories(test_fields)
model=ContextNQS()
if args.load_model:
    ck=torch.load(args.load_model,map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); contexts=ck.get('contexts')
    hist=np.array([])
else:
    st=time.time();
    if args.fit_mode=='trajectory': contexts,hist=fit_model(model,train_states)
    else: contexts,hist=fit_anchor_only(model)
    print('fit seconds',time.time()-st)
if args.save_model:
    torch.save({'model':model.state_dict(),'contexts':contexts,'args':vars(args)},OUT/'model.pt')

# Oracle training fidelity.
with torch.no_grad():
    psi=state_batch(model,contexts.reshape(args.train_protocols*args.t_steps,args.ctx_tokens,args.d_model)).reshape(args.train_protocols,args.t_steps,DIM)
    tar=torch.tensor(train_states,dtype=CDTYPE); train_fid=(torch.abs(torch.sum(torch.conj(tar)*psi,dim=-1))**2).cpu().numpy()
print('oracle mean/worst',train_fid.mean(),train_fid.min())

# Test SR rollouts.
rolls=[]
for q in range(args.test_protocols):
    print('rollout test',q)
    r=rollout(model,test_fields[q],test_states[q]); rolls.append(r)
    save_probability_plots(r,test_states[q],q)
    save_tangent_plot(r,q)
    save_energy_entanglement_plot(r,q)
    cov=np.array([d['coverage_opt'] for d in r['diags']]); rk=np.array([d['rank'] for d in r['diags']])
    print('  mean/final global F',r['global_fid'].mean(),r['global_fid'][-1],'mean local F',r['local_fid'].mean(),'cov opt',cov.mean(),cov.min(),'rank med',np.median(rk))

summ=[]
for q,r in enumerate(rolls):
    cov=np.array([d['coverage'] for d in r['diags']]); covo=np.array([d['coverage_opt'] for d in r['diags']]); rk=np.array([d['rank'] for d in r['diags']])
    summ.append(dict(test=q,mean_global_fid=float(r['global_fid'].mean()),final_global_fid=float(r['global_fid'][-1]),
                     mean_local_fid=float(r['local_fid'].mean()),worst_local_fid=float(r['local_fid'].min()),
                     mean_coverage=float(cov.mean()),mean_coverage_opt=float(covo.mean()),worst_coverage_opt=float(covo.min()),
                     median_rank=float(np.median(rk)),max_rank=int(rk.max())))
summary=dict(config=vars(args),oracle_mean=float(train_fid.mean()),oracle_worst=float(train_fid.min()),tests=summ,
             avg_test_mean_global=float(np.mean([s['mean_global_fid'] for s in summ])),avg_test_final_global=float(np.mean([s['final_global_fid'] for s in summ])),
             avg_coverage_opt=float(np.mean([s['mean_coverage_opt'] for s in summ])))
print(json.dumps(summary,indent=2))
with open(OUT/'summary.json','w') as f: json.dump(summary,f,indent=2)
save_data=dict(train_fields=train_fields,test_fields=test_fields,train_fid=train_fid,fit_hist=hist)
for i,r in enumerate(rolls):
    save_data[f'gfid_{i}']=r['global_fid']
    save_data[f'lfid_{i}']=r['local_fid']
    save_data[f'M_{i}']=r['M']
    save_data[f'Mdot_{i}']=r['Mdot']
    save_data[f'site_probs_{i}']=r['site_probs']
    save_data[f'log_derivatives_{i}']=r['log_derivatives']
    save_data[f'local_energies_{i}']=r['local_energies']
    save_data[f'sample_weights_{i}']=r['sample_weights']
    save_data[f'exact_log_derivatives_{i}']=r['exact_log_derivatives']
    save_data[f'energy_{i}']=r['energy']
    save_data[f'exact_energy_{i}']=r['exact_energy']
    save_data[f'entanglement_{i}']=r['entanglement']
    save_data[f'exact_entanglement_{i}']=r['exact_entanglement']
    save_data[f'coverage_{i}']=np.array([d['coverage_opt'] for d in r['diags']])
    save_data[f'rank_{i}']=np.array([d['rank'] for d in r['diags']])
np.savez_compressed(OUT/'results.npz',**save_data)
