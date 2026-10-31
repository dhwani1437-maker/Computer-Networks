import pickle, numpy as np, json
from scipy import stats
import fogsim as f
res, sweep = pickle.load(open('results.pkl','rb'))
def arr(sc,cd,vn,k): return np.array([r[k] for r in res[(sc,cd,vn)]],float)
FOG='Fog-adaptive (proposed)'; CL='Cloud-adaptive'; FX='Fixed-time'; WB='Webster'
S={}
short={'S1-Light':'S1 (Light)','S2-Moderate':'S2 (Moderate)','S3-Heavy':'S3 (Heavy)','S4-Rush':'S4 (Rush hour)','S5-Bursty':'S5 (Bursty)'}
# T-delay
rows=[]
for sc in f.SCEN:
    m={v:(arr(sc,'Normal',v,'delay').mean(),arr(sc,'Normal',v,'delay').std(ddof=1)) for v in [FX,WB,CL,FOG]}
    gf=100*(m[FX][0]-m[FOG][0])/m[FX][0]; gw=100*(m[WB][0]-m[FOG][0])/m[WB][0]
    rows.append(f"{short[sc]} & {m[FX][0]:.1f}$\\pm${m[FX][1]:.1f} & {m[WB][0]:.1f}$\\pm${m[WB][1]:.1f} & {m[CL][0]:.1f}$\\pm${m[CL][1]:.1f} & \\textbf{{{m[FOG][0]:.1f}$\\pm${m[FOG][1]:.1f}}} & {gf:.1f} & {gw:.1f} \\\\")
S['T_DELAY']="\n".join(rows)
# bar coords
S['BAR']={v:" ".join(f"({short[sc].split()[0]},{arr(sc,'Normal',v,'delay').mean():.1f})" for sc in f.SCEN) for v in [FX,WB,CL,FOG]}
# throughput / qmax / ev per scenario normal
rows=[]
for sc in f.SCEN:
    c=[]
    for v in [FX,WB,CL,FOG]:
        c.append(f"{arr(sc,'Normal',v,'thr').mean():.0f}")
    q=[f"{arr(sc,'Normal',v,'qmax').mean():.0f}" for v in [FX,WB,CL,FOG]]
    e=[f"{np.nanmean(arr(sc,'Normal',v,'ev')):.1f}" for v in [FX,WB,CL,FOG]]
    rows.append(f"{short[sc]} & "+" & ".join(c)+" & "+" & ".join(q)+" & "+" & ".join(e)+" \\\\")
S['T_MULTI']="\n".join(rows)
# conditions
def agg(cd,v):
    d=np.mean([arr(sc,cd,v,'delay').mean() for sc in f.SCEN]); e=np.nanmean([np.nanmean(arr(sc,cd,v,'ev')) for sc in f.SCEN])
    pre=100*sum(sum(r['ev_pre'] for r in res[(sc,cd,v)]) for sc in f.SCEN)/sum(sum(r['ev_n'] for r in res[(sc,cd,v)]) for sc in f.SCEN)
    al=np.nanmean([np.nanmean(arr(sc,cd,v,'acc_lat')) for sc in f.SCEN]); ad=100*np.mean([arr(sc,cd,v,'acc_det').mean() for sc in f.SCEN])
    return d,e,pre,al,ad
rows=[]; cond_ev={}; cond_det={}
for cd in f.COND:
    for v,nm in [(CL,'Cloud'),(FOG,'Fog')]:
        d,e,pre,al,ad=agg(cd,v); cond_ev[(cd,nm)]=e; cond_det[(cd,nm)]=ad
        rows.append(f"{cd} & {nm} & {d:.1f} & {e:.2f} & {pre:.1f} & {ad:.1f} & {al:.2f} \\\\")
S['T_COND']="\n".join(rows)
S['EVBAR']={nm:" ".join(f"({cd},{cond_ev[(cd,nm)]:.2f})" for cd in f.COND) for nm in ['Cloud','Fog']}
S['DETBAR']={nm:" ".join(f"({cd},{cond_det[(cd,nm)]:.1f})" for cd in f.COND) for nm in ['Cloud','Fog']}
# latency
rows=[]
lat={}
for t,nm in [('fog','Fog'),('cloud','Cloud')]:
    for cd in ['Normal','Congested']:
        x=f.mean_latency(t,cd)*1000
        lat[(nm,cd)]=x
        rows.append(f"{nm} & {cd} & {x.mean():.0f} & {np.percentile(x,50):.0f} & {np.percentile(x,95):.0f} & {np.percentile(x,99):.0f} \\\\")
S['T_LAT']="\n".join(rows)
# sweep
rows=[]; c1=[];c2=[];c3=[]
for L,r in sweep.items():
    dl=np.mean([x['delay'] for x in r]); ev=np.nanmean([x['ev'] for x in r]); al=np.nanmean([x['acc_lat'] for x in r])
    rows.append(f"{L:g} & {dl:.1f} & {ev:.2f} & {al:.1f} \\\\")
    c1.append(f"({L:g},{ev:.2f})"); c2.append(f"({L:g},{al:.1f})"); c3.append(f"({L:g},{dl:.1f})")
S['T_SWEEP']="\n".join(rows); S['SW_EV']=" ".join(c1); S['SW_AL']=" ".join(c2); S['SW_DL']=" ".join(c3)
# ablation (normal and outage pooled? use Normal), pooled over scenarios
rows=[]
full_d=np.concatenate([arr(sc,'Normal',FOG,'delay') for sc in f.SCEN]); full_e=np.concatenate([arr(sc,'Normal',FOG,'ev') for sc in f.SCEN])
pvals={}
abl=[(FOG,'Full fog system'),('Fog w/o adaptive','w/o adaptive control'),('Fog w/o EV priority','w/o EV priority'),('Fog w/o accident alert','w/o accident alert')]
for v,nm in abl:
    d=np.concatenate([arr(sc,'Normal',v,'delay') for sc in f.SCEN]); e=np.concatenate([arr(sc,'Normal',v,'ev') for sc in f.SCEN])
    qm=np.mean([arr(sc,'Normal',v,'qmax').mean() for sc in f.SCEN]); th=np.mean([arr(sc,'Normal',v,'thr').mean() for sc in f.SCEN])
    ad=100*np.mean([arr(sc,'Normal',v,'acc_det').mean() for sc in f.SCEN])
    # accident-run delay not separate
    if v==FOG: pd='--'; pe='--'; dd=0
    else:
        pd=stats.wilcoxon(d,full_d).pvalue; 
        mask=~np.isnan(e)&~np.isnan(full_e); pe=stats.wilcoxon(e[mask],full_e[mask]).pvalue if (e[mask]-full_e[mask]).any() else 1.0
    fmt=lambda p: p if isinstance(p,str) else ("$<$0.001" if p<1e-3 else f"{p:.3f}")
    rows.append(f"{nm} & {d.mean():.1f} & {np.nanmean(e):.1f} & {qm:.1f} & {th:.0f} & {ad:.0f} & {fmt(pd)} & {fmt(pe)} \\\\")
S['T_ABL']="\n".join(rows)
d=np.concatenate([arr(sc,'Normal',CL,'delay') for sc in f.SCEN]); e=np.concatenate([arr(sc,'Normal',CL,'ev') for sc in f.SCEN])
# stats table: per scenario Normal
rows=[]; allp=[]
tests=[]
for sc in f.SCEN:
    for o in [FX,WB,CL]:
        a=arr(sc,'Normal',FOG,'delay'); b=arr(sc,'Normal',o,'delay'); dd=b-a
        p=stats.wilcoxon(a,b).pvalue; tests.append((sc,o,dd.mean(),p,dd.mean()/dd.std(ddof=1)))
# Holm
order=np.argsort([t[3] for t in tests]); m=len(tests); adj=np.zeros(m); run=0
for rank,i in enumerate(order):
    run=max(run,min(1,(m-rank)*tests[i][3])); adj[i]=run
rows=[]
for sc in f.SCEN:
    cells=[]
    for o in [FX,WB,CL]:
        i=[k for k,t in enumerate(tests) if t[0]==sc and t[1]==o][0]; t=tests[i]
        ps="$<$0.001" if adj[i]<1e-3 else f"{adj[i]:.3f}"
        cells.append(f"{t[2]:.2f} & {ps} & {t[4]:.2f}")
    rows.append(f"{short[sc]} & "+" & ".join(cells)+" \\\\")
S['T_STAT']="\n".join(rows)
# Friedman across 4 controllers pooled seeds per scenario
fr=[]
for sc in f.SCEN:
    x=[arr(sc,'Normal',v,'delay') for v in [FX,WB,CL,FOG]]
    s,p=stats.friedmanchisquare(*x); fr.append((sc,s,p))
S['T_FRIED']=" ".join(f"{short[sc].split()[0]}: $\\chi^2_F$={s:.1f}" for sc,s,p in fr)
# outage EV stat
a=np.concatenate([arr(sc,'Outage',FOG,'ev') for sc in f.SCEN]); b=np.concatenate([arr(sc,'Outage',CL,'ev') for sc in f.SCEN])
S['OUT_EV']=(np.nanmean(a),np.nanmean(b),stats.wilcoxon(a,b).pvalue)
# bootstrap CI for fog vs fixed pooled mean delay diff, fog vs webster
rng=np.random.default_rng(0)
def ci(x):
    bs=[rng.choice(x,len(x)).mean() for _ in range(5000)]; return np.percentile(bs,[2.5,97.5])
fx_d=np.concatenate([arr(sc,'Normal',FX,'delay') for sc in f.SCEN]); wb_d=np.concatenate([arr(sc,'Normal',WB,'delay') for sc in f.SCEN]); cl_d=np.concatenate([arr(sc,'Normal',CL,'delay') for sc in f.SCEN])
S['CI']=dict(fx=(fx_d-full_d).mean(), fxci=ci(fx_d-full_d).tolist(), wb=(wb_d-full_d).mean(), wbci=ci(wb_d-full_d).tolist(), fxpct=100*(fx_d.mean()-full_d.mean())/fx_d.mean(), wbpct=100*(wb_d.mean()-full_d.mean())/wb_d.mean())
S['LATSTATS']={f"{k[0]}-{k[1]}":[x.mean(),np.percentile(x,95),np.percentile(x,99)] for k,x in lat.items()}
json.dump(S,open('snips.json','w'),indent=1,default=float)
for k in ['T_STAT','T_FRIED','OUT_EV','CI','T_ABL','T_COND','T_MULTI','T_SWEEP','T_LAT','T_DELAY']: print(k); print(S[k])
