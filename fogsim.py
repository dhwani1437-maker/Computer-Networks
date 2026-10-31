"""
Fog vs Cloud smart-traffic simulator (single 4-approach signalised intersection).
Discrete-time (dt = 1 s), 3600 s horizon, paired random seeds across controllers.
All parameters are ASSUMPTIONS stated in the paper (Table of parameters).
"""
import numpy as np, json, itertools, sys
from scipy import stats

T = 3600; S = 0.5; STARTUP = 2; YEL = 3
PH = [(0, 2), (1, 3)]            # phase 0 = N,S ; phase 1 = E,W
MING, MAXG, MARGIN = 10, 60, 3
D_DET, V_EV = 200.0, 13.9        # EV detection distance (m) and speed (m/s)
T_ARR = D_DET / V_EV
P_EV, P_ACC = 0.95, 0.90
ACC_CAP, ACC_DUR, T_OBS = 0.2, 300, 2.0

# ---------------- demand scenarios (veh/s per approach) ----------------
def demand(name):
    t = np.arange(T)
    if name == "S1-Light":     lam = np.tile([0.08]*4, (T, 1))
    elif name == "S2-Moderate": lam = np.tile([0.14]*4, (T, 1))
    elif name == "S3-Heavy":   lam = np.tile([0.18]*4, (T, 1))
    elif name == "S4-Rush":
        bell = np.exp(-((t-1800)/700.0)**2)
        ns = 0.08 + 0.17*bell; ew = 0.07 + 0.08*bell
        lam = np.stack([ns, ew, ns, ew], 1)
    elif name == "S5-Bursty":  lam = np.tile([0.18, 0.06, 0.18, 0.06], (T, 1))
    return lam

SCEN = ["S1-Light", "S2-Moderate", "S3-Heavy", "S4-Rush", "S5-Bursty"]
COND = ["Normal", "Congested", "Outage"]

# ---------------- latency model ----------------
def latency(tier, cond, rng, const=None):
    if const is not None and tier == "cloud":
        return const
    if tier == "fog":
        rho = 0.75 if cond == "Congested" else 0.6
        net = 2*rng.lognormal(np.log(0.01), 0.4)
        return net + 0.04 + rng.exponential(1/(50*(1-rho)))
    if cond == "Congested":
        net = 2*rng.lognormal(np.log(0.15), 0.8); tx = 0.6*rng.lognormal(0, 0.5); rho = 0.97
    else:
        net = 2*rng.lognormal(np.log(0.06), 0.5); tx = 0.08; rho = 0.8
    return net + tx + 0.02 + rng.exponential(1/(40*(1-rho)))

def mean_latency(tier, cond, n=20000, seed=1):
    r = np.random.default_rng(seed)
    x = np.array([latency(tier, cond, r) for _ in range(n)])
    return x

def webster_cycle(lam_mean):
    y = 2*max(lam_mean[0], lam_mean[2]) / (2*S) if False else max(lam_mean[0], lam_mean[2])/S + max(lam_mean[1], lam_mean[3])/S
    Lc = 2*(YEL+STARTUP)
    Y = min(y, 0.9)
    C = (1.5*Lc + 5)/(1-Y)
    return int(np.clip(C, 30, 120)), Y

# ---------------- one run ----------------
def run(scen, cond, v, seed, const_lat=None):
    """v: dict(ctrl in {fixed,webster,adaptive}, tier in {fog,cloud}, ev, acc)"""
    er = np.random.default_rng(seed); lr = np.random.default_rng(seed + 99991)
    lam = demand(scen)
    arr = er.poisson(lam)
    if scen == "S5-Bursty":
        for tb in range(60, T, 120):
            arr[tb, er.integers(4)] += 8
    n_ev = er.poisson(T/240.0)
    ev_t = np.sort(er.integers(30, T-80, n_ev)); ev_a = er.integers(0, 4, n_ev); ev_ok = er.random(n_ev) < P_EV
    t_acc = int(er.integers(600, 2400)); a_acc = int(er.integers(4)); acc_ok = er.random() < P_ACC
    # cloud outages
    out = np.zeros(T, bool)
    if cond == "Outage" and v["tier"] == "cloud":
        t0 = 0.0
        while True:
            t0 += lr.exponential(600.0)
            if t0 >= T: break
            out[int(t0):int(t0 + lr.uniform(15, 45))] = True
    def down(t): return out[min(int(t), T-1)]

    q = np.zeros(4, int); credit = np.zeros(4); served = np.zeros(4, int)
    hist = [q.copy()]
    phase, state, timer, nxt = 0, "green", 0, 0
    qsum = 0.0; qmax = 0; arrivals = 0
    # events
    ev_active = []   # dicts
    ev_delays = []; ev_missed = 0
    acc_alert_t = None; acc_lat = None
    wasted = 0.0
    if v["ctrl"] == "fixed": C = 60; gsplit = [27, 27]
    elif v["ctrl"] == "webster":
        C, Y = webster_cycle(lam.mean(0)); g = C - 2*YEL
        tot = max(lam.mean(0)[0], lam.mean(0)[2]) + max(lam.mean(0)[1], lam.mean(0)[3]) + 1e-9
        g0 = int(np.clip(g*max(lam.mean(0)[0], lam.mean(0)[2])/tot, 10, g-10)); gsplit = [g0, g-g0]
    evi = 0
    for t in range(T):
        # --- arrivals
        q += arr[t]; arrivals += arr[t].sum()
        # EV detection events (EV passes detection point at ev_t, reaches stop line at ev_t+T_ARR)
        while evi < n_ev and ev_t[evi] == t:
            a = int(ev_a[evi]); req = None
            if v["ev"] and ev_ok[evi]:
                if v["tier"] == "cloud" and down(t): req = None
                else:
                    req = t + latency(v["tier"], cond, lr, const_lat)
            ev_active.append(dict(a=a, t_arr=t + T_ARR, req=req, target=None, exited=False))
            evi += 1
        # accident
        cap = np.ones(4)
        if t_acc <= t < t_acc + ACC_DUR: cap[a_acc] = ACC_CAP
        if v["acc"] and acc_ok and t == t_acc:
            if not (v["tier"] == "cloud" and down(t)):
                acc_lat = T_OBS + latency(v["tier"], cond, lr, const_lat)
                acc_alert_t = t_acc + acc_lat
        aware = acc_alert_t is not None and acc_alert_t <= t < t_acc + ACC_DUR
        # EV queue join
        for e in ev_active:
            if e["target"] is None and t >= e["t_arr"]:
                ahead = int(0.5*q[e["a"]]); q[e["a"]] += 1
                e["target"] = served[e["a"]] + ahead + 1; e["t_join"] = t
        # preemption request
        req_phase = None
        for e in ev_active:
            if (not e["exited"]) and e["req"] is not None and t >= e["req"]:
                req_phase = 0 if e["a"] in PH[0] else 1; break
        # --- signal control
        stale = hist[max(0, len(hist)-1 - int(round(const_lat if (const_lat is not None and v["tier"]=="cloud") else
                       (latency(v["tier"], cond, lr) if v["ctrl"] == "adaptive" else 0))))]
        cloud_down = (v["tier"] == "cloud" and down(t))
        if state == "yellow":
            timer += 1
            if timer >= YEL:
                phase = nxt; state = "green"; timer = 0
        else:
            timer += 1
            sw = None
            if req_phase is not None:
                if req_phase != phase: sw = req_phase
            else:
                if v["ctrl"] == "adaptive" and not cloud_down:
                    wq = stale.astype(float).copy()
                    if aware: wq[a_acc] = 0
                    wc = wq[list(PH[phase])].sum(); wo = wq[list(PH[1-phase])].sum()
                    gmax = int(np.clip(90*wc/(wc+wo+1e-9), MING, 60))
                    if timer >= MING and wo > 0 and (wc <= 1 or timer >= gmax):
                        sw = 1 - phase
                else:   # fixed-time (also fallback when cloud is down)
                    if v["ctrl"] == "adaptive": gs = [27, 27]
                    elif v["ctrl"] == "fixed": gs = gsplit
                    else: gs = gsplit
                    if timer >= gs[phase]: sw = 1 - phase
            if sw is not None:
                state = "yellow"; timer = 0; nxt = sw
        # --- discharge
        if state == "green":
            for a in PH[phase]:
                if timer > STARTUP:
                    credit[a] += S*cap[a]
                    k = min(int(credit[a]), q[a])
                    q[a] -= k; credit[a] -= k; served[a] += k
                    if k == 0 and q[a] == 0: credit[a] = 0
                    if a == a_acc and cap[a] < 1 and q[a] > 0 and not aware and v["acc"]: wasted += (S - S*cap[a])
        # EV exit
        for e in ev_active:
            if (not e["exited"]) and e["target"] is not None and served[e["a"]] >= e["target"]:
                e["exited"] = True; e["t_exit"] = t; ev_delays.append(t - e["t_arr"])
        qsum += q.sum(); qmax = max(qmax, q.sum())
        hist.append(q.copy())
        if len(hist) > 200: hist = hist[-100:]
    for e in ev_active:
        if not e["exited"]: ev_delays.append(T - e["t_arr"])
    n_det = sum(1 for e in ev_active if e["req"] is not None)
    return dict(delay=qsum/max(arrivals, 1), qavg=qsum/T, qmax=qmax, thr=int(served.sum()),
                ev=float(np.mean(ev_delays)) if ev_delays else np.nan,
                ev_n=len(ev_delays), ev_pre=n_det,
                acc_lat=acc_lat if acc_lat is not None else np.nan,
                acc_det=int(acc_alert_t is not None))

VARIANTS = {
    "Fixed-time":        dict(ctrl="fixed",    tier="fog",   ev=False, acc=False),
    "Webster":           dict(ctrl="webster",  tier="fog",   ev=False, acc=False),
    "Cloud-adaptive":    dict(ctrl="adaptive", tier="cloud", ev=True,  acc=True),
    "Fog-adaptive (proposed)": dict(ctrl="adaptive", tier="fog", ev=True, acc=True),
    "Fog w/o adaptive":  dict(ctrl="fixed",    tier="fog",   ev=True,  acc=True),
    "Fog w/o EV priority": dict(ctrl="adaptive", tier="fog", ev=False, acc=True),
    "Fog w/o accident alert": dict(ctrl="adaptive", tier="fog", ev=True, acc=False),
}
NSEED = 30

if __name__ == "__main__":
    res = {}
    for sc in SCEN:
        for cd in COND:
            for vn, vv in VARIANTS.items():
                res[(sc, cd, vn)] = [run(sc, cd, vv, s) for s in range(NSEED)]
        print("done", sc, file=sys.stderr)
    # latency sweep (heavy, cloud-adaptive with constant latency)
    sweep = {}
    for L in [0.1, 0.5, 1, 2, 5, 10, 20]:
        sweep[L] = [run("S4-Rush", "Normal", VARIANTS["Cloud-adaptive"], s, const_lat=L) for s in range(NSEED)]
    import pickle
    pickle.dump((res, sweep), open("results.pkl", "wb"))
    print("saved")
