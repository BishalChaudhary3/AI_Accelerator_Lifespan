"""
Lifecycle carbon, fleet e-waste and scheduling simulation for AI accelerators.
Reproduces every number and figure in the paper (seed fixed).
Run: python3 ewaste_model.py
Outputs: figures (PDF) and results.json in ./out
"""
import heapq
import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
OUT = "out"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------
# Parameters (sources and assumptions are documented in the paper, Table I)
# ----------------------------------------------------------------------
GPU = {  # FP16 tensor dense peak TFLOPS, board TDP in W (vendor spec sheets)
    "V100": dict(peak=125.0, tdp=300.0),
    "A100": dict(peak=312.0, tdp=400.0),
    "H100": dict(peak=989.0, tdp=700.0),
}
GPU_SHARE = 0.75        # GPU share of server power (assumption informed by literature)
PUE = dict(low=1.03, central=1.13, high=1.62)         # Wadenstein & Vanderbauwhede
CI = dict(low=0.017, central=0.231, high=0.642)       # kgCO2e/kWh, same source
DUTY = dict(low=0.2, central=0.4, high=0.6)           # time-busy fraction (Bodner levels)
CEMB = dict(low=521.5, central=1043.0, high=1564.5)   # kgCO2e per GPU slot (8341/8 +-50%)
MASS = dict(low=12.0, central=16.0, high=20.0)        # kg per GPU slot (assumption)
RHO = dict(low=0.5, central=1.0, high=1.0)            # realised share of peak speed-up
FP = dict(low=0.6, central=0.8, high=1.0)             # GPU power at load / TDP
P_IDLE_FRAC = 0.3       # idle power as fraction of busy power (assumption)


def busy_power_w(tdp, fp):
    return tdp / GPU_SHARE * fp


# ----------------------------------------------------------------------
# E1: break-even analysis
# ----------------------------------------------------------------------
def breakeven(old, new, cemb=CEMB["central"], ci=CI["central"], pue=PUE["central"],
              duty=DUTY["central"], rho=1.0, fp=FP["central"]):
    """Years until replacing `old` by `new` for a fixed workload pays back in carbon."""
    s = GPU[new]["peak"] / GPU[old]["peak"]
    r = 1.0 / (1.0 + rho * (s - 1.0))            # fraction of a new GPU per old GPU
    e_old = busy_power_w(GPU[old]["tdp"], fp) * duty * 8760 * pue / 1000.0
    e_new = r * busy_power_w(GPU[new]["tdp"], fp) * duty * 8760 * pue / 1000.0
    save_kwh = e_old - e_new
    c_new = r * cemb
    if save_kwh <= 0:
        return math.inf
    return c_new / (ci * save_kwh)


PAIRS = [("V100", "A100"), ("A100", "H100"), ("V100", "H100")]


def e1_table():
    rows = {}
    for old, new in PAIRS:
        key = f"{old}->{new}"
        rows[key] = {}
        for ci_k in ("low", "central", "high"):
            rows[key][f"CI_{ci_k}"] = {}
            for ce_k in ("low", "central", "high"):
                rows[key][f"CI_{ci_k}"][f"Cemb_{ce_k}"] = breakeven(
                    old, new, cemb=CEMB[ce_k], ci=CI[ci_k])
    return rows


def fig_breakeven():
    ci = np.logspace(math.log10(0.01), math.log10(0.9), 200)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    colors = {"V100->A100": "tab:blue", "A100->H100": "tab:orange", "V100->H100": "tab:green"}
    for old, new in PAIRS:
        k = f"{old}->{new}"
        c = [breakeven(old, new, ci=x) for x in ci]
        lo = [breakeven(old, new, ci=x, cemb=CEMB["low"]) for x in ci]
        hi = [breakeven(old, new, ci=x, cemb=CEMB["high"]) for x in ci]
        ax.plot(ci, c, color=colors[k], label=f"{old}$\\to${new}")
        ax.fill_between(ci, lo, hi, color=colors[k], alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(3, color="gray", ls=":", lw=0.8)
    ax.axhline(5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Grid carbon intensity (kgCO$_2$e/kWh)", fontsize=8)
    ax.set_ylabel("Carbon break-even (years)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_breakeven.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# E2: fleet stock-and-flow
# ----------------------------------------------------------------------
GENS = {  # capacity relative to V100 (peak ratio), busy server-level power per slot (W)
    "A100": dict(cap=312 / 125, p=400 / GPU_SHARE),
    "H100": dict(cap=989 / 125, p=700 / GPU_SHARE),
    # hypothetical future generations: capacity x2.5, efficiency x1.8 per generation
    "X1": dict(cap=989 / 125 * 2.5, p=700 / GPU_SHARE * 2.5 / 1.8),
    "X2": dict(cap=989 / 125 * 6.25, p=700 / GPU_SHARE * 6.25 / 3.24),
}
BUY_GEN = {2020: "A100", 2021: "A100", 2022: "A100", 2023: "H100", 2024: "H100",
           2025: "H100", 2026: "X1", 2027: "X1", 2028: "X1", 2029: "X2", 2030: "X2"}
YEARS = list(range(2020, 2031))
D0 = 20000.0            # initial demand in V100-equivalent compute units
GROWTH = 0.30           # annual demand growth (anchored on IEA accelerated-server growth)
U_TARGET = 0.5          # provisioning headroom: capacity >= demand / U_TARGET
U_MAX = 0.8             # maximum per-unit utilisation under tiered (newest-first) dispatch


def fleet(life, tiered, ci=CI["central"], pue=PUE["central"], cemb=CEMB["central"],
          mass=MASS["central"], growth=GROWTH):
    cohorts = []
    cum_waste_t, cum_emb_t, cum_op_t = 0.0, 0.0, 0.0
    series = []
    purchases_total = 0.0
    for y in YEARS:
        # retire
        keep = []
        for c in cohorts:
            if y - c["year"] >= life:
                cum_waste_t += c["n"] * mass / 1000.0
            else:
                keep.append(c)
        cohorts = keep
        demand = D0 * (1 + growth) ** (y - YEARS[0])
        cap_total = sum(c["n"] * GENS[c["gen"]]["cap"] for c in cohorts)
        need = demand / U_TARGET
        if cap_total < need:
            g = BUY_GEN[y]
            n = (need - cap_total) / GENS[g]["cap"]
            cohorts.append(dict(year=y, gen=g, n=n))
            purchases_total += n
            cum_emb_t += n * cemb / 1000.0
            cap_total = need
        # dispatch
        power_w = 0.0
        if not tiered:
            util = demand / cap_total
            for c in cohorts:
                pb = GENS[c["gen"]]["p"]
                power_w += c["n"] * (P_IDLE_FRAC * pb + (1 - P_IDLE_FRAC) * pb * util)
        else:
            rem = demand
            order = sorted(cohorts, key=lambda c: -GENS[c["gen"]]["cap"] / GENS[c["gen"]]["p"])
            for c in order:
                pb = GENS[c["gen"]]["p"]
                capc = c["n"] * GENS[c["gen"]]["cap"]
                load = min(rem, capc * U_MAX)
                rem -= load
                util = load / capc
                power_w += c["n"] * (P_IDLE_FRAC * pb + (1 - P_IDLE_FRAC) * pb * util)
        energy_kwh = power_w * 8760 / 1000.0 * pue
        cum_op_t += energy_kwh * ci / 1000.0
        series.append(dict(year=y, cum_waste_t=cum_waste_t, cum_emb_t=cum_emb_t,
                           cum_op_t=cum_op_t, units=sum(c["n"] for c in cohorts)))
    return dict(series=series, waste_t=cum_waste_t, emb_t=cum_emb_t, op_t=cum_op_t,
                total_t=cum_emb_t + cum_op_t, purchases=purchases_total)


SCENARIOS = {
    "A": dict(life=3, tiered=False, label="A: 3 yr"),
    "B": dict(life=2, tiered=False, label="B: 2 yr"),
    "C": dict(life=5, tiered=False, label="C: 5 yr"),
    "D": dict(life=5, tiered=True, label="D: 5 yr tiered"),
}


def e2_table():
    out = {}
    for k, sc in SCENARIOS.items():
        out[k] = {}
        for ci_k in ("low", "central", "high"):
            r = fleet(sc["life"], sc["tiered"], ci=CI[ci_k])
            out[k][f"CI_{ci_k}"] = {m: r[m] for m in ("waste_t", "emb_t", "op_t", "total_t", "purchases")}
        out[k]["series"] = fleet(sc["life"], sc["tiered"])["series"]
    return out


def fig_ewaste(e2):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, sc in SCENARIOS.items():
        s = e2[k]["series"]
        ax.plot([x["year"] for x in s], [x["cum_waste_t"] / 1000 for x in s],
                marker="o", ms=2.5, label=sc["label"])
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Cumulative e-waste (kt)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_ewaste.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# E3: scheduling simulation
# ----------------------------------------------------------------------
L_DESIGN = 6.0          # design life in years at nominal conditions (assumption)
U_REF = 0.5             # reference duty cycle for the design life
A_CAL = 0.05            # powered-on calendar ageing: life fraction per year even when idle (assumption)
EA_EV = 0.7             # Arrhenius activation energy (assumption)
KB = 8.617e-5
T_AMB = 35.0            # coolant/ambient (assumption)
R_TH = 0.07             # K per W of GPU power (assumption)
CAP_FRAC, CAP_SPEED_EXP = 0.85, 0.7   # power cap on worn devices and speed scaling
SPEED = {g: GPU[g]["peak"] / GPU["V100"]["peak"] for g in GPU}
FLEET_SPEC = [("V100", 12, 5.0), ("A100", 18, 3.0), ("H100", 18, 1.0)]  # gen, count, age
HORIZON_H = 120 * 24
LOAD = 0.7
JOB_MIX = dict(training=(0.15, 40.0), inference=(0.55, 2.0), batch=(0.30, 8.0))  # share, mean size (V100-h)
PRIORITY = {"training": 0, "inference": 1, "batch": 2}


def af(t_c):
    t = t_c + 273.15
    tref = 60.0 + 273.15
    return math.exp(EA_EV / KB * (1 / tref - 1 / t))


class Dev:
    def __init__(self, idx, gen, age):
        self.idx, self.gen = idx, gen
        self.h0 = max(0.0, 1.0 - age / L_DESIGN)
        self.wear = 0.0
        self.busy_h = 0.0
        self.energy_busy_wh = 0.0
        self.busy_until = 0.0
        self.free = True
        self.tnom = T_AMB + R_TH * GPU[gen]["tdp"]

    @property
    def health(self):
        return self.h0 - self.wear


def make_fleet():
    devs = []
    i = 0
    for g, n, age in FLEET_SPEC:
        for _ in range(n):
            devs.append(Dev(i, g, age))
            i += 1
    return devs


def gen_jobs(rng, load=LOAD):
    total_speed = sum(SPEED[g] * n for g, n, _ in FLEET_SPEC)
    mean_work = sum(p * m for p, m in JOB_MIX.values())
    lam = load * total_speed / mean_work          # jobs per hour
    t, jobs = 0.0, []
    kinds = list(JOB_MIX)
    probs = [JOB_MIX[k][0] for k in kinds]
    while True:
        t += rng.exponential(1 / lam)
        if t > HORIZON_H:
            break
        k = kinds[rng.choice(len(kinds), p=probs)]
        size = rng.exponential(JOB_MIX[k][1])
        jobs.append((t, k, max(size, 0.05)))
    return jobs


PREF = {"training": ["H100", "A100", "V100"],
        "inference": ["A100", "H100", "V100"],
        "batch": ["V100", "A100", "H100"]}


def choose(policy, job_kind, free, state):
    if policy == "round_robin":
        n = state["n"]
        for k in range(n):
            d = state["devs"][(state["rr"] + k) % n]
            if d.free:
                state["rr"] = (d.idx + 1) % n
                return d, 1.0
        return None, 1.0
    if policy in ("efficiency_first", "efficiency_cap"):
        best = max(free, key=lambda d: (GPU[d.gen]["peak"] / GPU[d.gen]["tdp"], -d.idx))
        cap = 1.0
        if policy == "efficiency_cap" and state["waiting"] == 0:
            cap = CAP_FRAC          # opportunistic power cap when nobody is waiting
        return best, cap
    # health-aware tiered (with or without power cap)
    for g in PREF[job_kind]:
        cand = [d for d in free if d.gen == g]
        if cand:
            d = max(cand, key=lambda d: (d.health, -d.idx))
            cap = 1.0
            if policy == "tiered_cap" and d.health < 0.4:
                cap = CAP_FRAC
            return d, cap
    return None, 1.0


def simulate(policy, jobs, ci=CI["central"], pue=PUE["central"]):
    devs = make_fleet()
    state = dict(devs=devs, n=len(devs), rr=0, waiting=0)
    queue = {0: [], 1: [], 2: []}
    ev = [(t, 0, i, "arr") for i, (t, _, _) in enumerate(jobs)]
    heapq.heapify(ev)
    waits = []
    turns = []
    t_end = 0.0
    seq = len(jobs)

    def dispatch(now):
        nonlocal seq
        while True:
            free = [d for d in devs if d.free]
            if not free:
                return
            job = None
            for p in (0, 1, 2):
                if queue[p]:
                    job = queue[p][0]
                    break
            if job is None:
                return
            kind, size, t_arr = job["kind"], job["size"], job["t"]
            state["waiting"] = sum(len(q) for q in queue.values()) - 1
            dev, cap = choose(policy, kind, free, state)
            if dev is None:
                return
            queue[PRIORITY[kind]].pop(0)
            speed = SPEED[dev.gen] * (cap ** CAP_SPEED_EXP)
            dur = size / speed
            pwr = busy_power_w(GPU[dev.gen]["tdp"], FP["central"]) * cap
            t_c = T_AMB + R_TH * GPU[dev.gen]["tdp"] * cap
            rate = af(t_c) / af(dev.tnom) * (1.0 / L_DESIGN - A_CAL) / (8760 * U_REF)   # life-fraction per busy hour
            dev.wear += rate * dur
            dev.busy_h += dur
            dev.energy_busy_wh += pwr * dur
            dev.free = False
            waits.append(now - t_arr)
            turns.append(now - t_arr + dur)
            seq += 1
            heapq.heappush(ev, (now + dur, 1, seq, ("done", dev.idx)))

    while ev:
        t, kind, i, payload = heapq.heappop(ev)
        t_end = max(t_end, t)
        if payload == "arr":
            _, k, s = jobs[i]
            queue[PRIORITY[k]].append(dict(kind=k, size=s, t=t))
        else:
            devs[payload[1]].free = True
        dispatch(t)

    th = max(t_end, HORIZON_H)
    energy_wh = 0.0
    wear_total, ewaste_kg, emb_kg = 0.0, 0.0, 0.0
    scale = 8760.0 / th
    for d in devs:
        pb = busy_power_w(GPU[d.gen]["tdp"], FP["central"])
        idle_wh = P_IDLE_FRAC * pb * (th - d.busy_h)
        energy_wh += d.energy_busy_wh + idle_wh
        w_yr = d.wear * scale + A_CAL
        wear_total += w_yr
        ewaste_kg += MASS["central"] * w_yr
        emb_kg += CEMB["central"] * w_yr
    energy_mwh = energy_wh * pue / 1e6 * scale
    waits = np.array(waits)
    turns = np.array(turns)
    return dict(energy_mwh=energy_mwh,
                op_t=energy_mwh * 1000 * ci / 1000.0,
                emb_t=emb_kg / 1000.0,
                ewaste_kg=ewaste_kg,
                device_lives=wear_total,
                mean_wait_h=float(waits.mean()),
                p95_wait_h=float(np.percentile(waits, 95)),
                mean_turn_h=float(turns.mean()),
                p95_turn_h=float(np.percentile(turns, 95)))


POLICIES = ["round_robin", "efficiency_first", "efficiency_cap", "tiered", "tiered_cap"]


def e3(n_seeds=20, load=LOAD):
    res = {p: [] for p in POLICIES}
    for s in range(n_seeds):
        rng = np.random.default_rng(SEED + s)
        jobs = gen_jobs(rng, load)
        for p in POLICIES:
            r = simulate(p, jobs)
            r["total_t"] = r["op_t"] + r["emb_t"]
            res[p].append(r)
    summary = {}
    for p in POLICIES:
        summary[p] = {}
        for m in res[p][0]:
            v = np.array([r[m] for r in res[p]])
            summary[p][m] = dict(mean=float(v.mean()),
                                 ci95=float(1.96 * v.std(ddof=1) / math.sqrt(len(v))))
    return summary


# ----------------------------------------------------------------------
# E5: sensitivity (tornado) and Monte Carlo on break-even
# ----------------------------------------------------------------------
def tornado(old, new):
    base = breakeven(old, new, rho=RHO["central"])
    params = {
        "Embodied carbon": ("cemb", CEMB),
        "Grid intensity": ("ci", CI),
        "PUE": ("pue", PUE),
        "Duty cycle": ("duty", DUTY),
        "Speed-up realised": ("rho", RHO),
        "GPU power at load": ("fp", FP),
    }
    rows = []
    for name, (arg, rng_) in params.items():
        lo = breakeven(old, new, **{arg: rng_["low"]})
        hi = breakeven(old, new, **{arg: rng_["high"]})
        rows.append((name, lo, hi))
    return base, rows


def fig_tornado():
    base, rows = tornado("A100", "H100")
    rows.sort(key=lambda r: -abs(math.log(max(r[1], 1e-9) / max(r[2], 1e-9))))
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, (name, lo, hi) in enumerate(rows[::-1]):
        a, b = sorted([lo, hi])
        ax.barh(i, b - base, left=base, color="tab:red", alpha=0.7, height=0.6)
        ax.barh(i, a - base, left=base, color="tab:blue", alpha=0.7, height=0.6)
    ax.axvline(base, color="k", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("A100$\\to$H100 carbon break-even (years)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_tornado.pdf")
    plt.close(fig)
    return base, rows


def monte_carlo(n=20000):
    rng = np.random.default_rng(SEED)
    res = {}
    for old, new in PAIRS:
        vals = []
        for _ in range(n):
            kw = dict(
                cemb=rng.triangular(CEMB["low"], CEMB["central"], CEMB["high"]),
                ci=math.exp(rng.uniform(math.log(CI["low"]), math.log(CI["high"]))),
                pue=rng.triangular(PUE["low"], PUE["central"], PUE["high"]),
                duty=rng.triangular(DUTY["low"], DUTY["central"], DUTY["high"]),
                rho=rng.uniform(0.5, 1.0),
                fp=rng.triangular(FP["low"], FP["central"], FP["high"]),
            )
            vals.append(breakeven(old, new, **kw))
        v = np.array(vals)
        res[f"{old}->{new}"] = dict(
            p5=float(np.percentile(v, 5)), median=float(np.median(v)),
            p95=float(np.percentile(v, 95)),
            frac_within_3yr=float((v <= 3).mean()), frac_within_5yr=float((v <= 5).mean()))
    return res


if __name__ == "__main__":
    results = {}
    results["E1"] = e1_table()
    fig_breakeven()
    results["E2"] = e2_table()
    fig_ewaste(results["E2"])
    results["E3"] = e3()
    results["E3_load"] = {str(l): e3(load=l) for l in (0.4, 0.7, 0.9)}
    base, rows = fig_tornado()
    results["E5_tornado"] = dict(base=base, rows=[dict(name=n, low=l, high=h) for n, l, h in rows])
    results["E5_mc"] = monte_carlo()
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("done")
