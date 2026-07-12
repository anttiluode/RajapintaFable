"""
EXP1 — Kibble-Zurek defect census
=================================
Two systems, one protocol: anneal temperature T -> 0 over tau steps
(the quench), freeze, count topological defects. Sweep tau.

System A — 1D XY ring (calibration).
    E = -sum cos(phi[a+1]-phi[a]).  The literature system: trapped
    winding |W| should fall as a power of quench time. If our census
    machinery can't see KZ here, it can't see it anywhere.

System B — the character model (minimal grokking analog).
    Sites a = 0..P-1 on the ring Z_P. Energy from group triples:
        E = -sum_{a,b} cos(phi[a] + phi[b] - phi[(a+b)%P])
    Exact ground states are the characters phi[a] = 2*pi*k*a/P
    (any winding k) — the same solution family RCNet crystallizes
    into on modular addition. 'bmax' controls constraint locality:
    b drawn from 1..bmax per step. bmax small = local updates,
    bmax = P-1 = every constraint global (full-batch analog).
    Defects = domain walls between regions of different local winding.

Do not hype. Do not lie. Just show.
"""
import numpy as np, json, os

rng_master = np.random.default_rng(7)

def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi

# ---------------- System A: XY ring ----------------
def quench_xy(N, tau, eta=0.1, T0=3.0, seed=0, relax=1500):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, N)
    for t in range(tau):
        T = T0 * (1 - t/tau)
        d = wrap(np.roll(phi,-1) - phi)          # forward difference
        g = -np.sin(d) + np.sin(wrap(phi - np.roll(phi,1)))
        phi = phi - eta*g + np.sqrt(2*eta*T)*rng.standard_normal(N)
    for _ in range(relax):                        # relax at T=0 (winding is topological: relax cleans kinks, cannot unwind)
        d = wrap(np.roll(phi,-1) - phi)
        g = -np.sin(d) + np.sin(wrap(phi - np.roll(phi,1)))
        phi = phi - eta*g
    d = wrap(np.roll(phi,-1) - phi)
    W = int(round(d.sum()/(2*np.pi)))             # trapped winding
    return W*W

# ---------------- System B: character model ----------------
def energy_walls(phi, P):
    d = wrap(np.roll(phi,-1) - phi)
    k_local = np.round(d * P / (2*np.pi)).astype(int)
    walls = int((k_local != np.roll(k_local,-1)).sum())
    return walls, k_local

def quench_characters(P, tau, bmax, nb=4, eta=0.06, T0=2.0, seed=0, relax=150):
    # relax is SHORT by design: walls in this model are mobile (not topological),
    # so the census must happen at freeze-out. Long relax anneals everything --
    # that fact is itself reported in the README.
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, P)
    a = np.arange(P)
    def step(T):
        nonlocal phi
        g = np.zeros(P)
        for b in rng.integers(1, bmax+1, nb):
            c = (a + int(b)) % P
            th = phi[a] + phi[int(b)] - phi[c]
            s = np.sin(th)
            np.add.at(g, a,  s)
            np.add.at(g, c, -s)
            g[int(b)] += s.sum()
        phi = phi - eta*g/nb + (np.sqrt(2*eta*T)*rng.standard_normal(P) if T>0 else 0)
    for t in range(tau):
        step(T0 * (1 - t/tau))
    for _ in range(relax):
        step(0.0)
    walls, _ = energy_walls(phi, P)
    return walls

if __name__ == "__main__":
    out = {"A": {}, "B": {}}
    taus = [10, 30, 100, 300, 1000, 3000]
    taus_A = [30, 100, 300, 1000]   # scaling window; floor beyond (reported)
    seeds = 32

    print("System A — XY ring (N=256)")
    for tau in taus_A:
        W2 = [quench_xy(256, tau, seed=s) for s in range(seeds)]
        out["A"][tau] = {"W2": float(np.mean(W2))}
        print(f"  tau={tau:5d}  <W^2>={np.mean(W2):.2f}")

    print("System B — character model (P=60), locality sweep")
    for bmax in [3, 59]:
        out["B"][bmax] = {}
        for tau in taus:
            ws = [quench_characters(60, tau, bmax, seed=s) for s in range(seeds)]
            out["B"][bmax][tau] = float(np.mean(ws))
            print(f"  bmax={bmax:2d} tau={tau:5d}  <walls>={np.mean(ws):.2f}")

    # power-law fits (log-log slope), only over the decaying region
    def slope(xs, ys):
        xs, ys = np.array(xs, float), np.array(ys, float)
        m = ys > 0
        if m.sum() < 3: return None
        return float(np.polyfit(np.log(xs[m]), np.log(ys[m]), 1)[0])
    out["fits"] = {
        "A_W2_slope": slope(taus_A, [out["A"][t]["W2"] for t in taus_A]),
        "B_local_slope":  slope(taus, [out["B"][3][t]  for t in taus]),
        "B_global_slope": slope(taus, [out["B"][59][t] for t in taus]),
    }
    print("fits:", out["fits"])
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/exp1_kz.json","w"), indent=1)
