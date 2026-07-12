"""
EXP3 — The winding fossil (path independence has a topological exception)
=========================================================================
Quench the character model (exp1) and then relax at T=0 until it fully
crystallizes. Measure, per run:
  purity     — RMS distance to the nearest exact character (does the
               smooth sector forget its history?)
  W@freeze   — total winding at the end of the quench
  W@final    — total winding of the final crystal
  conserved  — W@freeze == W@final?

Registered framing: HorizonNet found its fixed point path-INdependent
(AA = 1.000). RCNet crystallizes into SOME Fourier mode k. This
experiment asks what the frozen state remembers.

NOTE (honest ledger): this file replaces a planned directed-vs-random
nucleation experiment. That experiment is KILLED for this substrate:
the metastable states here are strained high-winding configurations,
not pinned domain walls — the 'domains' my census reported were
rounding artifacts of smooth strain, and kicks aimed at them measured
nothing. Diagnosis preserved in the repo README. The nucleation
question ships as a protocol for RCNet (where plateaus are real),
not as a result.
"""
import numpy as np, json, os

def wrap(x): return (x + np.pi) % (2*np.pi) - np.pi

def grad(phi, P, rng, bmax=3, nb=4):
    a = np.arange(P); g = np.zeros(P)
    for b in rng.integers(1, bmax+1, nb):
        c = (a + int(b)) % P
        s = np.sin(phi[a] + phi[int(b)] - phi[c])
        np.add.at(g, a, s); np.add.at(g, c, -s); g[int(b)] += s.sum()
    return g / nb

def total_winding(phi):
    return int(round(wrap(np.roll(phi,-1)-phi).sum()/(2*np.pi)))

def run(P, seed, tau, relaxN=30000, eta=0.06, T0=2.0):
    rng = np.random.default_rng(seed)
    phi = np.random.default_rng(seed).uniform(-np.pi, np.pi, P)
    for t in range(tau):
        T = T0*(1 - t/tau)
        phi = phi - eta*grad(phi, P, rng) + np.sqrt(2*eta*T)*rng.standard_normal(P)
    Wf = total_winding(phi)
    for _ in range(relaxN):
        phi = phi - eta*grad(phi, P, rng)
    Wl = total_winding(phi)
    a = np.arange(P); base = 2*np.pi*Wl*a/P
    off = np.angle(np.mean(np.exp(1j*(phi - base))))
    purity = float(np.sqrt(np.mean(wrap(phi - base - off)**2)))
    return Wf, Wl, purity

if __name__ == "__main__":
    P, seeds = 60, 24
    taus = [10, 30, 100, 300, 1000]
    out = {}
    print(f"{'tau':>5} | (stats over runs that fully crystallized, purity<0.05)")
    for tau in taus:
        rows = [run(P, s, tau) for s in range(seeds)]
        ok   = [(wf, wl, p) for wf, wl, p in rows if p < 0.05]   # fully crystallized
        pfrac= len(ok)/len(rows)
        cons = np.mean([wf == wl for wf, wl, _ in ok]) if ok else float('nan')
        dW   = np.mean([abs(wl - wf) for wf, wl, _ in ok]) if ok else float('nan')
        wmag = np.mean([abs(wl) for _, wl, _ in ok]) if ok else float('nan')
        out[tau] = {"purified_frac": pfrac, "conserved_frac": float(cons),
                    "mean_abs_dW": float(dW), "mean_abs_Wfinal": float(wmag),
                    "rows": [[int(a), int(b), float(c)] for a, b, c in rows]}
        print(f"{tau:>5} | purified {pfrac:4.2f} | conserved {cons:4.2f} | mean|dW| {dW:5.2f} | mean|W| {wmag:5.1f}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/exp3_winding_fossil.json", "w"), indent=1)
