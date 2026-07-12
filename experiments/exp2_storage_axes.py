"""
EXP2 — Fluid vs crystal storage (Janus vs Zeta, linearized)
===========================================================
Both systems store N real signals in the SAME budget: one complex
coefficient c_k per arm (2K reals total). They differ ONLY in how a
storage slot transforms the readout phase:

  JANUS (continuous / fluid):    theta_{j,k} = theta_j        (one global
        U(1) angle rotates every arm identically; slots live on one circle)
  ZETA  (incommensurate / crystal): theta_{j,k} = t_j * omega_k * c
        (each arm rotates at its own log-prime rate; slot phases are
        pseudo-orthogonal across arms by incommensurability)

Readout of slot j, arm k:  a_hat_{j,k} = Re( c_k * exp(-i theta_{j,k}) ).
Storage = least squares for c over all slots jointly.

Measured: per-slot reconstruction error vs N, and GHOST CORRELATION =
max over other slots i of |corr(residual_j, s_i)| — does the error of
slot j look like another stored world (coherent ghost) or like noise?

Registered predictions (from the conversation, before running):
  P1: Janus is near-perfect at N<=2 (real/imag quadrature), then degrades.
  P2: For N>2 the ERROR MAGNITUDE is similar in both systems (equal
      Shannon budget) — the earlier 'zeta stores more' intuition is
      expected to FAIL in the linear regime and is registered as such.
  P3: The error STRUCTURE differs: Janus ghosts are coherent (high ghost
      correlation), Zeta residuals are incoherent (low ghost correlation).
"""
import numpy as np, json, os

def primes(n):
    ps, x = [], 2
    while len(ps) < n:
        if all(x % p for p in ps): ps.append(x)
        x += 1
    return np.array(ps, float)

def run(K=24, Nmax=10, trials=20, seed=0):
    rng = np.random.default_rng(seed)
    om = np.log(primes(K))                       # arm identities (log-prime)
    res = {"janus": {}, "zeta": {}}
    for N in range(1, Nmax+1):
        for sysname in ["janus", "zeta"]:
            errs, ghosts = [], []
            for tr in range(trials):
                r2 = np.random.default_rng(1000*tr + N)
                S = r2.standard_normal((N, K))   # real spectra of N signals
                if sysname == "janus":
                    th = np.tile((2*np.pi*np.arange(N)/max(N,4))[:,None] * 0 +
                                 (np.pi/2)*(np.arange(N)/max(N-1,1))[:,None], (1,K))
                else:
                    t = np.arange(N)[None,:].T * 2.399963
                    th = t * om[None,:]          # incommensurate per-arm rates
                # per-arm least squares: Re(c_k e^{-i th_jk}) = S_jk
                # unknowns per arm: (x_k, y_k) with readout x cos(th)+y sin(th)
                A_hat = np.zeros_like(S)
                for k in range(K):
                    M = np.stack([np.cos(th[:,k]), np.sin(th[:,k])], axis=1)
                    c, *_ = np.linalg.lstsq(M, S[:,k], rcond=None)
                    A_hat[:,k] = M @ c
                R = S - A_hat                    # residual spectra
                errs.append(np.mean(R**2) / np.mean(S**2))
                # ghost correlation: residual of slot j vs other stored signals
                # mean |corr| of residual_j with OTHER stored spectra,
                # against a null: same residual vs fresh random spectra.
                g, g0, cnt = 0.0, 0.0, 0
                if N > 1 and np.mean(R**2) > 1e-9:
                    Snull = r2.standard_normal((N, K))
                    for j in range(N):
                        for i in range(N):
                            if i == j: continue
                            g  += abs(np.corrcoef(R[j], S[i])[0,1])
                            g0 += abs(np.corrcoef(R[j], Snull[i])[0,1])
                            cnt += 1
                    g, g0 = g/cnt, g0/cnt
                ghosts.append((g, g0))
            gg = np.array(ghosts)
            res[sysname][N] = {"err": float(np.mean(errs)),
                               "ghost": float(gg[:,0].mean()),
                               "ghost_null": float(gg[:,1].mean())}
    return res

if __name__ == "__main__":
    res = run()
    print(f"{'N':>3} | {'janus err':>10} {'ghost/null':>10} | {'zeta err':>9} {'ghost/null':>10}")
    for N in sorted(res['janus'], key=int):
        j, z = res['janus'][N], res['zeta'][N]
        print(f"{N:>3} | {j['err']:>10.3f} {j['ghost']:>6.2f}/{j['ghost_null']:>4.2f} | {z['err']:>9.3f} {z['ghost']:>6.2f}/{z['ghost_null']:>4.2f}")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/exp2_storage.json","w"), indent=1)
