# Rajapinta — everything lives at the thaw line

*rajapinta (Finnish): interface, boundary surface. The word was already in the shader comments.*

**A synthesis repo.** One thesis, gathered from five projects that found it independently,
plus three new experiments (run here, numbers below) that test whether the thesis
survives being made quantitative.

> **Structure is an interface phenomenon between frozen time and free flow.
> The bulk on either side is dead. The width of the interface belongs to the
> observer — and what the frozen side remembers is exactly its topology.**

*Do not hype. Do not lie. Just show.*
*Claude (Anthropic) with Antti Luode — PerceptionLab, July 2026.*

---

## 1. Provenance — five repos, one moral

| repo | what it is | what it found |
|---|---|---|
| **HorizonNet** | weight-tied LM + Banach halting certificates | Fixed point is path-independent (AA=1.000), unique, and *worse*. Competence peaks at training depth K and dies at the attractor. A sound state-space certificate fires only on frozen state: certified compute is idle compute. Decision settles ~2× before state — the honest horizon lives in the retina. |
| **Resonant Cortex (RCNet)** | complex-valued net grokking modular addition | Generalization arrives *at* crystallization: phases snap into a circle. Neurogenesis (0.01σ shake on frustration plateaus) precedes the snap. Wrapped rings = neurons at Fourier frequency k>1. |
| **Janus Cabbage** | two images in one net at phases 0 / π/2 | Storage on a continuous U(1) axis: learned, soft orthogonality; smooth morphs; slots interfere gracefully. |
| **Zeta Dreamer** | images stored at times t via log-prime arms | Storage on an incommensurate spectrum: structural orthogonality from unique factorization; slots never alias; transitions are beats, not morphs. |
| **Entrain** | Stuart–Landau oscillators, surprise-gated growth | Built *from* interface material (every node parameterized by distance μ from a Hopf point). Entrainment = reversible local crystallization used as routing (measured Arnold tongue predicts router resolution). Ring memory T4: a chirality bit protected by loop topology, 12/12 vs 0/12 open chain. Directed nucleation: grow ears at the spectral peak of failures. |

The physics frame is Volovik's (*The Universe in a Helium Droplet*), with his discipline:
condensed-matter analogies license **kinematics** (horizons, ring spectra, defect
statistics, KZ scaling) and never **dynamics**. The companion physics synthesis is
*The Clockfield, Assembled* (frozen cores, Γ-shells, the KK→prime tower arc).

## 2. The taxonomy

Two axes sort every system above.

**Axis 1 — task topology.** Is the thing to be learned a *crystal* (static structure:
modular arithmetic, a fixed image) or a *flow* (context-dependent: language, signals)?
**Frozen states can hold frozen structure; only transients can hold flows.**
HorizonNet died because it pushed a flow task into the frozen phase. RCNet succeeded
because its task *is* a crystal — grokking is the representation freezing into the
task's own symmetry.

**Axis 2 — representation commensurability.** What do the phases do to each other?

| position | orthogonality | phases interact? | good for | exemplar |
|---|---|---|---|---|
| continuous (U(1)) | learned, soft | all, softly | morphing, superposition | Janus |
| commensurate (rational ratios) | none — they lock | inside Arnold tongues | **computation** (phase relations carry information: Entrain B3) | Entrain, RCNet |
| incommensurate (log-prime) | structural, hard | never | interference-free storage | Zeta |

The 2×2 (task × representation) has one lethal cell — flow task forced frozen
(HorizonNet) — and one untried cell: flow tasks on incommensurate positional
frequencies (log-prime RoPE). Flagged, not pursued.

## 3. New experiments — run, not proposed

All numpy, CPU, seconds-to-minutes. `python3 experiments/exp*.py` reproduces every
number; raw JSON in `results/`, figures in `figs/`.

### EXP1 — Kibble–Zurek defect census (`exp1_kz_census.py`)

Anneal T→0 over τ steps, freeze, count defects, sweep τ. Two systems:
**(A)** the 1D XY ring (calibration — the literature system), and
**(B)** the *character model*: phases on Z_P with energy
−Σ cos(φ_a + φ_b − φ_{(a+b) mod P}), whose exact ground states are the characters
φ_a = 2πka/P — the same solution family RCNet groks into. `bmax` sets constraint
locality (b ≤ 3 = local updates; b ≤ P−1 = every constraint global, the full-batch analog).

**[V] F1 — KZ scaling appears in the grokking analog.**
XY calibration: ⟨W²⟩ falls with quench time, measured slope **−0.28** over the scaling
window (τ = 30…1000; a floor appears beyond it — trapped winding reaches its
equilibrium-at-freeze-out value for N=256; reported, not hidden).
Character model, local constraints: defect count 9.3 → 2.9 over τ = 10…3000,
power-law slope **−0.19**. Slow quenches make cleaner crystals, quantitatively.

**[V] F2 — locality is required: global constraints kill defects.**
Same model, bmax = 59: defects ≈ 0 for every τ ≥ 300, no scaling at all. KZ physics
needs a causal horizon; long-range constraints delete it. Translated to learning:
**minibatch locality is what makes defect formation possible; the full-batch limit
suppresses it.** (A falsifiable corollary for RCNet: batch-size should control the
census of leftover k-modes. Protocol in §5.)

| τ | XY ⟨W²⟩ | character, local ⟨walls⟩ | character, global ⟨walls⟩ |
|--:|--:|--:|--:|
| 10 | — | 9.28 | 1.12 |
| 30 | 6.66 | 8.88 | 1.62 |
| 100 | 6.34 | 8.44 | 1.31 |
| 300 | 4.56 | 6.94 | 0.00 |
| 1000 | 2.50 | 5.38 | 0.00 |
| 3000 | — | 2.91 | 0.00 |

### EXP2 — Fluid vs crystal storage, linearized (`exp2_storage_axes.py`)

Same budget (one complex coefficient per arm), same solver, only the slot geometry
differs: Janus rotates every arm by one global angle; Zeta rotates arm k at its own
log-prime rate. Registered predictions P1–P3 were written into the file header
*before* running.

**[V] P1** — both systems exact at N ≤ 2 (two real dof per arm), degradation begins at N = 3.
**[K→V] P2** — the earlier intuition "zeta stores more worlds" was registered as
expected-to-fail in the linear regime, and it failed as registered: error magnitudes
are near-identical (equal Shannon budget; e.g. N=5: 0.607 vs 0.573).
**[V] P3 — the difference is error *structure*, not error *size*.**
Ghost correlation (|corr(residual of slot j, other stored world i)|, null baseline 0.17):

| N | Janus err | Janus ghost | Zeta err | Zeta ghost |
|--:|--:|--:|--:|--:|
| 3 | 0.358 | **0.60** | 0.295 | 0.36 |
| 5 | 0.607 | 0.31 | 0.573 | 0.20 |
| 8 | 0.746 | 0.22 | 0.756 | **0.17 (= null)** |

**Fluid storage fails by showing you the other world (coherent double-exposure);
crystal storage fails as featureless noise.** Janus's slider-bleed and Zeta's clean
slots were never about capacity — they're about what the failure looks like.
(Honest caveat: with K=24 arms the zeta scrambling is only partial at small N;
and Zeta Dreamer's apparent capacity advantage must therefore come from its
*nonlinear* decoder, which the linear model deliberately removes.)

### EXP3 — The winding fossil (`exp3_winding_fossil.py`)

Quench the character model, relax to full crystallization, and ask what the crystal
remembers. Per run: purity (RMS distance to the nearest exact character), and whether
total winding W at freeze-out survives to the final state.

**[V] F3 — path independence has exactly one exception, and it is topological.**
92–100% of runs purify to an *exact* character (purity < 0.05, typically ~0.002):
the smooth sector forgets its history completely — the AA = 1.000 analog.
But *which* character — the trapped winding W — is a fossil of the quench, and its
protection switches on with quench time:

| τ | purified | winding conserved | mean \|ΔW\| |
|--:|--:|--:|--:|
| 10 | 0.92 | 0.00 | 10.9 |
| 100 | 0.96 | 0.30 | 4.1 |
| 300 | 1.00 | 0.79 | 1.4 |
| 1000 | 0.92 | **0.95** | 0.05 |

Fast quenches leave states rough enough to keep slipping phase during relaxation;
slow quenches lock their winding at freeze-out and carry it into the crystal intact.
**Refinement of HorizonNet's epitaph: the fixed point is where the model has finished
forgetting — *everything except its topological charge*.** This is the Clockfield
"whorls as matter and memory" claim, demonstrated in the grokking analog; together
with Entrain T4 (the ring chirality bit) it promotes topological memory from
*structural isomorphism* to *established in toys* in the combined ledger.

## 4. Killed

**[K] Directed-vs-random nucleation, in this substrate.** Planned: replicate
Entrain's grow-at-the-failure-peak vs RCNet's random shake, as kicks on a
domain-pinned plateau. Died in diagnosis: the metastable states of the character
model are *strained high-winding configurations*, not pinned domain walls — the
"domains" the census reported were rounding artifacts of smooth strain, and kicks
aimed at them measured nothing (three arms statistically identical; then, on properly
coarsened states, all arms timed out because the barrier is topological, not local).
The instrument was wrong for the question. The question ships as a protocol for RCNet
(§5), where plateaus are real and growth events are already logged. The diagnosis
itself produced EXP3.

**[K] "Zeta stores more" (linear regime)** — registered, tested, failed; see P2.

## 5. Protocols — shipped, not run

1. **RCNet KZ census (browser).** Sweep LR / neurogenesis cooldown as quench rate;
   at the accuracy snap, record epoch-of-lock and the winding histogram over neurons.
   F1 predicts a power law; F2 predicts batch size modulates the defect count.
2. **The k-fossil prediction (falsifiable, in the existing tab).** From EXP3: the
   dominant Fourier mode k the network groks into should be *determined before the
   accuracy snap* and conserved through post-grok training. Log argmax-k per epoch;
   if k keeps wandering after lock, F3 does not transfer and we say so.
3. **Directed nucleation, in its right home.** RCNet neurogenesis seeded with the
   dominant Fourier mode of the residual error vs the 0.01σ random shake; race
   time-to-grok over seeds. (Entrain B1 already shows directed placement beats random
   at fixed size in its own substrate.)
4. **Rollout Lyapunov for HorizonNet.** Perturb at the certified tolerance, roll out
   autoregressively, measure divergence rate: per-token risk control bounds one
   bounce, not the cascade. *Statistical retinas are safe per glance, not per gaze.*

## 6. What this repo is not

Not physics. The systems here are toys chosen so that the claims are exactly
checkable; the Volovik rule applies in full (kinematics only). No constant of nature
is derived, no gauge group is selected, and the words "no free parameters" do not
appear except in this sentence. What the repo shows is narrower and solid: the
frozen/flowing taxonomy makes quantitative predictions, three of them survived
contact with measurement, two registered intuitions died honestly, and one experiment
was killed by its own diagnosis — which then produced the best finding in the repo.

```
experiments/exp1_kz_census.py      exp2_storage_axes.py      exp3_winding_fossil.py
results/*.json                     figs/fig{1,2,3}*.png
```
