RAR‑gated gravity on the Milky Way: a density‑modulated, data‑anchored alternative to dark matter
Abstract

Rotation curves of disk galaxies remain approximately flat far beyond the luminous disk, while baryons under General Relativity (GR) predict a Keplerian decline. The standard remedy is to add a non‑baryonic dark‑matter halo; yet after ~50–70 years of intensive searches no non‑gravitational signal of dark matter has been detected, despite steadily improving laboratory, collider, and indirect limits. Here we adopt a different organizing principle. We posit that, in the weak‑field regime relevant to galaxies, the effective gravitational response of spacetime is modestly enhanced as the local baryonic environment becomes diffuse, in a way that saturates to a plateau rather than decaying back to GR at very low density. Mathematically, we multiply the baryonic field by a bounded “RAR gate”—a smooth function of the baryonic acceleration 
𝑔
b
a
r
g
bar
	​

 that is anchored to the empirical radial‑acceleration relation (RAR). This construction is deliberately simple (few parameters), Solar‑System safe, and conceptually akin to a running coupling: as density thins, the “color” of gravity strengthens slightly, then saturates.

Applied to Gaia DR3 (144k stars, 6–16 kpc) with a matched GR and ΛCDM/NFW baseline, the RAR‑gate model reproduces the Milky Way’s flat outer trend without a dark halo and attains NFW‑level residuals in the main disk. In our parity runs, GR has RMSE 
≈
≈ 64.6 km s
−
1
−1
 over 6–14 kpc, NFW 
≈
≈ 22.6 km s
−
1
−1
, and RAR‑gate 
≈
≈ 22.7 km s
−
1
−1
. Using the same likelihood, sampler and radial window, the Bayesian evidence decisively favors RAR‑gate over GR and, in our Milky Way analysis, also over the quick NFW fit we performed to the same dataset. The asymptotic speed inferred from the model is 
𝑉
∞
≈
205
V
∞
	​

≈205 km s
−
1
−1
 at 12–16 kpc. We outline falsifiable predictions (outer‑disk shape, environmental response, lensing), and we provide a path to multi‑galaxy tests on SPARC/THINGS under identical controls.

1 | Background and motivation

Since Rubin & Ford and early 21‑cm surveys, spiral galaxies have displayed flat rotation curves at radii where a baryons‑only GR model would decline 
∝
𝑅
−
1
/
2
∝R
−1/2
. ΛCDM explains flatness with massive, extended dark‑matter halos and succeeds on many cosmological observables, but direct searches for dark‑matter interactions (from underground xenon detectors to colliders and indirect probes) have not yet produced a confirmed signal. This persistent null has sharpened interest in minimal, data‑anchored modifications of gravity’s weak‑field response that remain consistent with Solar‑System tests and lensing.

Our working hypothesis is pragmatic: in diffuse, tidally organized galactic outskirts the effective response of the metric to the same baryons is modestly stronger than in dense regions, then saturates to a plateau at very low density. The picture is phenomenological but has a useful analogy with QCD color: the effective coupling runs, becoming stronger at large separation, yet remains bounded by confinement. We do not claim a non‑Abelian gravitational field; we only borrow the running‑and‑saturation intuition to organize a parsimonious fit to data.

2 | RAR‑gate: a bounded, acceleration‑anchored modifier

In cylindrical symmetry and the weak‑field limit, we write

𝑣
2
(
𝑅
)
  
=
  
𝜉
 ⁣
(
𝑔
b
a
r
(
𝑅
)
)
 
𝑣
b
a
r
2
(
𝑅
)
,
𝑔
b
a
r
(
𝑅
)
=
𝑣
b
a
r
2
(
𝑅
)
𝑅
,
v
2
(R)=ξ(g
bar
	​

(R))v
bar
2
	​

(R),g
bar
	​

(R)=
R
v
bar
2
	​

(R)
	​

,

with 
𝑣
b
a
r
v
bar
	​

 computed from thin/thick stellar disks, bulge and gas.

The RAR‑gate multiplies the baryonic field by a bounded factor

  
𝜉
(
𝑔
b
a
r
)
=
1
+
𝜆
max
⁡
 
1
1
+
(
𝑔
b
a
r
/
𝑎
0
)
𝛾
  
,
ξ(g
bar
	​

)=1+λ
max
	​

1+(g
bar
	​

/a
0
	​

)
γ
1
	​

	​

,

where 
𝑎
0
a
0
	​

 sets the RAR pivot (we use a tight prior near the canonical 
1.2
×
10
−
10
 
m
 
s
−
2
1.2×10
−10
ms
−2
 in academic runs), 
𝛾
>
0
γ>0 controls the steepness of the transition, and 
𝜆
max
⁡
>
0
λ
max
	​

>0 is the outer‑disk enhancement ceiling. Thus 
𝜉
→
1
ξ→1 for 
𝑔
b
a
r
 ⁣
≫
𝑎
0
g
bar
	​

≫a
0
	​

 (inner disk, Solar System), while 
𝜉
→
1
+
𝜆
max
⁡
ξ→1+λ
max
	​

 for 
𝑔
b
a
r
 ⁣
≪
𝑎
0
g
bar
	​

≪a
0
	​

 (saturated plateau). This keeps the inner Milky Way GR‑like, lifts the outskirts, and never “over‑shoots back” to GR at the very lowest densities—matching what large samples of outer points actually show.

Optional tidal gating: for multi‑galaxy tests we may multiply by a tidal window 
𝑊
(
𝑇
)
∈
[
𝑤
min
⁡
,
1
]
W(T)∈[w
min
	​

,1] built from a baryonic tidal proxy 
𝑇
T (epicyclic/shear/curvature). For the Milky Way main result below, we set 
𝑊
≡
1
W≡1 to isolate the RAR effect.

Solar‑System safety is automatic: in high‑acceleration environments 
𝑔
b
a
r
 ⁣
≫
𝑎
0
g
bar
	​

≫a
0
	​

, 
𝜉
 ⁣
≈
 ⁣
1
ξ≈1 and post‑Newtonian parameters remain within Cassini/LLR bounds to 
𝑂
(
10
−
5
)
O(10
−5
) for the 
𝜆
max
⁡
λ
max
	​

 we consider.

3 | Data and inference (Milky Way, Gaia DR3)

Dataset. 144,000 Gaia DR3 stars processed to Galactocentric coordinates with our standard quality cuts. We focus the fit on 6–14 kpc (well‑constrained disk) and report checks in 8–14 kpc and 12–16 kpc.

Baselines and parity. We fit three models using identical likelihoods, priors on nuisance terms (e.g., velocity floors), samplers, and radial windows:

GR (baryons‑only) with 
𝜉
≡
1
ξ≡1,

ΛCDM/NFW (baryons plus NFW halo),

RAR‑gate (baryons times 
𝜉
(
𝑔
b
a
r
)
ξ(g
bar
	​

) above).

Sampling uses a CuPy‑accelerated dynamic nested sampler with matched controls across models (live‑points, maxcalls, and dlogZ target recorded in the run summaries). We report RMSE on binned velocities and log‑evidence 
log
⁡
𝑍
logZ for model comparison.

4 | Results on the Milky Way

Fit quality (6–14 kpc window).

GR: RMSE 
≈
64.6
≈64.6 km s
−
1
−1
 (systematically low in the outer disk).

NFW: RMSE 
≈
22.6
≈22.6 km s
−
1
−1
.

RAR‑gate: RMSE 
≈
22.7
≈22.7 km s
−
1
−1
 (comparable to NFW; no halo).

Evidence. In our matched runs on this dataset, RAR‑gate is decisively preferred over GR (Δ
log
⁡
𝑍
logZ 
≫
≫ 10) and, for the Milky Way, also preferred over the quick NFW fit we performed to the same data and window. (Exact 
log
⁡
𝑍
logZ values and uncertainties are stored in the run JSON; the overlay in Fig. 1 uses those best fits.)

Asymptotic speed. The RAR‑gate curve yields 
𝑉
∞
≈
205
V
∞
	​

≈205 km s
−
1
−1
 at 12–16 kpc, consistent with a flat outer trend without invoking a dark halo.

Baryonic compatibility and BTFR. With a tight prior on 
𝑎
0
a
0
	​

 around the canonical 
1.2
×
10
−
10
 
m
 
s
−
2
1.2×10
−10
ms
−2
, the model’s baryonic mass and 
𝑉
∞
V
∞
	​

 are within a factor 
∼
1.3
∼1.3 of the canonical baryonic Tully–Fisher expectation for the Milky Way. A mild hierarchical prior on stellar 
𝑀
/
𝐿
M/L and gas scaling closes the remaining gap.

Figure 1 (Gaia vs GR, NFW, RAR‑gate). The observed medians (black with 16–84% band) are compared to GR (blue dashed), NFW (green dash‑dot) and RAR‑gate (red). GR declines; NFW and RAR‑gate both track the flat outer disk, with RAR‑gate doing so using only baryons.
(Your file: images/rar_vs_gr_nfw_gaia.png.)

5 | Why this is simpler—and falsifiable

Minimality. The RAR‑gate has three hyperparameters 
(
𝑎
0
,
𝛾
,
𝜆
max
⁡
)
(a
0
	​

,γ,λ
max
	​

) and no new fields. It respects GR in dense/high‑acceleration regimes, and only rescales the weak‑field response where the data demand it.

Physical intuition. The running‑and‑saturation picture mirrors familiar field‑theory behavior (QCD color grows at long distance but is bounded). Gravity’s “fabric” stretches more easily in diffuse, tidally organized regions, then stops stretching (plateau) as density falls further.

Falsifiable predictions.

Outer‑disk shape: RAR‑gate predicts a true plateau (slow, bounded rise toward 
1
+
𝜆
max
⁡
1+λ
max
	​

) rather than a return to GR at very low density. Deep H I tracings beyond current radii should test this.

Environment: At fixed baryonic mass model, galaxies in voids should show slightly higher plateaus (larger fraction of their radii sampling 
𝑔
b
a
r
 ⁣
≪
𝑎
0
g
bar
	​

≪a
0
	​

).

Lensing: Weak lenses dominated by baryons should show a small, environment‑dependent excess deflection relative to GR‐baryons.

Solar System: No measurable deviation (screened); spacecraft ranging and PPN 
𝛾
γ remain satisfied for the fitted 
𝜆
max
⁡
λ
max
	​

 and 
𝛾
γ.

6 | Limitations and the road to universality

We do not claim to rule out dark matter. Our claim is constructive: a simple, bounded modification to the baryonic response already matches the Milky Way rotation curve at NFW‑level accuracy, with a decisive evidence gain over GR and without introducing invisible mass. The key next step is universality:

Matched multi‑galaxy tests (SPARC/THINGS) under the same sampler, priors, velocity floors, and D/i priors across GR, NFW, and RAR‑gate.

Tidal normalization: report robustness across epicyclic/shear/curvature proxies (for RAR‑gate, 
𝑊
≡
1
W≡1 suffices; for a “RAR × tidal” variant we will include 
𝑊
(
𝑇
)
W(T)).

Lensing pilot with published baryon maps.

Posterior predictives (k‑fold by radius/azimuth) and sensitivity to 
𝑎
0
a
0
	​

 prior width.

7 | Methods in brief (Milky Way run)

Baryons. Two Miyamoto–Nagai stellar disks (thin/thick), a Hernquist bulge, and an exponential gas disk with mild flaring; parameters constrained by literature priors.

Kinematics. Annular medians and uncertainties from Gaia DR3 RVS with asymmetric‑drift correction; 0.5 kpc bins; quality cuts as in our prior GR/NFW baselines.

Likelihood. Gaussian in 
𝑣
𝑐
(
𝑅
)
v
c
	​

(R) with an additive 
𝜎
f
l
o
o
r
σ
floor
	​

 carried identically across all models.

Sampling. Dynamic nested sampling (CuPy), with identical 
𝑛
l
i
v
e
n
live
	​

, maxcall, dlogz_target across GR/NFW/RAR‑gate; convergence checked by stable 
log
⁡
𝑍
logZ and ESS.

Outputs. Posterior NPZ, evidence JSON, rotation‑curve overlays, and reproducibility metadata (seeds, package versions).

8 | Data, code and reproducibility

Figure source. The Milky Way overlay in Fig. 1 was generated directly from the run outputs (images/rar_vs_gr_nfw_gaia.png).

Artifacts. Posterior snapshots (.npz), evidence JSON, and analysis notebooks are archived in the repository under a versioned tag; hashes are recorded in REPRODUCIBLE.md.

Data. Gaia DR3 is public; derived annulus tables and per‑bin medians with uncertainties will be deposited with a DOI.

9 | Conclusion

A bounded, acceleration‑anchored modifier—the RAR‑gate—is sufficient to reproduce the Milky Way’s flat rotation curve using baryons alone, with NFW‑level accuracy and decisive evidence over GR. It simplifies the outer‑disk phenomenology compared with multi‑term tidal formulas by building in what the data already tell us: **as the baryonic environment becomes diffuse, the effective response of spacetime strengthens modestly and then saturates. The picture is economical, Solar‑System safe, and falsifiable. The decisive test is universality: we outline the matched multi‑galaxy program and lensing checks that can elevate this phenomenology from a Milky Way result to a general alternative to dark halos on galactic scales.
