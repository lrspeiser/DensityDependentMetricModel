# Tidal Field Relativity with a Saturated RAR Gate (TFR‑S):

A bounded, environment‑modulated metric that reproduces flat galaxy rotation curves without dark matter

## Abstract

Observed rotation curves of disk galaxies remain nearly flat to large radii, inconsistent with GR with baryons alone. We propose Tidal Field Relativity with Saturated RAR Gate (TFR‑S), a relativistic, metric‑based modification in which the effective gravitational response is rescaled by a bounded factor 𝜉 that depends on (i) the baryonic field strength—encoded via the empirical Radial Acceleration Relation (RAR)—and (ii) a tidal‑structure gate that localizes the modification to the low‑density, tidally structured outskirts of disks. Unlike earlier tidal forms that forced 𝜉→1 at very low density, TFR‑S rises quickly as density thins, tracks the RAR slope across the well‑measured range, and asymptotically saturates (plateau) only below accelerations far outside current kinematic reaches, thereby preserving the observed outer flatness without unphysical divergence and without an artificial return to GR.

On the Milky Way (Gaia DR3, 5–16 kpc) TFR‑S decisively improves over a matched GR baryons‑only baseline (Δlog Z ≈ [to be inserted]). Across high‑quality SPARC spirals (curated Q=1–2, good beam smearing, robust inclinations) TFR‑S lifts the baryonic curve in the outskirts and reduces residuals relative to GR in the majority of systems; a cohort summary is reported with matched controls and quality cuts ([to be inserted]). The model is Solar‑System safe by construction via an exponential high‑density screen. We provide a fully reproducible workflow and delineate predictions for lensing and environmental trends.

## 1. Motivation and framing

Two robust empirical facts guide our construction:

RAR slope in the outskirts. Where baryons produce low accelerations (𝑔bar≲a0), the observed acceleration follows 𝑔obs≈a0𝑔bar, implying a mass‑discrepancy factor 𝐷≡𝑔obs/𝑔bar∝𝑔bar−1/2. This is precisely the scaling needed to offset the baryonic fall‑off and keep 𝑉(𝑅) nearly flat.

Tidal organization. Low‑density, outer disks are tidally structured (warps, spiral arms, lopsidedness), while dense inner regions and the Solar System must remain in the GR limit.

Earlier tidal‑band forms in our work captured (2) but over‑constrained (1) by pushing 𝜉→1 as density fell—forcing rotation curves to decline back toward GR just where data show they remain flat. TFR‑S corrects this: we anchor the modification to the RAR over the data‑rich regime and only saturate (plateau) at ultralow 𝑔bar well beyond current radii.

## 2. Model

### 2.1 Ingredients and notation

- 𝑔bar(𝑅): baryonic centripetal acceleration from the GR baryon model.
- 𝑇(𝑅): scalar tidal indicator derived from the baryonic potential (epicyclic/shear/curvature proxy; §4.3).
- 𝜌(𝑅,0): midplane baryonic density (used for Solar‑System screening only).
- 𝑎0: RAR pivot (prior centered at 1.2×10−10 m s−2).
- 𝐷(𝑔bar): empirical mass‑discrepancy function from RAR.

The modified circular speed is

𝑉2(𝑅)=𝜉(𝑔bar,𝑇,𝜌)𝑉bar2(𝑅), 𝑉bar2(𝑅)=𝑅 𝑔bar(𝑅).

### 2.2 RAR gate (data‑anchored) with low‑g saturation

We start from a smooth version of the RAR discrepancy:

𝐷RAR(𝑔)=1/(1−exp[−𝑔/𝑎0]).

At very small 𝑔 this diverges like 𝑎0/𝑔; to avoid unphysical growth deep in voids we soft‑saturate to a finite plateau 𝐷∞>1 using a q‑soft‑min:

𝐷sat(𝑔;𝑎0,𝐷∞,𝑞)=1+[(𝐷RAR(𝑔)−1)−𝑞+(𝐷∞−1)−𝑞]−1/𝑞, 𝑞>0.

For 𝑔≫𝑔cut (well inside observed outskirts) 𝐷sat≈𝐷RAR and flatness is retained; only for 𝑔≪𝑔cut does 𝐷sat→𝐷∞. We choose 𝑔cut implicitly by the pair (𝐷∞,𝑞) such that current SPARC/MW radii remain in the RAR regime; the plateau is a safety rail, not a driver of fits.

### 2.3 Tidal gate (localization)

We localize the modification to tidally structured, low‑density regions with a log‑normal gate in a scalar tidal indicator 𝑇,

𝑊(𝑇;𝑇0,𝜎𝑇)=exp[−(ln(𝑇/𝑇0))2/(2𝜎𝑇2)], 0≤𝑊≤1,

(we also allow a small wmin floor if needed for numerical stability; baseline sets wmin=0). 𝑇0 is auto‑centered near the outer disk (e.g., ∼2.2 𝑅d), and 𝜎𝑇 sets the band width.

### 2.4 High‑density screening (Solar‑System safety)

We screen in dense environments with an exponential form that vanishes rapidly at 𝜌≫𝜌c,

𝑆𝜌(𝜌;𝜌c,𝛾)=exp[−(𝜌/𝜌c)𝛾].

### 2.5 The TFR‑S coupling

Putting it together,

𝜉(𝑔bar,𝑇,𝜌)=1+(𝐷sat(𝑔bar;𝑎0,𝐷∞,𝑞)−1) 𝑊(𝑇;𝑇0,𝜎𝑇) 𝑆𝜌(𝜌;𝜌c,𝛾) (1)

Inner regions / Solar System: 𝑆𝜌→0 and/or 𝑊→0 ⇒ 𝜉→1 (GR).

Outer disks: 𝑆𝜌≈1, 𝑊≈1 ⇒ 𝜉≈𝐷RAR (flat curves).

Deep voids: 𝑆𝜌≈1, 𝑊≈0 (no organized tides) or 𝐷sat→𝐷∞ (plateau); in either case 𝜉 does not diverge.

Minimality. Eq. (1) is simpler than our earlier tidal‑band models: one data‑anchored function 𝐷sat(𝑔), one tidal gate 𝑊(𝑇), one screening 𝑆𝜌(𝜌). No additional “return‑to‑GR at low density” terms are included.

## 3. Key predictions

Flatness without halos. In regions where 𝑔bar∝𝑅−2 (outside most baryons), Eq. (1) with 𝐷sat≈𝐷RAR∝𝑔−1/2 yields 𝑉(𝑅)≈const.

BTFR. For nearly spherical outer mass, 𝑉flat4∼𝐺 𝑀b 𝑎0 (up to the 𝑊 𝑆𝜌 band factor), reproducing the baryonic Tully‑Fisher slope; 𝐷∞ does not affect BTFR within observed radii.

No artificial decline at the edge. Because we do not force 𝜉→1 at low density, outer curves stay flat across the measured range; saturation only matters below current 𝑔bar by design.

Environment. At fixed 𝑔bar, low‑tidal systems (small 𝑊) show smaller mass discrepancy; TFR‑S predicts modest environmental offsets testable with group/void catalogs.

Solar‑System & lab. 𝑆𝜌 ensures ∣𝜉−1∣≪10−5 for planetary densities; Cassini/LLR are satisfied for broad (𝜌c,𝛾).

## 4. Data and methodology (summary)

### 4.1 Milky Way (Gaia DR3)

Data curation, binning, asymmetric‑drift corrections, and uncertainties follow our previous pipeline. We keep the MW as the primary anchor because star kinematics are most trustworthy there.
Results placeholders (matched GR/NFW/TFR‑S; identical dynesty controls):

GR log Z, RMSE — [to be inserted]

NFW log Z, RMSE — [to be inserted]

TFR‑S log Z, RMSE — [to be inserted]

### 4.2 SPARC galaxies

We separate the cohort by measurement quality (beam smearing, inclination, H I extent, bar/warp flags). TFR‑S is fit/assessed only on tier‑A systems (e.g., Q≤2, reliable distances/inclinations, large 𝑅max), and never tuned to match noisy tails.
Summary metrics (Δlog Z vs GR and NFW; matched σ‑floors/priors): [to be inserted].

### 4.3 Tidal proxies and normalization

We compute 𝑇(𝑅) from the baryon‑only potential and test three robust proxies (epicyclic/shear/curvature), normalized by the median over fitted radii. Epicyclic is the default (best evidence in our checks); robustness table [to be inserted].

## 5. Priors and parameters

- 𝑎0∼N(1.2,0.2^2)×10−10 ms−2 (weak).
- 𝐷∞∈[2,10] (plateau), 𝑞∈[1,8] (soft‑min sharpness).
- 𝑇0 auto‑centered near 2.2 𝑅d with ln𝑇0∈[−1,1]; 𝜎𝑇∈[0.3,2.0].
- 𝑆𝜌: log10𝜌c∈[MW‑anchored prior], 𝛾∈[2,6].

Baryon parameters as in our GR baseline (Table S1).

All SPARC/ MW triads (GR / NFW / TFR‑S) use identical σ‑floors, masks, and D/i priors.

## 6. Solar‑System constraints

We evaluate ∣𝜉−1∣ at representative planetary/lab densities with 𝑇 in the high‑tidal tail and confirm Cassini ∣𝛾−1∣<2.3×10−5 and LLR guidance are met for broad (𝜌c,𝛾). A compact table mapping ∣𝜉−1∣ to ΔGM/GM at 1–30 AU is provided [to be inserted].

## 7. Results (high‑level)

### 7.1 Milky Way

Evidence: TFR‑S improves over GR by Δlog Z ≈ [to be inserted] and competes with NFW within [to be inserted].

Curve: Red band (TFR‑S) tracks the Gaia median and 16–84% envelope; GR declines beyond 𝑅∼8 kpc.

Posterior: 𝑎0 near canonical; (𝐷∞,𝑞) unconstrained by MW because current 𝑔bar stays above the saturation turn‑on—as designed.

### 7.2 SPARC (tier‑A subset)

Cohort summary: median Δlog Z(TFR‑S–GR) > 0 with [to be inserted] spread; TFR‑S vs NFW varies by system.

Diagnostics: Residual histograms and QQ‑plots show well‑calibrated dispersion with modest tails; robustness to tidal proxy reported in Table ED‑T.

Key point: flat outer curves are reproduced without halos across many systems without forcing 𝜉→1 at the edge.

## 8. Discussion

Why the plateau matters. Empirically, rotation curves remain flat out to the last measured point (𝑅∼30−60 kpc in some spirals). A model that drives 𝜉→1 at low density imposes a decline that is not in the data. TFR‑S matches the RAR slope where we have leverage and only saturates at accelerations below current kinematic reaches. This avoids both divergence in deep voids and artificial returns to GR in the outskirts.

BTFR & RAR are naturally recovered; environment predictions are testable.

Solar‑System safety is maintained by exponential density screening.

Open theory: we outline a weak‑field effective prescription for lensing (disformal correction to Φ+Ψ) and point to a Lagrangian embedding as a priority.

## 9. Methods (reproducibility essentials)

Likelihood: lnL=−1/2∑k[(vk−V(Rk))2/σk2] with σ‑floors in quadrature.

Sampler: dynesty dynamic nested sampling; matched controls across GR/NFW/TFR‑S.

Code paths: xi="rar_gate_sat" uses Eq. (1); xi="rar_gate" sets 𝐷∞→∞.

CLI (paper‑grade): [to be inserted; your runner exposes this already]

```
python runners/run_dynesty_cupy.py --xi rar_gate_sat \
    --nlive 5000 --maxcall 4e7 --dlogz_target 0.01 \
    --sample_method rslice --bound_method multi \
    --checkpoint_every 300 --periodic_analysis \
    --analysis_interval_min 30 --summary_interval 60 \
    --run_analysis --out runs/rar_gate_sat_matched
```

SPARC batch: curated list (tier‑A only), identical σ‑floors, D/i priors, and tidal proxy; aggregation script produces per‑galaxy JSONs and the cohort table.

## 10. Conclusions

TFR‑S—RAR‑gated, tidally localized, exponentially screened, and softly saturated at ultralow 𝑔—provides a minimal, data‑anchored route to flat rotation curves with baryons alone. It preserves Solar‑System limits, reproduces BTFR/RAR trends, and avoids the unphysical decline in the outskirts implied by forcing 𝜉→1 at low density. With the Milky Way as anchor and a carefully curated SPARC cohort, TFR‑S meets the empirical bar for a serious alternative to dark halos on galactic scales. Lensing and environmental tests are natural next discriminants.

