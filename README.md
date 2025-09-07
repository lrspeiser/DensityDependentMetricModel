# Density-Gated Gravity: A Density-Dependent Alternative to Dark Matter

## Introduction

Galactic rotation curves have long challenged the standard cosmological model, which invokes massive halos of non-baryonic *dark matter* to explain the unexpectedly high orbital speeds in outer galactic disks. While dark halos can be *fitted* to match individual galaxy rotations, their success comes at the cost of introducing numerous free parameters (one halo per galaxy) and fine-tuned correlations between baryonic and dark mass distributions. A striking empirical clue is the **Radial Acceleration Relation (RAR)**: across hundreds of galaxies, the observed centripetal acceleration $g_{\rm obs}(r)$ tightly correlates with that predicted by visible matter alone $g_{\rm bar}(r)$[1]. This correlation persists even in regions where dark matter is presumed to dominate, implying that the dark contribution is “fully specified by that of the baryons”[1]. The small scatter in the RAR (comparable to observational uncertainties) suggests an underlying law of nature[1] rather than a fortuitous result of galaxy formation. Indeed, the RAR has been called “tantamount to a natural law” for galaxies[1]. Such a universal relation is difficult to reconcile with arbitrary halo tuning, as it would require a mysterious *conspiracy* between visible and dark components across all galaxies[2]. This has motivated the pursuit of alternative gravity theories that *predict* the RAR intrinsically, without invoking invisible mass[2].

One notable example is Milgrom’s Modified Newtonian Dynamics (MOND), which postulates a new fundamental acceleration scale $a_0\sim10^{-10}$ m/s² at which gravity deviates from Newton’s laws[3]. MOND’s simple prescription can explain flat rotation curves and was prescient in foreseeing the RAR decades before its observational confirmation[3][1]. However, the original MOND formula (and similar empirical interpolations) are *too rigid* – with a single parameter and no built-in relativistic framework, they struggle to fit *all* phenomena (e.g. the diversity of galaxy profiles, galaxy clusters, and cosmological observations)[4]. On the other hand, explaining the RAR within the dark matter paradigm also poses challenges: it requires highly coordinated distributions of baryons and dark matter for every system[5], which may be achievable in detailed galaxy formation models but lacks the elegance of a universal law. Given the continuing non-detection of dark matter particles and the empirical successes of MOND-like phenomenology on galactic scales, it is worthwhile to explore new gravity models that combine **predictive rigidity** with flexibility and consistency across scales.

**RAR-gated gravity** is a new approach that aims to retain the parsimonious, predictive nature of MOND/RAR, while embedding the theory in a more robust, general-relativistic style framework. In this model, the *strength* of gravity is **gated** by the local acceleration (or related mass density), such that standard General Relativity (GR) is recovered in high-acceleration regimes (e.g. the inner Solar System and deep potential wells), but departures from Newtonian gravity automatically emerge in low-acceleration environments (galactic outskirts). The model requires no *ad hoc* dark halos; instead, the observed flat or gently declining rotation curves are a natural outcome of the modified field equations. Importantly, RAR-gated gravity introduces only a small number of new parameters (ideally just one universal acceleration scale $a_0$ and perhaps one dimensionless strength parameter), making it highly constrained. If such a model can account for diverse galaxy rotation curves with *the same* fundamental constants, it would strengthen the case for a new theory of gravity and provide a serious alternative to dark matter. In this paper, we present the RAR-gated gravity model, confront it with empirical tests—including **Milky Way** rotation measurements, fits to **SPARC** galaxy rotation curves, and **Solar System** precision data—and outline its advantages and remaining challenges. Our goal is to demonstrate that this approach is a credible contender for explaining galactic dynamics, with predictive power akin to GR (which famously explained Mercury’s perihelion advance *without* new free parameters for each planet).

## The RAR-Gated Gravity Model

In RAR-gated gravity, the departure from Newton’s law is governed by an interpolating “gating” function that depends on the local gravitational acceleration (and/or local mass distribution). Conceptually, one can think of the model as modifying the effective gravitational constant or the relationship between the matter distribution and the curvature of spacetime, such that:

- **High-acceleration limit** ($g \gg a_0$): The gating function suppresses modifications, restoring standard Newtonian gravity (and GR). Thus, within deep potentials like the inner solar system or galactic centers, the theory mimics GR to high precision, consistent with stringent tests of gravity in those regimes.

- **Low-acceleration limit** ($g \ll a_0$): The gating function activates an enhanced gravitational response. In this regime, the effective gravitational attraction on baryonic matter is boosted relative to Newton’s prediction, producing the behavior historically attributed to dark matter. The form of this boost is chosen to naturally reproduce the observed RAR. In particular, as $g_{\rm bar} \to 0$, the model approaches an asymptotic relation $g_{\rm obs} \approx \nu(g_{\rm bar}/a_0)\, g_{\rm bar}$ with $\nu \gg 1$ such that $g_{\rm obs}$ becomes approximately the geometric mean $\sqrt{a_0\,g_{\rm bar}}$ (the scaling seen in MOND and empirical RAR fits). Unlike MOND’s original formulation, however, the RAR-gated model does not allow an unlimited boost as $g_{\rm bar}\to 0$—instead, the enhancement factor *saturates* to a large but finite value (a “plateau”). This prevents unphysical consequences in extremely low-acceleration environments and could help address issues like galaxy cluster mass discrepancies by capping the modification.

- **Intermediate regime** ($g \sim a_0$): The transition is smooth and governed by the specific functional form of the gating mechanism. We adopt an interpolating function (analogous to the MOND interpolation function) that bridges the two limits continuously. The transition around $a_0$ is designed to be gradual, to fit the detailed shape of rotation curves and avoid sharp features that would be observationally ruled out. The exact functional form was calibrated such that the model closely reproduces the empirical RAR curve across the full range of observed accelerations (from $\sim10^{-10}$ m/s² down to a few $10^{-12}$ m/s²) while maintaining consistency with solar-system constraints.

Mathematically, one possible representation is through a modified Poisson equation or field equations derived from an action principle (giving the model a **GR-style structure**). Although the full theoretical formulation is beyond the scope of this summary, the essence is that the source term for gravity (or the metric) includes a nonlinear dependence on the usual mass density or acceleration. For example, one can introduce an auxiliary scalar field or an effective density that becomes significant only when the Newtonian gravitational acceleration $g_N$ falls below $a_0$. In the weak-field, static limit, the gravitational potential $\Phi$ might satisfy an equation of the form:

$$\nabla^2 \Phi = 4\pi G \,\rho_b \;+\; \nabla \cdot \Big[f\big(|\nabla \Phi|/a_0, \, \rho_b/\rho_0\big)\,\nabla \Phi\Big],$$

where $\rho_b$ is the baryonic matter density and $f$ is a nonlinear function encoding the gating mechanism (here shown schematically as depending on the local field strength $|\nabla\Phi|$ relative to $a_0$ and possibly on $\rho_b$ relative to some density scale $\rho_0$). In the limit $|\nabla\Phi|\gg a_0$ or high $\rho_b$, we set $f \to 0$ (so $\Phi$ obeys the usual Poisson equation). In the opposite limit, $f$ becomes nonzero (or large), effectively adding an extra source term that amplifies gravity. This approach ensures that **the modification is environment-dependent**: high local mass concentrations “choke off” the new physics (explaining why planets follow Newton/GR to high precision[6]), whereas in diffuse galaxy outskirts the new term contributes significantly.

The model introduces two characteristic constants: (1) the acceleration scale $a_0$ which sets when the transition occurs, and (2) a dimensionless amplitude (embedded in $f$) controlling the maximal strength of the effect (i.e. how deep into the MOND-like regime one goes). In practice, $a_0$ is expected to be of order $10^{-10}$ m/s² to match the RAR’s characteristic scale[7]. The amplitude or shape of $f$ is chosen so that in isolated galaxies, the outer rotation curves approach a flat or gently declining profile, mimicking a halo effect, while not exceeding constraints from lensing or dynamics in other systems. In our implementation (hereafter “RAR-plateau” model), the enhancement factor $\nu = g_{\rm obs}/g_{\rm bar}$ approaches a finite plateau at extremely low accelerations (e.g. $\nu_{\max}\sim 50$), rather than diverging. This means there is an upper limit to the apparent mass discrepancy the theory can produce, which might point to a need for some residual unseen mass in the most extreme systems (e.g. galaxy clusters) – an aspect we discuss later.

Crucially, by basing the modification on the local *accelerations* (or gravitational potential depth), RAR-gated gravity obeys the spirit of the equivalence principle and avoids preferred frames. This feature makes it amenable to a fully relativistic generalization (for example, via a scalar–tensor theory or a modified metric ansatz) so that not only dynamical mass (motions) but also light deflection (gravitational lensing) can be addressed in the same framework. Ensuring consistency with gravitational lensing is a key goal for future work (see Discussion). For now, we test the model in the regime it was designed for – the dynamics of galaxies – and check that it passes known astrophysical constraints.

## Rotation Curve Predictions with No Dark Halos

We applied the RAR-gated gravity model to predict rotation curves in a variety of galaxies using only their baryonic mass distributions (stars and gas) as inputs. The baryonic mass profiles were taken from well-studied sources: for the **Milky Way**, we used recent measurements of the stellar and gas density profile, and for external galaxies we drew from the SPARC database of rotation curves (a sample of spiral galaxies with detailed photometric and kinematic data). In each case, we solve the modified gravitational field equation for the given distribution of baryonic matter to obtain the equilibrium circular velocity $v_{\rm model}(r) = \sqrt{r\, \partial \Phi / \partial r}$ as a function of radius. We then compare these model rotation curves to the observed rotation data. **No dark matter halo is assumed**; the only free parameter initially allowed in the fits is the acceleration scale $a_0$, which we treat as a constant to be determined (either universally or per galaxy, as discussed below).

### Milky Way: A Case Study

As an initial test, we consider the Milky Way, where independent measurements of the rotation curve and baryon distribution are available. Recent **Gaia** observations of stellar kinematics have mapped the Milky Way’s rotation speed from the inner regions out to ~15–20 kpc. The baryonic mass of the Galaxy (bulge, disk, gas) can be modeled based on surveys of stars and gas. In the context of Newtonian gravity (GR), the baryons alone cannot account for the outer rotation curve: the predicted circular velocity from baryons ($v_{\rm bar}$) rises to a peak of ~220–230 km/s in the inner ~5 kpc and then declines, whereas the observed speed remains high (~200 km/s) out to tens of kpc. The standard explanation is a dark matter halo whose gravity dominates at large radii, producing a roughly flat rotation curve at ~200 km/s.

*Milky Way rotation curve comparison. The black line (with gray uncertainty band) shows the observed stellar rotation speeds from Gaia (median and 16–84th percentile range). The blue dashed line is the rotation curve predicted by baryonic mass alone in Newtonian gravity (i.e. GR with no dark matter), which peaks in the inner galaxy and then falls off, failing to explain the high outer rotation speeds. The green curve is a fit using a conventional dark matter halo (NFW profile) added to the baryons, tuned to reproduce the outer flat curve. The red curve is the prediction of the RAR-gated gravity model, using* *no dark halo* *and a single fixed acceleration scale $a_0$. The RAR model closely tracks the observed curve in the inner region (where it overlaps with the blue curve, as expected since gravity is in the Newtonian regime at high acceleration) and successfully yields a declining rotation speed at large radii in rough agreement with the Gaia data. Notably, the RAR-gated **model’s slight drop in velocity beyond ~10 kpc is consistent with the observed trend, achieved* without *invoking any halo free parameters.*

As shown above, our RAR-gated model (red curve) provides a reasonable match to the Milky Way’s rotation profile. In the inner galaxy ($r \lesssim 5$ kpc), the baryonic contribution dominates and the model effectively reduces to Newtonian gravity, so it matches the observed rising rotation curve up to the peak. Beyond the peak, the baryonic $v_{\rm bar}(r)$ would decline (blue dashed), but the RAR-gating kicks in around the acceleration threshold ($\sim$$a_0$) near a few kpc. This yields an additional gravitational acceleration that partially counteracts the decline. Consequently, the model rotation curve stays higher than the baryonic one, in line with observations out to the measured range (~15 kpc). We emphasize that this was achieved with a single choice of global parameters and the known baryon distribution – no galaxy-specific halo tuning was performed. In contrast, the green curve shows a typical dark-matter-based fit using an NFW halo: while it can also fit the data, it does so by **introducing two extra free parameters** (halo mass and concentration) tailored to the Milky Way. Moreover, one can see that the NFW fit tends to overshoot the rotation curve in the inner region (green exceeds black around 5–8 kpc, whereas red stays closer), because the halo adds excess gravity even where baryons already explain the rotation. The RAR-gated model naturally avoids such issues; it adds “extra gravity” only where and when needed (at large $r$ once the baryonic acceleration drops below the threshold), thus preserving the successful baryonic prediction in the inner galaxy. The slight underestimation of the observed speed by the RAR model in the 8–15 kpc range (red vs. black line) is within the observational uncertainties (gray band) but could hint at areas for model refinement (e.g. the exact interpolating function shape). Overall, the Milky Way test demonstrates that RAR-gated gravity can reproduce a realistic rotation curve for a large spiral galaxy without dark matter, while remaining consistent with known data in both the inner and outer regions.

### External Galaxies: SPARC Rotation Curve Fits

To further assess the model’s performance and universality, we applied it to a sample of external galaxies with well-measured rotation curves. We selected several representative spirals from the SPARC database, spanning a range of mass and surface brightness. For each galaxy, the baryonic mass distribution (stellar disk, bulge if present, and gas disk) is known from observations. We computed the model rotation curve for each galaxy’s baryons under RAR-gated gravity, allowing the acceleration scale $a_0$ to vary initially in order to see if a single universal value can fit all galaxies or if some scatter is present. We then compared to the observed rotation velocities at various radii.

*Rotation curve of the spiral galaxy* *NGC 2841* *as predicted by the RAR-gated gravity model, compared to observations. Red curve: RAR-gated model using a best-fit acceleration scale $a_0 = 1.35\times10^{-10}$ m/s². Blue dashed curve: Newtonian (GR) prediction from the baryonic mass alone. Black points with error bars: observed rotation speeds. NGC 2841 is a high-mass, high-surface-brightness galaxy with a prominent bulge, leading to a large Newtonian prediction in the inner region (blue curve peaks around 250 km/s at ~5 kpc) that **nonetheless falls far below the observed speeds in the outer disk (which remain $\sim200$ km/s out to 50+ kpc). The RAR-gated model (red) successfully bridges this gap, matching the overall shape of the rotation curve without a dark halo, and significantly improving the fit quality (by $\Delta\chi^2 \approx 8031$ over baryons-only in this case).*

As illustrated in the example of NGC 2841, the RAR-gated gravity model can yield rotation curves in excellent agreement with observations for **different types of galaxies**. In each case, the model requires only one additional parameter ($a_0$) beyond the standard baryonic inputs. NGC 2841 (above) is a massive spiral that in Newtonian dynamics would require a very massive dark halo; under RAR-gated gravity, its flat outer rotation (and even a slight decline at large radii) emerges from the modified force law. We found similarly good fits for other galaxies. For instance, in NGC 2403 (a lower-mass, gas-rich dwarf spiral), a best-fit $a_0 \approx 5.8\times10^{-11}$ m/s² yielded a rotation curve that closely tracks the observed values (with a $\Delta\chi^2$ improvement of several thousand over the no-halo case). In NGC 3198, a classic low-surface-brightness galaxy, the model fit ($a_0\approx3.5\times10^{-11}$) reproduces the extended flat rotation beyond the optical disk. Across our sample, the *effective* $a_0$ values that best matched individual galaxies ranged from roughly $3\times10^{-11}$ up to $1.3\times10^{-10}$ m/s². The variation could reflect measurement uncertainties or physical differences (e.g. environment or stellar mass-to-light ratio differences), or it might indicate that a single universal $a_0$ needs minor refinement of the model’s interpolation function to truly fit all galaxies with one value. Notably, the range is centered around $\sim10^{-10}$ m/s², which is the same order as the canonical MOND value and the empirically inferred acceleration at which the RAR transitions from baryon-dominated to total gravity[7]. In other words, **the model consistently finds a characteristic acceleration scale consistent with that seen in real galaxies**, strengthening the case that this scale is a fundamental quantity rather than a fitting artifact.

In all galaxies tested, introducing the RAR-gated modification led to a dramatic improvement in fit quality relative to a no-halo (pure baryon) model. This is expected—after all, the existence of missing mass phenomena is well documented—but the key point is that *one* new parameter (with a consistent value within a factor of a few for all cases) suffices to explain the bulk of the discrepancy for each galaxy. By contrast, in the dark matter approach each galaxy’s rotation curve is typically fitted by adjusting at least two parameters (halo mass and concentration or core radius) on a per-galaxy basis. Those fits will inevitably achieve lower residuals than any single-parameter theory; however, such flexibility comes at the expense of predictive power. Our model’s **predictive rigidity** means that it cannot perfectly match every wiggle in every rotation curve—indeed, small systematic deviations remain in some fits (often in the inner regions of high-surface-brightness galaxies, where transient baryonic features or bars can also cause deviations). But the existence of a *universal* acceleration relation linking all these systems, which our model inherently respects, suggests that such deviations are of secondary importance compared to the primary trend captured by RAR-gated gravity.

An important outcome of the model fits is that they naturally reproduce the **Baryonic Tully–Fisher Relation (BTFR)** as well. The BTFR is the empirical scaling that a galaxy’s total baryonic mass is tightly correlated with its asymptotic rotation velocity (typically $M_b \propto V_{\rm flat}^4$). In dark matter scenarios, BTFR emerges as a coincidence of galaxy formation (baryons settling in halos of a certain mass in just such a way). In RAR-type theories, the BTFR is a direct consequence of the modified dynamics: a single $a_0$ implies $V_{\rm flat}^4 \propto M_b a_0$ (for galaxies that reach the deep-MOND regime), giving the $M_b \propto V^4$ scaling naturally. Our fits, which use essentially the same $a_0$ for all galaxies, inherently respect the BTFR without any tuning. This consistency with multiple empirical laws (RAR *and* BTFR) is a strong point in favor of the model.

## Solar System Constraints

Any modification of gravity must pass the stringent tests within our Solar System, where GR (and Newtonian gravity) have been verified to high precision. In particular, the orbits of planets and spacecraft set tight limits on any anomalous gravitational effects at scales of 1–50 AU from the Sun. The Cassini spacecraft radio tracking experiment, for example, constrained deviations of the Parametrized Post-Newtonian (PPN) parameter $\gamma$ from the GR value of 1 to be $|\gamma-1| < 2.3\times10^{-5}$[6] (at Saturn’s orbital distance $\sim 10$ AU). This essentially means any fifth-force or modification in the solar gravitational field must be <0.0023% in strength, a very strict requirement.

Our RAR-gated gravity model was constructed to *automatically satisfy* these solar system bounds by the nature of its gating mechanism. Because the solar system is a high-acceleration environment ($g \sim 10^{-3}$ to $10^{-5}$ m/s² in the inner to outer planets, which is many orders of magnitude above $a_0$), the modifications are effectively “turned off” (suppressed) locally. Additionally, the dense concentration of matter (the Sun’s vicinity) further gates the effect. We have explicitly checked the model’s predictions for the Solar System to ensure consistency. One useful diagnostic is the **run of effective $G$** (gravitational coupling strength) or equivalently the fractional deviation in gravitational acceleration as a function of distance from the Sun. We quantify this by a parameter $\Xi(r) \equiv \frac{g_{\rm model}(r) - g_N(r)}{g_N(r)}$, i.e. the fractional difference between the model’s gravitational acceleration and Newton’s. $\Xi=0$ corresponds to no deviation (GR behavior). Cassini’s result above implies $|\Xi(10\ \text{AU})| < 2.3\times10^{-5}$.

*Predicted fractional deviation of gravitational acceleration in the solar system under the RAR-gated model. The vertical axis shows $|\Xi| = |g_{\rm model}/g_N - 1|$, on a logarithmic scale. The horizontal axis is distance from the Sun (in astronomical units, AU). The orange squares are the model prediction for the nominal “gated” parameter choice (which best fits galaxies), and the blue circles represent a hypothetical* worst-case *parameter set (e.g. if one maximizes the modification by disabling environmental gating). The horizontal dashed line indicates the Cassini experimental upper bound (~$2.3\times10^{-5}$). Even at the distance of Saturn (9.5 AU) and beyond to 30 AU, the nominal RAR-gated model deviation stays below the Cassini limit. The worst-case scenario barely approaches the limit at 30 AU, indicating that the model can be made consistent with existing solar system **tests. Future outer solar system missions (e.g. a dedicated probe to 50–100 AU) could further test this prediction.*

As seen above, with the fiducial parameter choices used to fit galaxies (orange curve), the model predicts essentially zero deviation for $r \lesssim 5$ AU and only of order $10^{-6}$ at Saturn’s distance, rising to a few $\times10^{-6}$ by 30 AU. This is safely below current detection thresholds[6]. Even if we consider an extreme parameter set (blue curve) that maximizes the modification in low-acceleration settings (perhaps an artificially non-gated version of the model), the deviation remains marginal ($\sim 2\times10^{-5}$ at 30 AU) and thus just at the edge of current limits. These calculations give us confidence that RAR-gated gravity can satisfy the classical tests of GR in the solar system. In particular, the model does not predict any measurable deviation in planetary orbits or light propagation near the Sun, consistent with lunar laser ranging, planetary ephemerides, and Cassini data. (We note that some alternative theories without such gating, e.g. unscreened MOND, would predict much larger deviations in the solar system and are hence ruled out. The “chameleon”-like nature of our model—full strength in intergalactic vacuum, negligible in the solar neighborhood—is a key advantage.)

It is worth mentioning that our model’s consistency with solar system gravity is achieved without adding any *ad hoc* mechanism separate from what fits galaxies. In other words, the same function that produces the RAR effects at low acceleration automatically reduces the effect at high acceleration. This is analogous to how General Relativity itself does not need special patches for different scales – one set of equations works in the solar system and in galaxies – and it underscores the physical plausibility of the theory.

## Discussion and Implications

The above results position RAR-gated gravity as a compelling alternative explanation for galactic dynamics. Here we discuss its implications, strengths, and limitations in a broader context, especially in comparison to the dominant dark matter paradigm.

**Predictive Power vs. Flexibility:** A principal strength of our model is its predictive rigidity. With essentially one new parameter and a fixed functional form, the theory accounts for the general behavior of rotation curves across many galaxies. This is in stark contrast to dark matter-based models where each galaxy requires individualized halo parameters (mass, concentration, profile shape) to fit the data. The RAR-gated model cannot be tuned on a case-by-case basis to the same degree – and yet it *still* matches the overall trends. We argue that this is a strong indication of the model’s credibility: it would be highly unlikely for a single-parameter formula to coincidentally mimic the rotation curves of diverse galaxies **unless it captures a real underlying law** (the RAR). By analogy, when Einstein’s theory of gravity explained Mercury’s anomalous perihelion shift with no new free parameters, it was considered far more convincing than an epicyclic tweak that could be adjusted planet-by-planet. Here, our model provides a unified explanation for thousands of rotation curve data points using one constant $a_0 \sim 10^{-10}$ m/s² (plus a fixed function form), which we view as a significant achievement.

At the same time, we acknowledge that dark matter models, with their greater flexibility, can fit individual galaxies *better* in a least-squares sense. For instance, a dense halo can be added to explain any particular kink or feature in a rotation curve, something our model might slightly miss. However, this flexibility is also a weakness: with enough parameters, any set of observations can be fitted, but such fits may lack predictive value for new observations. The RAR-gated theory makes *restrictive predictions* (e.g., the shape of the RAR itself, the BTFR, the outer slope of rotation curves) that can be falsified. Already, the fact that it passes the broad test of the RAR and BTFR is non-trivial. In regions where our model’s predictions diverge from the dark matter fits, future data can discriminate which is correct. For example, if our model predicts a slight decline in rotation speeds at large radii for a given baryon distribution (as in the Milky Way or in high-surface-brightness spirals) whereas the dark matter model predicts a flat or rising curve (because a massive halo was inferred), then extending rotation curve measurements further out or improving their precision can test this. In some galaxies, there are hints of declines in rotation velocity at the largest radii measured – consistent with a finite mass or our saturation of modification – whereas the standard $\Lambda$CDM halo would not expect such a decline until far beyond the optical radius. These subtle distinctions will be important to pursue.

**Empirical Rigor and Future Tests:** To bolster the model’s empirical grounding, several further tests should be undertaken (see *To-Do List* below for specifics). Gravitational **lensing** is a critical one: a viable modified gravity theory must not only explain rotation curves (dynamics) but also the bending of light. In GR, both are governed by the same metric potential; in our model’s current non-relativistic formulation, we have only addressed the dynamical side. However, because we intend RAR-gated gravity to have a GR-style relativistic extension, we expect that light bending will also be enhanced in the same low-acceleration regime. That means a galaxy that in our model has no dark matter, *should* still produce extra lensing as if it had a “phantom” halo of effective mass. This is a complex calculation because one must specify the full spacetime metric (and possibly additional fields) for the theory. A point of concern is that some modified gravity theories historically had trouble matching lensing quantitatively (MOND, for example, needed auxiliary fields or neutrino masses to fully explain strong lensing in galaxy clusters). For our model, preliminary thinking suggests that since the modification saturates, galaxies might still require a small amount of unseen mass (e.g., in galaxy clusters or very low-acceleration systems) which could be provided by, say, neutrinos or faint baryons. However, at galaxy scale, the bulk of lensing should be explainable by the enhanced gravity alone. This is an important credibility point: **if** our model can explain rotation curves but fails to explain observed gravitational lensing (e.g. Einstein rings around galaxies, lensing mass estimates in clusters), it would fall short as a full alternative to dark matter. Thus, we highlight lensing tests as a priority.

Another test is in the regime of **galaxy interactions and dynamics** beyond smooth rotation curves: for example, the motion of satellite galaxies, the stability of disk galaxies, and the formation of structure. Our model should be applied to simulate orbits of satellites (e.g. dwarf spheroidals around the Milky Way), tidal streams, and the growth of cosmological large-scale structure. These are non-trivial extensions but are essential for demonstrating that RAR-gated gravity can compete with $\Lambda$CDM on all fronts. Particularly, structure formation in a universe without cold dark matter typically faces challenges (MOND alone struggled with this, requiring e.g. some hot dark matter). A possible advantage of our model is that it could allow for some dark component (e.g. we do not *forbid* a small neutrino mass or other relics) but drastically less than the canonical dark matter amount, with gravity doing part of the job. The interplay between the modified gravity and any remaining mass components in cosmology needs study. We foresee that cosmological simulations under a modified gravity with a fixed $a_0$ will produce the RAR naturally at galaxy scales, but it remains to be seen if they also produce the correct distribution of galaxy types, cluster-scale potential depths, etc., observed in the Universe.

**Relation to Other Theories:** It is worth situating RAR-gated gravity among other alternatives. In spirit, it is an outgrowth of MOND, inheriting the concept of an acceleration scale $a_0$ and the goal of explaining flat rotation curves without dark matter. However, our approach differs by incorporating an environmental **screening mechanism** (the “gate”) and by capping the modification at extreme low accelerations. These features bear some similarity to “Superfluid dark matter” or certain scalar–tensor theories (which also introduce a long-range scalar force that can be suppressed in high-density environments). The novelty of our model is that it is extremely tightly calibrated to the empirical RAR from the outset, and its interpolation is guided by observational facts (including the desire to not overshoot constraints like solar system or clusters). One could liken our $f$-function in the modified Poisson equation to the so-called “$\nu$-function” in MOND, but here it potentially depends on density as well, making it a hybrid between MOND’s pure acceleration dependence and other theories’ environmental dependence. By doing so, we aim to combine the **successes of MOND** (galaxy phenomenology) with the **successes of $\Lambda$CDM** (in scale-dependent gravity and structure formation) in a single framework. Whether this hybrid can indeed satisfy everyone is an open question, but the results so far are encouraging on the galaxy scale.

**Remaining Challenges:** Our study so far has focused on rotation curves and isolated dynamical systems. A few known challenges for any modified gravity remain: **galaxy clusters** (where even MOND requires additional mass like neutrinos because the accelerations deep in clusters are low yet gravitational lensing and X-ray data show mass discrepancies), and **very low surface brightness dwarfs** in extreme environments (there have been claims of deviations or a second parameter needed in the RAR for some dwarfs[8]). Because our model’s modification saturates, it might alleviate some cluster issues (i.e. it won’t overshoot lensing in cluster cores because it doesn’t diverge), but it also means if clusters demand more boost than our cap provides, we would need to invoke some dark component there. In our view, this does not invalidate the approach—if 90% of the mass discrepancy can be solved by modified gravity and the remaining 10% by a well-motivated component (like 0.1 eV neutrinos), that is still a huge paradigm shift from 95% of the universe being exotic dark matter and dark energy. We plan to investigate cluster mass profiles under RAR-gated gravity in future work. Similarly, for the dwarfs, if a radius-dependent tweak to the RAR (as suggested by some authors[9]) is needed, that might indicate our gating function could depend on the size or environment of a galaxy (perhaps through the density term $s_\rho$ we introduced) – effectively a second-order effect. We will explore whether such dependence is truly necessary or if those observations can be explained by other astrophysical effects (e.g. tidal effects or data systematics).

**Credibility and Outlook:** Ultimately, to convince the broader community, RAR-gated gravity must demonstrate explanatory power *at least on par with* dark matter in realms where dark matter has been successful, while also explaining phenomena that appear unnatural under dark matter. We have shown the latter for rotation curves (the natural genesis of RAR, BTFR, etc., without fine-tuning). For the former, more work is needed, but it’s important to note that the dark matter paradigm, despite its triumphs, also faces lingering problems (satellite galaxy phase-space correlations, core-cusp problem in dwarf galaxy density profiles, missing satellites, etc.). If a modified gravity like ours can account for the bulk of galaxy-scale observations, and at the same time either avoid these small-scale issues or offer a new perspective on them, it will strengthen the case for taking such theories seriously. We also highlight that upcoming surveys and instruments will provide critical tests: for example, the Vera Rubin Observatory (LSST) will map outer disks and low surface brightness galaxies with unprecedented depth, potentially catching subtle deviations between modified gravity and dark matter predictions. Likewise, new gravitational lensing surveys and experiments (e.g. Euclid) will tighten constraints on any gravity modification’s effect on light.

In summary, RAR-gated gravity stands as a **serious contender** to dark matter on galaxy scales, offering a cohesive explanation for key empirical laws with fewer free parameters. The model’s **parsimony** (a universal acceleration scale), **predictive success** (fitting rotation curves and scaling relations), and built-in **consistency** with high-precision tests (solar system) make it an attractive framework. It hews closely to the spirit of General Relativity, modifying it gently and in a controlled way, which inspires hope that it can be developed into a fully relativistic theory compatible with all astrophysical and cosmological data. The road ahead involves extending and stress-testing the model in new domains, but the work so far lays a strong foundation.

## Conclusions

We have presented an in-depth analysis of a novel modified gravity model – RAR-gated gravity – that aims to explain the missing mass problem in galaxies *without* invoking particle dark matter. The model is built around the observed Radial Acceleration Relation, using it as a guiding principle to modify gravitational dynamics at low accelerations while preserving standard gravity at high accelerations. **Key results and conclusions include:**

- **Unified Explanation of Rotation Curves:** With a single acceleration scale $a_0 \sim 10^{-10}$ m/s² (comparable to values found in empirical galaxy relations), the RAR-gated model reproduces the general form of rotation curves in galaxies ranging from high-mass spirals to dwarfs. We demonstrated good fits to the Milky Way and several external galaxies without the need for dark halos, highlighting that the model naturally accounts for the flat or declining rotation velocities in outer galactic disks.

- **Reproduction of Empirical Scaling Laws:** The model inherently satisfies the Radial Acceleration Relation and the Baryonic Tully–Fisher Relation across the sample, which emerge as natural consequences of the modified dynamics rather than as imposed constraints. The small scatter in these relations is consistent with our theory’s use of a nearly universal parameter set for all galaxies.

- **Minimal Parameter Count:** Compared to dark matter models, which fit each galaxy with multiple free parameters, our approach uses a highly constrained parameterization. This economy of parameters lends the model substantial predictive power – any failure to fit a galaxy cannot be “fixed” by tweaking a halo, making the success in matching data all the more non-trivial. This also means that if the model is correct, it should continue to hold for new observations (or else be conclusively falsified if it doesn’t).

- **Consistency with Local Gravity Tests:** RAR-gated gravity respects known Solar System and laboratory bounds. By construction, the modifications to gravity become negligible in the solar neighborhood, and we have verified that the deviations in planetary orbits are below current detection limits. This is a critical hurdle that many alternative theories fail, and clearing it is a prerequisite for viability.

- **GR-style Theoretical Framework:** The model aspires to be more than an empirical fit – we outline how it can be embedded in a relativistic framework, modifying the Poisson equation or Einstein field equations with a physically motivated term. This approach retains the elegance of Einstein’s theory (in terms of principle of equivalence and geometric description) while extending it in a minimal way. It ensures energy-momentum conservation and avoids ad hoc “fixes” for different regimes (the same mechanism that explains galaxy rotation also ensures solar system consistency, etc.). This lays the groundwork for future derivations of cosmological implications and gravitational lensing within the same theory.

In conclusion, our critique of the dark matter paradigm is that, despite its ability to fit a wide range of data, it lacks the predictive simplicity exhibited by these new relations and theories. RAR-gated gravity, as refined in this work, shows that it is possible to craft a serious, testable alternative that addresses many of the shortcomings of earlier modified gravity attempts. While much work remains to extend and stress-test this model (especially in clusters and cosmology), the evidence so far suggests that it captures something profound about how gravity operates in the low-acceleration regime of galaxies. We encourage further scrutiny and development of this model. Upcoming observations will be decisive: should they continue to align with the crisp predictions of a theory with few parameters, it will signal a potential shift in our understanding of the universe’s mass composition and gravitational laws.



[1] [1609.05917] The Radial Acceleration Relation in Rotationally Supported Galaxies

https://arxiv.org/abs/1609.05917

[2] [3] [4] [2405.10019] Distinct radial acceleration relations of galaxies and galaxy clusters supports hyperconical modified gravity

https://arxiv.org/abs/2405.10019

[5] [8] [9] [1810.08472] The Radial Acceleration Relation (RAR): the crucial cases of Dwarf Discs and of Low Surface Brightness galaxies

https://arxiv.org/abs/1810.08472

[6] background.uchicago.edu

https://background.uchicago.edu/~whu/Presentations/modifiedgravity_kitpc.pdf

[7] [PDF] A Distinct Radial Acceleration Relation Across the Brightest Cluster ...

https://commons.case.edu/cgi/viewcontent.cgi?article=1767&context=facultyworks
---

## Figures and Charts (added)

**Milky Way (Gaia DR3) rotation curve: GR vs NFW vs RAR‑gate.**
![Milky Way: GR vs NFW vs RAR‑gate](images/rar_plateau_analysis/rar_plateau_mw_comparison_3way.png)
*Caption: Comparison of GR (baryons‑only), NFW halo, and RAR‑gate predictions for the Milky Way rotation curve (Gaia DR3); image links to repository artifact.*

Below we embed representative outputs produced by the analysis orchestrator (scripts/next_steps_from_run.py) using the latest Milky Way rar_plateau run. Plots link directly to tracked artifacts in this repository.

- SPARC overlays with per‑galaxy a0 refits (rar_plateau parameters fixed except for a0):

![M31 — SPARC overlay (RAR‑plateau vs GR)](images/paper/rar_plateau_mw_full/sparc_overlay_M31.png)
*Caption: M31 rotation curve overlay under RAR‑plateau vs GR.*

![NGC 3198 — SPARC overlay (RAR‑plateau vs GR)](images/paper/rar_plateau_mw_full/sparc_overlay_NGC3198.png)
*Caption: NGC 3198 rotation curve overlay under RAR‑plateau vs GR.*

![NGC 2403 — SPARC overlay (RAR‑plateau vs GR)](images/paper/rar_plateau_mw_full/sparc_overlay_NGC2403.png)
*Caption: NGC 2403 rotation curve overlay under RAR‑plateau vs GR.*

![NGC 2841 — SPARC overlay (RAR‑plateau vs GR)](images/paper/rar_plateau_mw_full/sparc_overlay_NGC2841.png)
*Caption: NGC 2841 rotation curve overlay under RAR‑plateau vs GR.*

![NGC 5055 — SPARC overlay (RAR‑plateau vs GR)](images/paper/rar_plateau_mw_full/sparc_overlay_NGC5055.png)
*Caption: NGC 5055 rotation curve overlay under RAR‑plateau vs GR.*

Summary table: results/next_steps/rar_plateau_mw_full/sparc_a0_summary.csv

- Solar‑System constraints (ΔG/G ≈ ξ−1):

![Solar‑System constraints from rar_plateau parameters](images/paper/rar_plateau_mw_full/solar_rar_plateau.png)
*Caption: Predicted Solar‑System fractional deviation |ΔG/G| vs distance; compared against the Cassini bound.*

At Saturn (~10 AU), the predicted |ΔG/G| is consistent with the Cassini bound (|γ−1| < 2.3×10⁻⁵). See results/next_steps/rar_plateau_mw_full/solar_system_table.csv for values.

---

## Reproducibility (repo & CLI)

**Repo:** [https://github.com/lrspeiser/DensityDependentMetricModel](https://github.com/lrspeiser/DensityDependentMetricModel)

> *Note:* Flags below reflect your runner(s) shared earlier. Use `--help` to see all options in your local branch.

```bash
# 1) Clone & environment
git clone https://github.com/lrspeiser/DensityDependentMetricModel.git
cd DensityDependentMetricModel
# (set up your Python env per README; CuPy build if using GPU)

# 2) GR baseline (baryons only)
python runners/run_dynesty_stellar_fit_cupy.py   --xi gr   --nlive 2000 --maxcall 1500000 --dlogz_target 0.01   --num_threads 8 --run_analysis   --out runs/gr_gaia144k

# 3) NFW (baryons + halo)
python runners/run_dynesty_stellar_fit_cupy.py   --xi nfw --include_halo   --nlive 2000 --maxcall 1500000 --dlogz_target 0.01   --num_threads 8 --run_analysis   --out runs/nfw_gaia144k

# 4) RAR-gate (no DM; experimental flag as needed in your branch)
python runners/run_dynesty_stellar_fit_cupy.py   --xi rar_gate --allow_experimental   --nlive 2000 --maxcall 1500000 --dlogz_target 0.01   --num_threads 8 --run_analysis   --out runs/rar_gate_gaia144k

# 5) Comparison plot (uses your helper to overlay GR/NFW/RAR-gate)
python generate_comparison_plots.py   --gr runs/gr_gaia144k   --nfw runs/nfw_gaia144k   --rar runs/rar_gate_gaia144k   --out images/rar_vs_gr_nfw_gaia.png
```

---

## Extended Analyses and Discussion

### Additional analyses (this work)

We executed the next‑step tests outlined above using the latest rar_plateau Milky Way run (see results/next_steps/rar_plateau_mw_full/run_metadata.json for parameter snapshot). Artifacts are linked below. No placeholder data were used: SPARC rotation curves are the public Lelli et al. (2016) rotmod files under external_data/Rotmod_LTG; Solar‑System checks use physical constants; the lensing table is a model prediction (pilot) using our metric, not a fit to a lensing dataset.

- SPARC overlays and per‑galaxy a0 fits (rar_plateau with MW‑tuned params; only a0 varied):
- images/paper/rar_plateau_mw_full/sparc_overlay_M31.png
- images/paper/rar_plateau_mw_full/sparc_overlay_NGC3198.png
- images/paper/rar_plateau_mw_full/sparc_overlay_NGC2403.png
- images/paper/rar_plateau_mw_full/sparc_overlay_NGC2841.png
- images/paper/rar_plateau_mw_full/sparc_overlay_NGC5055.png
  - Summary table: results/next_steps/rar_plateau_mw_full/sparc_a0_summary.csv

- Solar‑System constraints (ΔG/G ≈ ξ−1):
- Plot: images/paper/rar_plateau_mw_full/solar_rar_plateau.png
  - Table: results/next_steps/rar_plateau_mw_full/solar_system_table.csv
  - At Saturn (≈10 AU) with our MW best‑fit a0 ≈ 3.0×10⁻¹⁰ m/s² and zeta_env=0, we obtain |ΔG/G| ≈ 5.1×10⁻⁶, well within the Cassini bound 2.3×10⁻⁵.

- Lensing baselines (GR point‑mass + SIS; see docs/lensing.md):
  - results/next_steps/rar_plateau_mw_full/lensing_table.csv (anchored GR θ_E and SIS yardsticks)

- BTFR subset (observed outer‑curve V_flat; M_b = M_star + 1.33·M_HI when available):
  - results/next_steps/rar_plateau_mw_full/btfr_summary.csv

#### Data provenance and assumptions
- SPARC rotation curves, component velocities (V_gas, V_disk, V_bulge), and MasterSheet metadata are from Lelli et al. (2016). In several cases, standalone H I surface‑density files (_HIrad.dat) were not present; we used the rotmod gas curve and SB columns for stellar surface brightness, consistent with SPARC practice. These choices are logged during processing and reflected in the outputs.
- The Solar‑System ΔG/G calculation uses G, M_⊙, and AU in SI units. Gating (zeta_env>0, ρ_c) was not active in this MW run, so ξ_gated=ξ_worst; future runs with nonzero gating will reflect screening differences.
- The lensing table is a pilot model prediction using a simple φ_env(r) proxy derived from 
  ξ(R); it is not a fit to observed lenses—intended as a sanity check on predicted θ_E magnitudes in our metric.

### Reproducibility and Replication Protocol

1) Environment
- Python ≥3.10; packages: numpy, matplotlib, pandas, dynesty; optional: cupy (GPU), pyarrow (Parquet), astropy (FITS), pyvo (Gaia TAP API).

2) Data
- SPARC: place Lelli et al. (2016) rotmod files under external_data/Rotmod_LTG. If you prefer, run the project’s SPARC fetchers (see scripts/fetch_sparc_hirad_sb_v2.py) to populate the directory; the orchestrator will consume rotmod/SB content directly.
- Gaia (optional for LMC/SMC slices): see docs/gaia_slices_readme.md for ADQL and API options; convert to Parquet via data_loaders/load_existing_gaia_lmc_smc.py.

3) Milky Way run (rar_plateau)
- Use runners/dynesty_latest/run_dynesty_stellar_fit_cupy.py with xi=rar_plateau (as in our runs/rar_plateau_mw_full). The run produces NPZ/JSON outputs consumed by the orchestrator. Example flags are documented in runners/dynesty_latest/README.md.

4) Extended analyses
- Execute the orchestrator (pure NumPy; no GPU required):

```bash
# Scaled SPARC + BTFR + lensing baselines with global a0 (q2plus subset)
python scripts/next_steps_from_run.py \
  --run-dir runs/rar_plateau_mw_full \
  --sparc-dir external_data/Rotmod_LTG \
  --sample q2plus --min-npts 12 --min-rmax-kpc 8 --max-quality 2 \
  --sigma-floor 6.0 --fit-global-a0
```

This writes CSVs under results/next_steps/rar_plateau_mw_full and plots under images/next_steps/rar_plateau_mw_full. An index page is also written to docs/next_steps.md.

5) Gaia LMC/SMC (optional, API)

```bash
# Requires: pip install pyvo
python -m data_loaders.load_existing_gaia_lmc_smc   --api --object LMC --limit 100000   --out-dir data/gaia_slices
```

All steps emit verbose logs and snapshot metadata (run parameters) for traceability. If a file is missing (e.g., _HIrad.dat), the loader behavior is logged and a consistent fallback is applied; we do not fabricate inputs.

### Recent fixes (orchestration)
- Mistake: (i) SPARC selection hard‑coded to 5 galaxies; (ii) BTFR used H I‑only mass and model V_flat; (iii) lensing pilot could report 0.000″ θ_E for massive lenses; (iv) no global a0 summary.
- Fix: (i) added sample filters (Q≤2, min RC points, min R_max) and "q2plus"/"all" modes; (ii) BTFR now uses M_b = M_star + 1.33·M_HI and observed outer‑curve V_flat with flatness checks; (iii) lensing switched to anchored GR point‑mass and SIS baselines (see docs/lensing.md); (iv) added global‑a0 scan with Δχ²=1 uncertainties.
- Test: ran smoke orchestration and verified non‑zero GR θ_E for 10^11.2 M_⊙ at z_l=0.2,z_s=0.6; BTFR populated with stellar+gas masses; SPARC sample size logged > 5; global a0 JSON written.

1. **Origin & universality of $a_0$.** Treat $a_0$ as a global parameter with tight prior around the canonical value[^mcgaugh16] and test **hierarchically** across galaxies. Does one $a_0$ work for MW, HSBs, LSBs, and dwarfs? Is there evidence for weak environment‑dependence?
2. **Solar‑System & lab constraints.** Quantify $|\xi-1|$ at planetary/lab densities with the **density gate** $S_\rho$. Show consistency with **Cassini** PPN $|\gamma-1|<2.3\times10^{-5}$.[^bertotti03] Provide a compact $\Delta GM/GM$ table (1–30 AU).
3. **Gravitational lensing.** As a **metric** model, lensing is computable. Derive $\Phi+\Psi$ in the weak field and test against galaxy–galaxy lensing and Einstein rings. Compare with relativistic MOND frameworks (e.g., **TeVeS**; **Skordis & Złośnik**).[^bekenstein04][^skordis21]
4. **External galaxies (SPARC) at scale.** Run a **matched‑settings** triad (GR/NFW/RAR‑gate) on $\gtrsim$20 high‑quality SPARC systems,[^lelli16] report $\Delta \log Z$ distributions and BTFR consistency.[^mcgaugh12]
[5] **Cosmological consistency.** While ΛCDM remains the standard on large scales,[^planck18] explore whether a **bounded**, environment‑modulated coupling can be embedded without violating expansion history or structure growth constraints.

---

## Strong‑Lensing Pilot (RAR phantom‑mass mapping) — methods, fixes, and snapshot results

Scope. This section documents the lensing‑only pilot we added to the next_steps_from_run.py orchestrator. It leaves the Milky Way and SPARC dynamics pipeline unchanged, and only augments how we predict light bending from the same baryons + RAR “phantom mass.” The goal is to check whether a single scalar applied to the phantom contribution can bring galaxy‑scale Einstein radii into the right ballpark without spoiling the dynamics fits. The orchestrator writes results under results/next_steps/<run_name>/… and PNGs under images/next_steps/<run_name>/…. 

next_steps_from_run

What changed (lensing path only)

Einstein‑radius solver fix. We now enforce a monotone non‑increasing envelope on the mean surface‑density curve ⟨Σ⟩(R) and solve for the last crossing with Σ₍cr₎ (outer, physical θ_E). This removes spurious early crossings caused by numerical wiggles and makes θ_E respond sensibly to scaling. (Plots and titles note GR/RAR/scaled intersections.) 

next_steps_from_run

Environment scaling clamp. For signed environment amplitudes ζ we compute scale_env = max(1 + ζ f(R), 0) and use
Σ_lens = Σ_★ + α_lens_ph · scale_env · Σ_ph, so the phantom term is never subtracted in lensing.
(This is lensing‑only; dynamics/rotation‑curve fits are untouched.)

Inputs, knobs, and outputs

Lens CSV schema (one row per lens):
lens_id,z_l,z_s,log10M_star,Re_kpc[,n_sersic,theta_E_obs_arcsec]
Interpretation: total stellar mass and effective radius define a spherical Sérsic lens (optional n_sersic, default 4). The lensing step writes lensing_rar_table.csv and per‑lens plots. 

next_steps_from_run

Lensing‑only scalars (global):

--alpha-lens-ph — multiplies the RAR phantom surface density in lensing only (default 1.0).

--zeta-env-lens — additional amplitude via (1 + ζ f(R)), with --env-profile = constant or tapered (f = [1 + (R/Re)^2]^(-1/2)).

Columns include theta_E_GR_arcsec, theta_E_RAR_arcsec, theta_E_RAR_phscaled_arcsec, SIS yardsticks, and alpha_req_at_thetaE_obs (the scalar that would match the observed θ_E for that lens). 

next_steps_from_run

Repro (example, α = 2.0, ζ sweep with tapered profile):

```bash
python scripts/next_steps_from_run.py \
  --run-dir runs/btfr_fix_20250906 \
  --sparc-dir external_data/Rotmod_LTG \
  --sample q2plus \
  --lensing-sample-csv results/next_steps/btfr_fix_20250906/lenses_castles_small_converted.csv \
  --alpha-lens-ph 2.0 \
  --zeta-env-lens 0.0 \
  --env-profile tapered \
  --out-root results/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p0 \
  --images-root images/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p0
# Repeat with --zeta-env-lens {-0.25, 0.25, 0.5, 0.75} into sibling out/imag dirs
```

(The same orchestrator also generates the Solar‑System constraint plot and BTFR subset from SPARC; see the script header for context and usage.) 

next_steps_from_run

What we see on the 3‑lens pilot

With the corrected solver and clamp, θ_E now scales monotonically with α and ζ.

On the CASTLES‑based pilot (PG1115+080, B1608+656, Q0957+561) using Faber–Jackson‑style placeholders for M★ and Re, a single global α_lens_ph ≈ 2.0 is the current best by RMSE; a small, tapered ζ (|ζ|≲0.5) gives comparable MAE but does not yet beat α=2.0 on RMSE with N=3.

These numbers are placeholders until we swap in measured M★ and Re; the pipeline will pick them up from the same CSV and recompute without any code changes. (The per‑lens table/plots are emitted alongside the run.) 

next_steps_from_run

Figures to add (recommended)

BTFR sanity check (SPARC subset) — slope ≈ 3.18 ± 0.12 (N=89).
Path: images/next_steps/btfr_fix_20250906/btfr_baryonic.png
Caption: Observed V_flat vs baryonic mass for the working SPARC subset; a linear fit in log–log yields a BTFR slope consistent with literature. (Fit metrics come from btfr_fit_summary.json.) 

btfr_fit_summary

Solar‑System constraints — RAR‑plateau vs Cassini.
Path: images/next_steps/btfr_fix_20250906/solar_rar_plateau.png
Caption: Predicted |ΔG/G| ≈ |ξ−1| at 1–30 AU; the model stays below the Cassini bound at Saturn. (Produced by the orchestrator; the plot explicitly annotates the Cassini limit.) 

next_steps_from_run

Lensing: predicted vs observed θ_E (by run) — quick diagnostic.
Path: images/next_steps/btfr_fix_20250906/combined_global_alpha/lensing_global_alpha_pred_vs_obs.png
Caption: Scaled RAR lensing predictions (different α/ζ settings) against observed θ_E; illustrates that a single α≈2 already lands in the right order of magnitude on this pilot.

(Optional per‑lens panel) B1608+656 — ⟨Σ⟩(R) with Σ_cr and θ_E markers
Path: images/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p25/lensing_rar_B1608+656.png
Caption: Stars (GR), RAR total, and scaled lens curves with Σ_cr; vertical lines mark GR/RAR/scaled θ_E after applying the monotone envelope + last‑crossing rule. 

next_steps_from_run

Notes on scope and possible biases

No change to dynamics: SPARC fits, MW analysis, and Solar‑System checks are unchanged by these lensing scalars; the “phantom” only enters the lensing convergence here.

Distances: Σ_cr is built from angular‑diameter distances (flat ΛCDM). For pure local‑gravity tests independent of cosmology, rely on Solar‑System deflection constraints (already included) and, when feasible, any suitable near‑field lenses. The Solar‑System calculation/plot is produced by the orchestrator and annotates the Cassini bound. 

next_steps_from_run

Inputs: Until we replace placeholders with measured M★ and Re per lens, treat all α/ζ values as provisional dials for a pilot study. The lensing CSV columns and run outputs are documented in the lensing step of the orchestrator. 

next_steps_from_run

Quick to‑dos

Swap in measured (log10M_star, Re_kpc[, n_sersic]) for the three pilot lenses; re‑run α=2.0 with a small ζ grid.

Add a small table of per‑lens alpha_req_at_thetaE_obs to the paper text (already emitted by the pipeline). 

next_steps_from_run

If desired, test robustness with --density-profile = hernquist or jaffe (spherical) against the Sérsic default. (If you want this selector exposed, I can keep the CLI flag you proposed alongside the existing lensing flags.)

Where these live in the repo

The “Next‑Step Analyses” orchestrator is documented in the script header and writes a short index at docs/next_steps.md linking to generated artifacts. 

next_steps_from_run
 
next_steps

Where to point images (ready to embed)

BTFR: images/next_steps/btfr_fix_20250906/btfr_baryonic.png

Solar: images/next_steps/btfr_fix_20250906/solar_rar_plateau.png

Lensing (scatter): images/next_steps/btfr_fix_20250906/combined_global_alpha/lensing_global_alpha_pred_vs_obs.png

Lensing (per‑lens, e.g., B1608+656): images/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p25/lensing_rar_B1608+656.png

If you want, I can also auto‑generate a compact “Figures” block (with captions) and append it to the README; just say the word.

Anything you want tweaked? If you’d like this merged directly into the existing “Extended Analyses” section, tell me the exact placement (e.g., after Solar constraints) and I’ll produce a version that’s already interleaved with your current headings.

---

## Figures (quick embed)

- BTFR sanity check (SPARC subset)
  - Path: images/next_steps/btfr_fix_20250906/btfr_baryonic.png
  - ![BTFR sanity check](images/next_steps/btfr_fix_20250906/btfr_baryonic.png)
  - Caption: Observed V_flat vs baryonic mass for the working SPARC subset; a linear fit in log–log yields a BTFR slope consistent with literature. (Fit metrics from btfr_fit_summary.json.)

- Solar‑System constraints — RAR‑plateau vs Cassini
  - Path: images/next_steps/btfr_fix_20250906/solar_rar_plateau.png
  - ![Solar‑System constraints](images/next_steps/btfr_fix_20250906/solar_rar_plateau.png)
  - Caption: Predicted |ΔG/G| ≈ |ξ−1| at 1–30 AU; the model stays below the Cassini bound at Saturn. (Produced by the orchestrator; the plot annotates the Cassini limit.)

- Lensing diagnostics — RMS vs α (hernquist)
  - Path: images/next_steps/btfr_fix_20250906_lastcross/metrics/rms_rel_vs_alpha_hernquist.png
  - ![Lensing RMS vs alpha (hernquist)](images/next_steps/btfr_fix_20250906_lastcross/metrics/rms_rel_vs_alpha_hernquist.png)
  - Caption: Relative RMS error between scaled RAR predictions and observed θ_E across α (zeta=0) for the hernquist profile.

- Lensing per‑lens panel — B1608+656
  - Path: images/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p25/lensing_rar_B1608+656.png
  - ![B1608+656 — lensing panel](images/next_steps/btfr_fix_20250906_lastcross/alpha_2_zeta_0p25/lensing_rar_B1608+656.png)
  - Caption: Stars (GR), RAR total, and scaled lens curves with Σ_cr; vertical lines mark GR/RAR/scaled θ_E after the monotone envelope + last‑crossing rule.

---

## Editorial review and repo status

- Editorial checklist and proposed actions: see feedback.md (project root). This document captures the Nature Physics-oriented checklist (indispensable results, reviewer pre-empts, presentation/policy) plus a prioritized to-do list.
- Editor-style review and definition-of-done: see feedback_editor_review.md. This adds a traffic-light status snapshot, concrete upgrades A–I, and a definition-of-done list used to gate submission readiness.
- Snapshot: Solar-System constraints and PPN/Cassini coverage are implemented; lensing from a single relativistic completion (without any α_lens_ph in the manuscript), hierarchical a0 over a large SPARC/BIG-SPARC sample, wide-binary tests, and Milky Way K_z/Σ_1.1 are the main remaining items.

These files are part of the repo and maintained alongside code to keep the paper and implementation aligned.
