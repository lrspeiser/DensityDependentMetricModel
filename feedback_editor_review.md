Thanks—this is a strong, well‑structured plan. Below is an editor‑style review with (1) a quick status snapshot, (2) concrete, technical edits per section A–I, and (3) a short “definition‑of‑done” checklist you can use to lock the paper.

---

## 1) Status snapshot (traffic‑light)

* **A) Relativistic completion, PPN, GW, lensing (no $\alpha_{\rm lens\,ph}$) — 🔶 Needs work.** Your weak‑field lensing scaffolding is good, but the manuscript must replace any lensing‑only scaling with predictions from a single covariant completion, plus a PPN table and a clear $c_{\rm GW}=c$ statement consistent with GW170817. ([Physical Review Links][1])
* **B) Hierarchical single $a_0$ — 🔶 Needs work.** Per‑galaxy scans exist; a true hierarchical posterior over $a_0$ is still missing.
* **C) Solar System & wide binaries — 🔶 Needs work.** Cassini and Shapiro checks are in place; add a Gaia wide‑binary (WB) module and confront the current, mixed literature head‑on. ([Astrophysics Data System][2], [arXiv][3], [Oxford Academic][4])
* **D) Milky Way $K_z,\ \Sigma_{1.1}$** — 🔴 Failing benchmark flagged. This is an easy reviewer target; fix before submission. Cite Bovy & Rix / McMillan for targets. ([Astrophysics Data System][5], [Oxford Academic][6])
* **E) ΛCDM RAR overlays — 🟨 Minor addition.** Add EAGLE/NIHAO bands to a single, definitive RAR panel. ([arXiv][7], [Oxford Academic][8])
* **F) Cosmology checks — 🟨 Minimal feasibility note.** A one‑page consistency sketch (FRW embedding, linear growth, $c_T=1$) is sufficient for first review. ([Physical Review Links][1])
* **G) Reproducibility — 🟩 Almost there.** Add Source‑Data CSVs + Zenodo DOI + code/data availability statements per Nature Portfolio policy. ([Nature][9], [Springer Nature][10], [Nature Support][11])
* **H) Remove $\alpha_{\rm lens\,ph}$ from paper build — 🟩 Straightforward.** Keep only behind an internal flag.
* **I) New data — 🟩 Sensible.** BIG‑SPARC, SLACS, HSC/DES/KiDS, Gaia DR3 WBs are appropriate expansions. ([arXiv][12], [Astrophysics Data System][13])

---

## 2) Detailed review and concrete upgrades

### A) Relativistic completion, PPN, GW, and lensing

**What’s good:** You already formalized a weak‑field $\phi_{\rm env}=\tfrac12\ln\xi$ idea and built lensing pilots. Cassini tooling is present.

**Gaps & risks:**

* Nature Physics will not accept a **lensing‑only scalar** $\alpha_{\rm lens\,ph}$ in figures; predictions must come from $\Phi,\Psi$ of a single metric theory.
* You need a **PPN table** (γ, β, and preferred‑frame parameters $\alpha_1,\alpha_2$) and a clear statement that **$c_{\rm GW}=c$** in your completion (GW170817). Many scalar–tensor/disformal models are tightly constrained; choose a **$c_T=1$** (Horndeski/DHOST‑compatible) subclass. ([Physical Review Links][1])
* Lensing must use **$\Phi+\Psi$** (anisotropic stress matters); make sure your completion yields this consistently.

**Actionable edits:**

1. **Minimal covariant completion (appendix + code):** pick a $c_T=1$ scalar–tensor (e.g., restricted Horndeski/beyond‑Horndeski) with screening that reproduces your ξ‑gate in the weak field. Add a module (e.g., `theory/relativistic.py`) that maps $\{\text{model params}\}\to\{\Phi,\Psi\}$ and **computes ΔΣ(R) and $\theta_E$** directly. For inspiration/positioning, point to Skordis & Złośnik (relativistic MOND‑like). ([Physical Review Links][14])
2. **PPN exporter:** implement `ppn.evaluate(params, density=...)` → JSON/CSV: $\gamma, \beta, \alpha_1, \alpha_2$ at Solar‑System densities with posteriors from your galaxy fits. Include Cassini’s $|\gamma-1|<2.3\times10^{-5}$ band in the figure. ([Astrophysics Data System][2])
3. **GW speed guardrail:** add a symbolic/finite‑difference check that your EFT coefficients imply **$c_T=1$** over relevant backgrounds; hard‑fail runs that violate it (GW170817 constraint). ([Physical Review Links][1])
4. **Thin‑lens pipeline:** one routine that (i) builds $\Sigma(R)$ from baryons **plus** your theory’s “phantom” term, (ii) computes $\bar{\Sigma}(R)$, (iii) solves for the **last crossing** with $\Sigma_{\rm cr}$ (you already fixed monotonicity), (iv) outputs $\theta_E$ & ΔΣ. Use SLACS/CASTLES exemplars with measured $(M_\*, R_e, z_l, z_s)$. (This avoids any perception of tuning.)
5. **Remove $\alpha_{\rm lens\,ph}$ from the paper build** (keep behind `--internal-pilot` only).

**Citations to anchor claims:** Nature PPN/lensing standards; Cassini; GW170817‑derived $c_T=1$; relativistic MOND‑like precedents. ([Nature][9], [Astrophysics Data System][2], [Physical Review Links][1])

---

### B) Hierarchical single $a_0$ across SPARC/BIG‑SPARC

**What’s good:** You have per‑galaxy runs and a “global $a_0$ scan.”
**Gap:** No **hierarchical posterior** $p(a_0\,|\,\text{all})$ with intrinsic scatter decomposition.

**Actionable edits:**

* Implement a **two‑stage hierarchical** path that scales well:

  1. Per galaxy, tabulate $\ell_i(a_0)\equiv \log p(\text{data}_i\mid a_0,\theta_i)$ on a grid while marginalizing nuisance ($\Upsilon_\star$, distance, inclination, gas systematics).
  2. Sample hyperparameters $\mu,\sigma$ in $a_0\sim\mathrm{LogNormal}(\mu,\sigma)$ using

     $$
     \log \mathcal{L}(\mu,\sigma)=\sum_i \log\!\int \exp[\ell_i(a_0)]\,\mathcal{N}(\ln a_0;\mu,\sigma)\,da_0.
     $$
  3. Report $\mu,\sigma$, posterior predictive checks, and compare **Δlog Z** against **GR+baryons** and **GR+NFW** baselines on the same selection (≥100 galaxies, Q≤2).
* Recreate a **RAR master panel** with SPARC/BIG‑SPARC points, your posterior band, and ΛCDM bands (EAGLE/NIHAO). ([Astrophysics Data System][13], [arXiv][12])

---

### C) Solar System and wide binaries

**What’s good:** Cassini/PPN hooks and plots are present.
**Gap:** No **Gaia DR3 wide‑binary** test; referees will ask because WBs probe the same acceleration regime as RAR and the literature is active and mixed. ([Oxford Academic][4], [arXiv][3])

**Actionable edits:**

* Add `wb/` module to ingest a vetted WB catalog (start from **El‑Badry+ 2021**; then apply stricter DR3 cuts). Model contamination (triples, fly‑bys), perspective effects, and the **external field** consistently with your gate. Compute the WB velocity‑ratio statistic vs projected separation and compare with recent **pro‑Newtonian** and **pro‑MOND** analyses, showing where your screening places you. ([Astrophysics Data System][15], [Oxford Academic][4], [arXiv][16])

---

### D) Milky Way $K_z$ and $\Sigma_{1.1}$

**What’s good:** Pipeline exists; validation currently fails.
**Actionable edits:**

* Implement $K_z(R_0,z)$ and $\Sigma_{1.1}$ from the **same** baryon model + ξ gate; make sure the vertical density of the gas and thick disk are accurate (scale heights/flare). Target $\Sigma_{1.1}\approx 70\,M_\odot\,\mathrm{pc}^{-2}$ at $R_0$ as a sanity check, and show that parameters passing Cassini also pass $K_z$. Benchmark to Bovy & Rix (2013) and McMillan (2017/2022). ([Astrophysics Data System][5], [Oxford Academic][6])

---

### E) Positioning vs ΛCDM hydrodynamical RAR

**Actionable edits:**

* One overlay figure: SPARC points + your DGG curve ± intrinsic scatter + **EAGLE/NIHAO** bands; annotate where curves diverge (low $g_{\rm bar}$, mass‑dependence of scatter). This frames your novelty without over‑claiming. ([arXiv][7], [Oxford Academic][8])

---

### F) Cosmology checks

**Actionable edits (succinct):**

* One‑page appendix: show the completion can be embedded in FRW with **$c_T=1$** and no obvious ghost/gradient instabilities; sketch linear‑growth behaviour (e.g., EFT α‑parameters) constrained to be small enough not to violate **fσ8** and CMB‑era constraints. Explicitly state that full Boltzmann treatment is deferred. ([Physical Review Links][17])

---

### G) Data/code availability & reproducibility

**Actionable edits:**

* Add `make_source_data.py` that emits a CSV/Parquet for **every figure panel** (RAR, BTFR, MW, ΔΣ, $\theta\_E$, PPN table, WB stats).
* Create a **Zenodo**‑backed release (tag + DOI), include **CITATION.cff**, and add **Data/Code Availability** statements per Nature policy. ([Nature][9], [Springer Nature][10])

---

### H) Lensing‑only $\alpha_{\rm lens\,ph}$ removal

**Actionable edits:**

* Hide behind `--internal-pilot`; add an assert that it is **never** used in manuscript build. Replace all paper figures with metric‑predicted ΔΣ and $\theta_E$.

---

### I) Datasets to fetch

* **BIG‑SPARC** (public preprint, 4k galaxies) for sample scale. ([arXiv][12])
* **HSC / DES Y3 / KiDS‑1000** for **stacked galaxy–galaxy lensing** comparisons; reuse their public shear catalogs or published ΔΣ products where possible. ([Astrophysics Data System][18], [arXiv][19], [A&A][20])
* **SLACS/CASTLES** (strong lenses) with measured $(M_\*, R_e, z_l, z_s)$.
* **Gaia DR3 wide‑binary** catalogs (start from El‑Badry+; apply DR3 updates/cuts). ([Astrophysics Data System][15])

---

## 3) Definition‑of‑done (what the paper must include to clear editorial triage)

1. **Single‑theory predictions:** All rotation, $K_z$, lensing (ΔΣ, $\theta_E$), and Solar‑System predictions come from **one** parameter set—no lensing fudge factors. (GW170817‑safe, $c_T=1$). ([Physical Review Links][1])
2. **PPN table** with uncertainties from your posteriors and a Cassini‑band overlay at 1–30 AU; brief note on **$\alpha_1,\alpha_2$** (preferred‑frame), citing pulsar/solar limits. ([Astrophysics Data System][2], [arXiv][21])
3. **Hierarchical $a_0$**: posterior $p(a_0\,|\,\text{all})$ with intrinsic scatter; Δlog Z distributions vs GR+baryons and GR+NFW.
4. **Milky Way vertical dynamics**: $K_z(R_0,z)$ / $\Sigma_{1.1}$ figure matching classical constraints. ([Astrophysics Data System][5])
5. **Wide‑binary test**: your prediction vs DR3 WB statistics with contamination modeling and the Galactic external field—positioned relative to both Newtonian‑favouring and MOND‑favouring studies. ([arXiv][3], [Oxford Academic][4])
6. **RAR overlay vs ΛCDM** (EAGLE/NIHAO) with a written discriminant. ([arXiv][7])
7. **Reproducibility**: Zenodo DOI + Source‑Data for every panel + one‑command rebuild script; explicit Code/Data Availability statements. ([Nature][9], [Springer Nature][22])

---

## Minor, but useful, implementation notes

* **Identifiability:** Treat $\Upsilon_\star$ as hierarchical (IMF‑informed priors) to reduce degeneracy with $a_0$.
* **External‑field effect (EFE):** Make its treatment explicit in both WBs and outer‑disk fits; reviewers will ask.
* **Fair model comparison:** Penalize per‑galaxy halo freedom via evidence; show per‑galaxy and aggregate Δlog Z histograms.
* **Figure ethics:** Put the precise **selection cuts** in figure captions; publish the exact object lists used.

---

### References for key constraints & datasets (for your Methods)

* **Nature policy & availability:** data/code/source‑data requirements. ([Nature][9], [Springer Nature][10])
* **Cassini PPN $\gamma$:** $|\gamma-1| \lesssim 2.3\times10^{-5}$. ([Astrophysics Data System][2])
* **GW170817 $c_T=1$:** implications for scalar–tensor/disformal couplings. ([Physical Review Links][1])
* **Relativistic MOND‑like example:** Skordis & Złośnik. ([Physical Review Links][14])
* **Wide‑binary literature (mixed results):** Pittordis & Sutherland; Banik et al.; Chae; Hernandez (review). ([Oxford Academic][4], [arXiv][3])
* **MW vertical force benchmarks:** Bovy & Rix; McMillan. ([Astrophysics Data System][5], [Oxford Academic][6])
* **SPARC / BIG‑SPARC:** canonical rotation‑curve datasets. ([Astrophysics Data System][13], [arXiv][12])
* **ΛCDM RAR bands:** EAGLE/NIHAO style results. ([arXiv][7])
* **Lensing stacks:** HSC, DES Y3, KiDS‑1000 shear catalogs. ([Astrophysics Data System][18], [arXiv][19], [A&A][20])

---

### Bottom line

Your plan targets exactly the gaps that would block editorial triage: relativistic closure (with PPN/GW/lensing), universality of $a_0$, local‑universe consistency ($K_z$, WBs), and rigorous reproducibility. If you implement the upgrades above—especially removing $\alpha_{\rm lens\,ph}$ from the paper build, adding a hierarchical $a_0$, and fixing $K_z$—the package will be mature enough to send to review at **Nature Physics**.

[1]: https://link.aps.org/doi/10.1103/PhysRevLett.119.251302?utm_source=chatgpt.com "Dark Energy after GW170817 and GRB170817A"
[2]: https://ui.adsabs.harvard.edu/abs/2003Natur.425..374B/abstract?utm_source=chatgpt.com "A test of general relativity using radio links with the Cassini ..."
[3]: https://arxiv.org/abs/2311.03436?utm_source=chatgpt.com "Strong constraints on the gravitational law from $Gaia$ DR3 wide binaries"
[4]: https://academic.oup.com/mnras/article-abstract/488/4/4740/5531773?utm_source=chatgpt.com "Testing modified gravity with wide binaries in Gaia DR2"
[5]: https://ui.adsabs.harvard.edu/abs/2013ApJ...779..115B/abstract?utm_source=chatgpt.com "A Direct Dynamical Measurement of the Milky Way's Disk ..."
[6]: https://academic.oup.com/mnras/article/465/1/76/2417479?utm_source=chatgpt.com "mass distribution and gravitational potential of the Milky Way"
[7]: https://arxiv.org/abs/1610.07663?utm_source=chatgpt.com "a Natural Outcome of Galaxy Formation in Cold Dark ..."
[8]: https://academic.oup.com/mnras/article/471/2/1841/3939742?utm_source=chatgpt.com "origin of the mass discrepancy–acceleration relation in ΛCDM"
[9]: https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards?utm_source=chatgpt.com "Reporting standards and availability of data, materials ..."
[10]: https://www.springernature.com/gp/authors/research-data-policy/data-availability-statements?utm_source=chatgpt.com "Data Availability Statements | Publish your research"
[11]: https://support.nature.com/en/support/solutions/articles/6000237611-write-a-data-availability-statement-for-a-paper?utm_source=chatgpt.com "Write a data availability statement for a paper"
[12]: https://arxiv.org/abs/2411.13329?utm_source=chatgpt.com "BIG-SPARC: The new SPARC database"
[13]: https://ui.adsabs.harvard.edu/abs/2016AJ....152..157L/abstract?utm_source=chatgpt.com "SPARC: Mass Models for 175 Disk Galaxies with Spitzer ..."
[14]: https://link.aps.org/doi/10.1103/PhysRevLett.127.161302?utm_source=chatgpt.com "New Relativistic Theory for Modified Newtonian Dynamics"
[15]: https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.2269E/abstract?utm_source=chatgpt.com "A million binaries from Gaia eDR3: sample selection and ..."
[16]: https://arxiv.org/abs/2309.10404?utm_source=chatgpt.com "Robust Evidence for the Breakdown of Standard Gravity at ..."
[17]: https://link.aps.org/doi/10.1103/PhysRevLett.119.251304?utm_source=chatgpt.com "Dark Energy After GW170817: Dead Ends and the Road Ahead"
[18]: https://ui.adsabs.harvard.edu/abs/2018PASJ...70S..25M?utm_source=chatgpt.com "The first-year shear catalog of the Subaru Hyper Suprime ..."
[19]: https://arxiv.org/abs/2105.13541?utm_source=chatgpt.com "[2105.13541] Dark Energy Survey Year 3 Results"
[20]: https://www.aanda.org/articles/aa/full_html/2021/02/aa39063-20/aa39063-20.html?utm_source=chatgpt.com "KiDS-1000 Cosmology: Multi-probe weak gravitational lensing ..."
[21]: https://arxiv.org/abs/1209.5171?utm_source=chatgpt.com "New Constraints on Preferred Frame Effects from Binary Pulsars"
[22]: https://www.springernature.com/gp/open-science/code-policy?utm_source=chatgpt.com "Code Policy | Open science"

