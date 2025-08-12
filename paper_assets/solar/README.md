# Solar-System posterior assets

This folder aggregates the per-galaxy Solar-System posterior bands derived from the high-budget env-ER evidence runs.

Commit: 3c91b28b75906531f94bc7de0ed89ac164315ed8

Commands used (examples):

- Per-galaxy bands and table generation
  python tools/gen_solar_system_posterior.py \
    --exp-npz images/sparc_env_fit_NGC3198_exp_hi.npz \
    --pow-npz images/sparc_env_fit_NGC3198_power_hi.npz \
    --out-png images/solar_system_posterior_ngc3198.png \
    --out-md docs/solar/solar_posterior_ngc3198.md
  (Repeated for NGC2403, NGC2841, NGC6946, NGC5055)

NPZ source files:
- images/sparc_env_fit_NGC3198_exp_hi.npz, images/sparc_env_fit_NGC3198_power_hi.npz
- images/sparc_env_fit_NGC2403_exp_hi.npz, images/sparc_env_fit_NGC2403_power_hi.npz
- images/sparc_env_fit_NGC2841_exp_hi.npz, images/sparc_env_fit_NGC2841_power_hi.npz
- images/sparc_env_fit_NGC6946_exp_hi.npz, images/sparc_env_fit_NGC6946_power_hi.npz
- images/sparc_env_fit_NGC5055_exp_hi.npz, images/sparc_env_fit_NGC5055_power_hi.npz

Artifacts copied here:
- solar_system_posterior_ngc3198.png, solar_posterior_ngc3198.md
- solar_system_posterior_ngc2403.png, solar_posterior_ngc2403.md
- solar_system_posterior_ngc2841.png, solar_posterior_ngc2841.md
- solar_system_posterior_ngc6946.png, solar_posterior_ngc6946.md
- solar_system_posterior_ngc5055.png, solar_posterior_ngc5055.md

