# Data and external services (to be consulted before enabling any downloads)

- This project aims to run entirely on local/public datasets. If any step requires a web service or API key, document it here before enabling the code.
- Code that uses such services must include a comment pointing here, per repository rules.

Suggested layout (local-only by default)
- SPARC: place under external_data/Rotmod_LTG or a configured local path.
- SLACS-like strong lens sample: point to a local CSV/JSON you curate.
- Weak lensing stacks (DES/HSC/KiDS): use public releases saved locally.
- Cluster maps: use public convergence/X-ray maps stored locally.
- BAO/CMB compressed likelihood inputs: store as plain text/CSV locally.

No secrets in code
- Never inline secrets in code or logs. If a secret is ever needed, load it into an environment variable from your secret manager as a separate step.

