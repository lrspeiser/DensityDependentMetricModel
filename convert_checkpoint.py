import pickle
import numpy as np

# 🛠 REQUIRED: make sure these functions are in scope for unpickling
from run_dynesty import (
    log_likelihood_dynesty_debug,
    prior_transform_dynesty
)

# 📂 Path to your checkpoint
pkl_path = "chains_dynesty/cassini_safe_power_full/dynesty_checkpoint.pkl"
npz_path = "chains_dynesty/cassini_safe_power_full/dynesty_checkpoint_converted.npz"

# 🔓 Unpickle the object
with open(pkl_path, "rb") as f:
    obj = pickle.load(f)

# 🔍 Determine structure
if isinstance(obj, dict):
    print("✅ Loaded: plain dictionary")
    print("Keys:", list(obj.keys()))
    if "samples" not in obj or "logz" not in obj:
        raise KeyError("Missing required keys like 'samples' or 'logz'")
    data = obj
elif hasattr(obj, "samples"):
    print("✅ Loaded: Dynesty Results object")
    data = {
        "samples": obj.samples,
        "logz": obj.logz,
        "logzerr": obj.logzerr,
        "logl": getattr(obj, "logl", None),
        "blob": getattr(obj, "blob", None),
        "param_names": getattr(obj, "param_names", [f"param_{i}" for i in range(obj.samples.shape[1])])
    }
else:
    raise TypeError(f"❌ Unsupported object type: {type(obj)}")

# 💾 Save to compressed npz format
np.savez(npz_path, **data)
print(f"✅ Saved .npz to {npz_path}")
