# etl.py (USED BY RENDER)

import pickle
import numpy as np
import os

def run_etl_pipeline():
    required = ["data.pkl", "meta_sim.pkl", "desc_embeddings.pkl"]
    missing = [f for f in required if not os.path.exists(f)]

    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {', '.join(missing)}. "
            "Run etl_offline.py locally first."
        )

    print("🔹 Loading data.pkl...")
    df = pickle.load(open("data.pkl", "rb"))

    print("🔹 Loading meta_sim.pkl...")
    meta_sim = pickle.load(open("meta_sim.pkl", "rb"))

    print("🔹 Loading desc_embeddings.pkl...")
    desc_embeddings = pickle.load(open("desc_embeddings.pkl", "rb"))

    # Safety: enforce float32
    meta_sim = meta_sim.astype(np.float32, copy=False)
    desc_embeddings = desc_embeddings.astype(np.float32, copy=False)

    print("✅ Artifacts loaded successfully")
    return {
        "data": df,
        "meta_sim": meta_sim,
        "desc_embeddings": desc_embeddings
    }
