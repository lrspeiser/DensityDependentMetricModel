import json, glob, os, csv, re

def load_many(patterns):
    out = {}
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    d = json.load(f)
                gid = d.get("galaxy_id")
                if not gid:
                    m = re.search(r"_([A-Za-z0-9]+)\.json$", os.path.basename(p))
                    gid = m.group(1) if m else os.path.splitext(os.path.basename(p))[0]
                out[gid] = d
            except Exception:
                pass
    return out

def main():
    er = load_many([
        "images/sparc_er_evidence_*.json",
        "images/*er*evidence_*.json",
        "images/sparc_env_fit_*.json"  # ER evidence sidecars from env tool
    ])
    gr = load_many(["images/sparc_gr_evidence_*.json", "images/*gr*evidence_*.json"])
    nfw = load_many(["images/sparc_nfw_evidence_*.json", "images/*nfw*evidence_*.json"])
    galaxies = sorted(set(er) | set(gr) | set(nfw))
    rows = []
    for g in galaxies:
        er_d, gr_d, nfw_d = er.get(g), gr.get(g), nfw.get(g)
        def v(d, k):
            return (None if d is None else d.get(k))
        logZ_ER, eER = v(er_d,"logZ"), v(er_d,"logZ_err")
        logZ_GR, eGR = v(gr_d,"logZ"), v(gr_d,"logZ_err")
        logZ_NFW, eNFW = v(nfw_d,"logZ"), v(nfw_d,"logZ_err")
        d_ER_GR = (None if (logZ_ER is None or logZ_GR is None) else (logZ_ER - logZ_GR))
        d_NFW_GR = (None if (logZ_NFW is None or logZ_GR is None) else (logZ_NFW - logZ_GR))
        d_ER_NFW = (None if (logZ_ER is None or logZ_NFW is None) else (logZ_ER - logZ_NFW))
        rows.append({
            "galaxy": g,
            "logZ_GR": logZ_GR, "logZ_err_GR": eGR,
            "logZ_NFW": logZ_NFW, "logZ_err_NFW": eNFW,
            "logZ_ER": logZ_ER, "logZ_err_ER": eER,
            "Delta_ER_minus_GR": d_ER_GR,
            "Delta_NFW_minus_GR": d_NFW_GR,
            "Delta_ER_minus_NFW": d_ER_NFW,
        })
    out = os.path.join("images","sparc_evidence_triplet_summary.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(out)

if __name__ == "__main__":
    main()

