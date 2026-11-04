"""Replay the paper's Bayesian-optimisation step (Section 3.2.3, Table 3).

After 3 AL cycles (47 points) the paper fixes its XGBoost model and lets GPyOpt
refine r and T_cure for MW_epoxy = 370, MW_curing_agent = 400. All 4 proposals
had the same predicted strength (26.9 MPa), and the 4th one measured 35.8 MPa.
Questions:
  1. Does the fixed XGBoost distinguish recipes inside that (r, T) region at all?
  2. How likely was 35.8 MPa according to a GP fitted on the same 47 points?"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
from xgboost import XGBRegressor
from benchmark import gp, FEATURES, load

ROOT = Path(__file__).resolve().parents[1]
BEST_BEFORE_BO = 31.9  # sample 17

def main():
    df = load(); al = df[df.id <= 47]
    # Same scaling as the paper: standardisation fitted on the training data
    mu_x, sd_x = al[FEATURES].values.mean(0), al[FEATURES].values.std(0)
    z = lambda X: (np.asarray(X, float) - mu_x) / sd_x
    xgb = XGBRegressor(max_depth=4, gamma=2.86, learning_rate=0.4, n_estimators=100, random_state=0)
    xgb.fit(z(al[FEATURES].values), al.sigma_ad_MPa.values)

    r = np.round(np.arange(0.75, 1.5001, 0.01), 2); T = np.arange(90, 211, 1)
    R, TT = np.meshgrid(r, T)
    surf = np.c_[np.full(R.size, 370), np.full(R.size, 400), R.ravel(), TT.ravel()]
    pred = xgb.predict(z(surf)).reshape(R.shape)

    bo = df[df.stage == "bayesian_opt"]
    region = (R >= 0.85) & (R <= 1.35) & (TT >= 185)
    print("1) Fixed XGBoost, MWE=370, MWC=400")
    print(f"   distinct predicted values on the whole (r, T) plane: {len(np.unique(pred.round(3)))}")
    print(f"   in the region of the 4 BO proposals (r 0.85-1.35, T >= 185): "
          f"{pred[region].min():.1f} to {pred[region].max():.1f} MPa")
    print("   predictions at the 4 BO proposals:",
          np.round(xgb.predict(z(bo[FEATURES].values)), 1), "| paper: 26.9 for all")

    # A noisy GP on the same 47 points, molecular weights in log
    Xg = al[FEATURES].values.astype(float); Xg[:, :2] = np.log(Xg[:, :2])
    lo, hi = Xg.min(0), Xg.max(0); g = lambda X: (X - lo) / (hi - lo)
    m = gp().fit(g(Xg), al.sigma_ad_MPa.values)
    Xb = bo[FEATURES].values.astype(float); Xb[:, :2] = np.log(Xb[:, :2])
    mu, sd = m.predict(g(Xb), return_std=True)
    rows = []
    print("\n2) GP fitted on the 47 points, at the 4 BO proposals")
    for (_, b), u, s in zip(bo.iterrows(), mu, sd):
        p_beat = 1 - norm.cdf((BEST_BEFORE_BO - u) / s)
        zscore = (b.sigma_ad_MPa - u) / s
        rows.append(dict(r=b.r, T_cure=b.T_cure, measured=b.sigma_ad_MPa, gp_mean=u, gp_sd=s,
                         p_beat_31_9=p_beat, z_measured=zscore))
        print(f"   r={b.r:.2f} T={b.T_cure:.0f}: measured {b.sigma_ad_MPa:5.1f} | GP {u:4.1f} ± {s:3.1f} "
              f"| P(> 31.9) = {p_beat:.0%} | measured is {zscore:+.1f} sd from GP mean")
    pd.DataFrame(rows).to_csv(ROOT / "results" / "replay_bo.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.pcolormesh(R, TT, pred, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="σ_ad prédite par XGBoost (MPa)")
    ax.scatter(bo.r, bo.T_cure, c="white", edgecolor="k", s=80, zorder=3)
    for _, b in bo.iterrows():
        ax.annotate(f"{b.sigma_ad_MPa:.1f}", (b.r, b.T_cure), xytext=(6, -14), textcoords="offset points",
                    color="white", fontsize=11, fontweight="bold")
    ax.set_xlabel("ratio amine/époxyde r"); ax.set_ylabel("T de cuisson (°C)")
    ax.set_title("XGBoost figé (47 points), résine 370 / durcisseur 400\npoints : propositions de la BO et mesures", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "replay_bo.png", dpi=200)

if __name__ == "__main__":
    main()
