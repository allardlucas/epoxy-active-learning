"""Figures used in the slides (reads results/*.csv, writes results/*.png)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
COL = {"random": "#9AA0A6", "greedy_xgb": "#E8710A", "greedy_rf": "#F4B400",
       "gp_ei": "#1A73E8", "gp_ucb": "#0B8043"}
LAB = {"random": "Aléatoire", "greedy_xgb": "Greedy XGBoost (papier)", "greedy_rf": "Greedy RF",
       "gp_ei": "BO GP + EI", "gp_ucb": "BO GP + UCB"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

def campaign_history():
    df = pd.read_csv(ROOT / "data" / "epoxy_adhesive_dataset.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    c = {"initial_DoE": "#9AA0A6", "active_learning": "#E8710A", "bayesian_opt": "#1A73E8"}
    n = {"initial_DoE": "Plan initial (32)", "active_learning": "Active learning greedy (15)", "bayesian_opt": "Optim. bayésienne (4)"}
    for s, g in df.groupby("stage", sort=False):
        ax.errorbar(g.id, g.sigma_ad_MPa, g.sigma_std_MPa, fmt="o", color=c[s], label=n[s], ms=6, capsize=2)
    best = df[df.stage == "initial_DoE"].sigma_ad_MPa.max()
    ax.axhline(best, ls="--", color="k", lw=1); ax.text(1, best + 0.8, f"meilleur point initial : {best} MPa", fontsize=10)
    ax.set_xlabel("n° d'expérience"); ax.set_ylabel("σ_ad mesurée (MPa)"); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(RES / "campaign_history.png", dpi=200)

def r2_split():
    import sys; sys.path.insert(0, str(ROOT / "src"))
    from reproduce import load, loocv, models, FEATURES, TARGET
    from sklearn.metrics import r2_score
    df = load(); rows = []
    for n in [32, 37, 42, 47]:
        sub = df[df.id <= n]; y = sub[TARGET].values
        _, p = loocv(models()["XGBoost"], sub[FEATURES].values, y); m = (sub.id <= 32).values
        rows.append((n, r2_score(y, p), r2_score(y[m], p[m])))
    r = pd.DataFrame(rows, columns=["n", "all", "doe"]); r.to_csv(RES / "r2_split.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r.n, r["all"], "o-", color="#E8710A", label="R² LOOCV sur tout le jeu (ce que rapporte le papier)")
    ax.plot(r.n, r.doe, "o-", color="#1A73E8", label="R² sur les 32 points du plan initial")
    ax.set_xticks(r.n); ax.set_xlabel("taille du jeu de données"); ax.set_ylabel("R²"); ax.set_ylim(0.4, 0.9)
    ax.legend(frameon=False, fontsize=10, loc="lower right"); fig.tight_layout(); fig.savefig(RES / "r2_split.png", dpi=200)

def benchmark():
    """Final best true strength after 15 experiments, per oracle, strategy and initial design."""
    b = pd.read_csv(RES / "benchmark.csv"); last = b[b.n_new == 15]
    oracles = ["gp51", "rf51", "xgb51", "gp32"]
    olab = {"gp51": "GP\n51 mesures", "rf51": "Random Forest\n51 mesures", "xgb51": "XGBoost\n51 mesures",
            "gp32": "GP\n32 essais initiaux"}
    strategies = ["random", "greedy_xgb", "greedy_rf", "gp_ucb", "gp_ei"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, n_init in zip(axes, [32, 10]):
        g = last[last.n_init == n_init]
        for i, s in enumerate(strategies):
            m = g[g.strategy == s].groupby("oracle").best_true.agg(["mean", "sem"]).reindex(oracles)
            x = np.arange(len(oracles)) + (i - 2) * 0.14
            ax.errorbar(x, m["mean"], 1.96 * m["sem"], fmt="o", color=COL[s], label=LAB[s], ms=7, capsize=3,
                        lw=2 if s in ("greedy_xgb", "gp_ei") else 1.2)
        opt = g.groupby("oracle").optimum.first().reindex(oracles)
        ax.scatter(np.arange(len(oracles)), opt, marker="_", s=900, color="k", label="optimum du labo simulé")
        ax.set_xticks(np.arange(len(oracles))); ax.set_xticklabels([olab[o] for o in oracles], fontsize=10)
        ax.set_title(f"Départ : {n_init} essais" + (" (plan du papier)" if n_init == 32 else " (sous-plan bien réparti)"))
        ax.set_xlabel("labo simulé construit avec…")
    axes[0].set_ylabel("meilleure σ_ad vraie après 15 essais (MPa)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=10, loc="lower center", ncol=6)
    fig.suptitle("20 campagnes par case, moyenne et intervalle de confiance à 95 %", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(RES / "benchmark.png", dpi=200)

if __name__ == "__main__":
    campaign_history(); r2_split(); benchmark(); print("figures written to", RES)
