"""Reproduce the model comparison (Fig. 5) and the active-learning cycles (Fig. 7)
of Pruksawan et al. 2019 with leave-one-out cross-validation."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["MW_epoxy", "MW_curing_agent", "r", "T_cure"]
TARGET = "sigma_ad_MPa"

def load():
    return pd.read_csv(ROOT / "data" / "epoxy_adhesive_dataset.csv")

def models(xgb_params=None):
    xgb_params = xgb_params or dict(max_depth=5, gamma=3.3)
    return {
        "ElasticNet": ElasticNet(alpha=0.1),
        "RandomForest": RandomForestRegressor(max_depth=3, n_estimators=100, random_state=2),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0, **xgb_params),
    }

def loocv(model, X, y):
    pred = cross_val_predict(make_pipeline(StandardScaler(), model), X, y, cv=LeaveOneOut())
    return dict(R2=r2_score(y, pred), RMSE=mean_squared_error(y, pred) ** 0.5,
                MAE=mean_absolute_error(y, pred)), pred

def main():
    df = load()
    init = df[df.stage == "initial_DoE"]
    print("== Model comparison, initial dataset (n=32), LOOCV ==  paper: XGB R2~0.68")
    rows = []
    for name, m in models().items():
        s, _ = loocv(m, init[FEATURES].values, init[TARGET].values)
        rows.append({"model": name, **s}); print(f"{name:13s}", {k: round(v, 2) for k, v in s.items()})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "model_comparison.csv", index=False)

    print("\n== XGBoost along active-learning cycles (paper: R2 0.85 / RMSE 4.0 / MAE 3.0 at n=47) ==")
    hp = [dict(max_depth=5, gamma=3.3), dict(max_depth=4, gamma=2.7),
          dict(max_depth=4, gamma=1.8), dict(max_depth=4, gamma=2.86, learning_rate=0.4)]
    rows = []
    for cycle, n in enumerate([32, 37, 42, 47]):
        sub = df[df.id <= n]
        for tuned in (True, False):
            p = hp[cycle] if tuned else hp[0]
            p = {"learning_rate": 0.1, **p}
            m = XGBRegressor(n_estimators=100, random_state=0, **p)
            s, _ = loocv(m, sub[FEATURES].values, sub[TARGET].values)
            rows.append({"cycle": cycle, "n": n, "per_cycle_tuning": tuned, **s})
            print(f"cycle {cycle} n={n} tuned={tuned!s:5}", {k: round(v, 2) for k, v in s.items()})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "al_cycles.csv", index=False)

    best_init = init[TARGET].max()
    al = df[df.stage == "active_learning"][TARGET]
    print(f"\nBest initial sample: {best_init} MPa | best AL proposal: {al.max()} MPa "
          f"| AL proposals beating initial best: {(al > best_init).sum()}/15")

if __name__ == "__main__":
    main()
