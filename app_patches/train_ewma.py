import json, math
import pandas as pd
from pathlib import Path

def _series_por_item(df_ventas_train: pd.DataFrame):
    df = df_ventas_train.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Fecha","Item","Ventas_train"])
    return {it: g.set_index("Fecha").sort_index()["Ventas_train"]
            for it, g in df.groupby("Item", dropna=False)}

def _backtest_ewma(s: pd.Series, alpha: float) -> float:
    lvl = None; errs = []
    for x in s:
        pred = lvl if lvl is not None else s.iloc[0]
        errs.append(abs(float(pred) - float(x)))
        lvl = alpha*float(x) + (1-alpha)*(lvl if lvl is not None else float(x))
    return float(sum(errs)) / max(len(errs), 1)

def train_ewma(df_ventas_train: pd.DataFrame, alphas=(0.2,0.3,0.5)):
    per_item = _series_por_item(df_ventas_train)
    model = {}
    for it, s in per_item.items():
        best = None
        for a in alphas:
            mae = _backtest_ewma(s, a)
            if best is None or mae < best["mae"]:
                best = {"alpha": a, "mae": mae}
        # nivel final (última EWMA)
        lvl = None
        for x in s:
            lvl = best["alpha"]*float(x) + (1-best["alpha"])*(lvl if lvl is not None else float(x))
        model[it] = {"alpha": best["alpha"], "level": float(lvl)}
    return {"algo":"EWMA","model":model}

def save_model(model: dict, path="models/model_ewma.json"):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    return str(p)

def predict_next(model: dict) -> pd.DataFrame:
    rows = []
    for it, cfg in model.get("model", {}).items():
        rows.append({"Item": it, "Esperado_dia": float(cfg["level"])})
    return pd.DataFrame(rows)