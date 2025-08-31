# app_patches/imputation.py
import pandas as pd
from datetime import time

def detect_zero_spans(df_day: pd.DataFrame, time_col: str, stock_col: str):
    """Devuelve lista de (idx_ini, idx_fin_exclusivo) donde stock<=0 es continuo."""
    s = (df_day[stock_col].fillna(0) <= 0).astype(int)
    spans, start = [], None
    for i, v in enumerate(s):
        if v and start is None: start = i
        if (not v) and start is not None:
            spans.append((start, i)); start = None
    if start is not None: spans.append((start, len(s)))
    return spans

def estimate_lost_demand(df_day: pd.DataFrame, time_col: str, sales_col: str,
                         expected_total: float | None, zero_spans):
    """
    Estima demanda perdida en spans de stock=0 usando tasa base.
    Tasa base = (expected_total / duración_abierta) si expected_total; si no, usa (ventas_observadas / duración_no_cortada).
    """
    if df_day.empty: return 0.0
    # Duración total (n bins) y duración “no cortada”
    n = len(df_day)
    cut = sum((b-a) for a,b in zero_spans)
    open_bins = max(n, 1)
    open_bins_nocut = max(open_bins - cut, 1)

    sales_obs = float(df_day[sales_col].fillna(0).sum())
    if expected_total is not None and expected_total > 0:
        base_rate = expected_total / open_bins    # ventas/bloque esperadas
    else:
        base_rate = sales_obs / open_bins_nocut   # tasa observada donde había stock

    lost = base_rate * cut
    return float(lost)

def impute_day(df_day: pd.DataFrame, time_col: str, stock_col: str, sales_col: str,
               expected_total: float | None = None):
    """Retorna ventas_imputadas, perdidas, spans."""
    spans = detect_zero_spans(df_day, time_col, stock_col)
    lost = estimate_lost_demand(df_day, time_col, sales_col, expected_total, spans)
    ventas_obs = float(df_day[sales_col].fillna(0).sum())
    ventas_imp = ventas_obs + lost
    return ventas_imp, lost, spans
