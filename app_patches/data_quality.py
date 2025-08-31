# app_patches/data_quality.py
import pandas as pd

def audit_stock_neg(df: pd.DataFrame, cols=("Stock_ini","Stock_fin","Stock")) -> pd.DataFrame:
    issues=[]
    for c in cols:
        if c in df.columns:
            bad=df[df[c].fillna(0)<0].copy()
            if not bad.empty:
                bad["Col"]=c; bad["Valor"]=bad[c]
                issues.append(bad[["Fecha","Item","Col","Valor"] if "Item" in bad.columns else ["Fecha","Col","Valor"]])
    return pd.concat(issues, ignore_index=True) if issues else pd.DataFrame(columns=["Fecha","Item","Col","Valor"])

def detect_outliers_daily(diario: pd.DataFrame, fecha="Fecha", item="Item", hora="Hora", ventas="Ventas", k=1.5):
    if diario.empty or ventas not in diario.columns: 
        return pd.DataFrame(columns=[fecha,item,"Ventas_dia","Lo","Hi","is_outlier"])
    d=(diario.rename(columns={fecha:"Fecha",item:"Item",ventas:"Ventas"})
            .groupby(["Fecha","Item"],as_index=False)["Ventas"].sum()
            .rename(columns={"Ventas":"Ventas_dia"}))
    rows=[]
    for it,g in d.groupby("Item"):
        q1=g["Ventas_dia"].quantile(0.25); q3=g["Ventas_dia"].quantile(0.75); iqr=q3-q1
        lo=q1-k*iqr; hi=q3+k*iqr
        gg=g.copy(); gg["Lo"]=lo; gg["Hi"]=hi; gg["is_outlier"]=~gg["Ventas_dia"].between(lo,hi)
        rows.append(gg)
    return pd.concat(rows, ignore_index=True) if rows else d.assign(Lo=pd.NA,Hi=pd.NA,is_outlier=False)

def clip_negatives(df: pd.DataFrame, cols=("Stock_ini","Stock_fin","Stock")) -> pd.DataFrame:
    out=df.copy()
    for c in cols:
        if c in out.columns: out[c]=out[c].clip(lower=0)
    return out

def cap_sales(df_daily: pd.DataFrame, limits: pd.DataFrame) -> pd.DataFrame:
    # Une por Fecha-Item y capa Ventas_dia a Hi (si disponible)
    j=df_daily.merge(limits[["Fecha","Item","Hi"]],on=["Fecha","Item"],how="left")
    if "Ventas_dia" in j.columns and "Hi" in j.columns:
        j["Ventas_dia"]=j.apply(lambda r: min(r["Ventas_dia"], r["Hi"]) if pd.notna(r["Hi"]) else r["Ventas_dia"], axis=1)
    return j