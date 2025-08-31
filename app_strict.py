import io
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from datetime import datetime, date
from dataclasses import dataclass
import datetime as dt
import math
import time
from app_patches.imputation import impute_day
from app_patches.train_ewma import predict_next

import numpy as np
import pandas as pd
import re
import warnings
import streamlit as st


# ---------------- Utils ----------------
warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually",
)

DATE_FMT = "%d/%m/%Y"
DATE_REGEX = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$")

def _clean_str_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace("\u200b", "", regex=False)
        .str.strip()
        .str.replace(r"[^0-9/\-:\.\s]", "", regex=True)
    )

def parse_fecha_dmY(s: pd.Series) -> pd.Series:
    t = _clean_str_series(s)
    ok = t.str.match(DATE_REGEX)
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    if ok.any():
        out.loc[ok] = pd.to_datetime(t[ok], format=DATE_FMT, errors="coerce")
    return out

def parse_hora_HM(s: pd.Series) -> pd.Series:
    t = _clean_str_series(s)
    return pd.to_datetime(t, format="%H:%M", errors="coerce")

def format_fecha_dmY(dt_series: pd.Series) -> pd.Series:
    return dt_series.dt.strftime("%d/%m/%Y")

def week_of_month(dt: pd.Timestamp) -> int:
    return int((dt.day - 1) // 7) + 1

def ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.tz_localize(None) if getattr(series.dt, "tz", None) is not None else series
    return parse_fecha_dmY(series)

def sanitize_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        s = out[col]
        try:
            if pd.api.types.is_datetime64_any_dtype(s):
                out[col] = s.dt.strftime("%d/%m/%Y")
                continue
            if s.dtype == "object":
                if s.map(lambda x: isinstance(x, (pd.Timestamp, datetime, date, np.datetime64))).any():
                    s2 = format_fecha_dmY(parse_fecha_dmY(s))
                    out[col] = s2.fillna("")
                else:
                    out[col] = s.astype(str)
                continue
            if pd.api.types.is_categorical_dtype(s):
                out[col] = s.astype(str)
                continue
            if pd.api.types.is_integer_dtype(s):
                out[col] = pd.to_numeric(s, errors="coerce").astype("Int64")
                continue
            if pd.api.types.is_float_dtype(s):
                out[col] = pd.to_numeric(s, errors="coerce")
                continue
        except Exception:
            out[col] = s.astype(str)
    return out

def _read_bytes_retry(path: Path, retries: int = 5, delay_sec: float = 0.2) -> bytes:
    last_err: Optional[Exception] = None
    for _ in range(max(1, int(retries))):
        try:
            return path.read_bytes()
        except Exception as e:
            last_err = e
            time.sleep(max(0.0, float(delay_sec)))
    return path.read_bytes()

# ---------------- Audit helpers ----------------
def _is_dmY_series(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip()
    return t.str.match(DATE_REGEX)

def audit_dmY(df: pd.DataFrame, sheet_name: str, col_name: str, max_examples: int = 5) -> Dict[str, object]:
    if df is None or df.empty or col_name not in df.columns:
        return {"sheet": sheet_name, "column": col_name, "total": 0, "bad": 0, "examples": []}
    col = df[col_name]
    ok = _is_dmY_series(col)
    bad_mask = (~ok).fillna(True)
    bad_count = int(bad_mask.sum())
    total = int(len(col))
    examples = col[bad_mask].dropna().astype(str).unique().tolist()[:max_examples]
    return {"sheet": sheet_name, "column": col_name, "total": total, "bad": bad_count, "examples": examples}

def audit_dmY_many(targets: List[Tuple[pd.DataFrame, str, str]]) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    report: List[Dict[str, object]] = []
    rows: List[pd.DataFrame] = []
    for df, sheet, col in targets:
        if df is None or df.empty or col not in df.columns:
            report.append({"sheet": sheet, "column": col, "total": 0, "bad": 0, "examples": []})
            continue
        res = audit_dmY(df, sheet, col)
        report.append(res)
        if res["bad"] > 0:
            mask = (~_is_dmY_series(df[col])).fillna(True)
            tmp = df.loc[mask, [col]].copy()
            tmp.insert(0, "sheet", sheet)
            tmp.insert(1, "column", col)
            tmp = tmp.rename(columns={col: "value"})
            tmp.insert(3, "row_index", tmp.index)
            rows.append(tmp)
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sheet", "column", "value", "row_index"])
    return report, detail

def _is_HM_series(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip()
    return t.str.match(re.compile(r"^\s*([01]?[0-9]|2[0-3]):[0-5][0-9]\s*$"))

def audit_HM(df: pd.DataFrame, sheet_name: str, col_name: str, max_examples: int = 5) -> Dict[str, object]:
    if df is None or df.empty or col_name not in df.columns:
        return {"sheet": sheet_name, "column": col_name, "total": 0, "bad": 0, "examples": []}
    col = df[col_name]
    ok = _is_HM_series(col)
    bad_mask = (~ok).fillna(True)
    bad_count = int(bad_mask.sum())
    total = int(len(col))
    examples = col[bad_mask].dropna().astype(str).unique().tolist()[:max_examples]
    return {"sheet": sheet_name, "column": col_name, "total": total, "bad": bad_count, "examples": examples}

def audit_HM_many(targets: List[Tuple[pd.DataFrame, str, str]]) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
    report: List[Dict[str, object]] = []
    rows: List[pd.DataFrame] = []
    for df, sheet, col in targets:
        if df is None or df.empty or col not in df.columns:
            report.append({"sheet": sheet, "column": col, "total": 0, "bad": 0, "examples": []})
            continue
        res = audit_HM(df, sheet, col)
        report.append(res)
        if res["bad"] > 0:
            mask = (~_is_HM_series(df[col])).fillna(True)
            tmp = df.loc[mask, [col]].copy()
            tmp.insert(0, "sheet", sheet)
            tmp.insert(1, "column", col)
            tmp = tmp.rename(columns={col: "value"})
            tmp.insert(3, "row_index", tmp.index)
            rows.append(tmp)
    detail = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sheet", "column", "value", "row_index"])
    return report, detail

def get_ventas_train(df_imput, use_imputed: bool):
    import pandas as pd
    if df_imput is None or df_imput.empty:
        return pd.DataFrame(columns=["Fecha","Item","Ventas_train"])
    base = "Ventas_imp" if (use_imputed and "Ventas_imp" in df_imput.columns) else "Ventas_obs"
    return df_imput[["Fecha","Item",base]].rename(columns={base:"Ventas_train"})

def load_esperado_dia() -> pd.DataFrame:
    # 1) CSV preferente si existe
    p_csv = Path("import/ESPERADO_DIA.csv")
    if p_csv.exists():
        df = pd.read_csv(p_csv, encoding="utf-8")
    else:
        # 2) Excel: misma planilla que ya usás
        xls_path = Path(__file__).parent / "PITONISA.xlsx"
        try:
            if xls_path.exists():
                df = pd.read_excel(xls_path, sheet_name="ESPERADO_DIA")
            else:
                df = pd.DataFrame()
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=["Fecha","Item","Esperado_dia"])

    ren = {"fecha":"Fecha","item":"Item","esperado_dia":"Esperado_dia","esperado":"Esperado_dia"}
    df = df.rename(columns={k:v for k,v in ren.items() if k in df.columns and v not in df.columns})

    if "Fecha" in df.columns:
        df["Fecha"] = parse_fecha_dmY(df["Fecha"]).dt.strftime("%d/%m/%Y")
    df["Item"] = df.get("Item","").astype(str)
    df["Esperado_dia"] = pd.to_numeric(df.get("Esperado_dia", 0), errors="coerce").fillna(0).clip(lower=0)
    need = {"Fecha","Item","Esperado_dia"}
    if not need.issubset(df.columns):
        return pd.DataFrame(columns=["Fecha","Item","Esperado_dia"])
    return df[["Fecha","Item","Esperado_dia"]]

def _esperado_auto(df_diario: pd.DataFrame, df_imput: pd.DataFrame) -> pd.DataFrame:
    """
    Construye ESPERADO_DIA con:
      1) Ventas_imp (si existe) por Fecha-Item
      2) sino, sum(Ventas) observadas del DIARIO
      3) sino, 0
    """
    if df_diario is None or df_diario.empty:
        return pd.DataFrame(columns=["Fecha","Item","Esperado_dia"])

    # base de combinaciones Fecha-Item desde DIARIO
    base = (df_diario.rename(columns={"fecha":"Fecha","item":"Item","ventas":"Ventas"})
                    .loc[:, ["Fecha","Item"]].dropna().drop_duplicates())

    # ventas observadas por día
    obs = (df_diario.rename(columns={"fecha":"Fecha","item":"Item","ventas":"Ventas"})
                   .groupby(["Fecha","Item"], as_index=False)["Ventas"].sum()
                   .rename(columns={"Ventas":"_obs"}))

    out = base.merge(obs, on=["Fecha","Item"], how="left")

    # si tenemos resumen de imputación, úsalo
    if df_imput is not None and not df_imput.empty:
        imp = df_imput.loc[:, ["Fecha","Item","Ventas_imp"]].rename(columns={"Ventas_imp":"_imp"})
        out = out.merge(imp, on=["Fecha","Item"], how="left")

    out["_obs"] = pd.to_numeric(out["_obs"], errors="coerce").fillna(0)
    out["_imp"] = pd.to_numeric(out.get("_imp", 0), errors="coerce").fillna(0)

    out["Esperado_dia"] = out["_imp"].where(out["_imp"] > 0, out["_obs"])
    return out[["Fecha","Item","Esperado_dia"]]

def _save_esperado_csv(df: pd.DataFrame, path="import/ESPERADO_DIA.csv", keep_existing=True):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if keep_existing and p.exists():
        old = pd.read_csv(p, encoding="utf-8")
        old = old.rename(columns={"fecha":"Fecha","item":"Item","esperado_dia":"Esperado_dia"})
        # no sobrescribir valores existentes no nulos/positivos
        m = pd.merge(df, old, on=["Fecha","Item"], how="outer", suffixes=("", "_old"))
        m["Esperado_dia"] = m["Esperado_dia_old"].fillna(m["Esperado_dia"])
        df = m[["Fecha","Item","Esperado_dia"]]
    df.to_csv(p, index=False, encoding="utf-8")
    return str(p)

# ... (resto de las funciones de la app) ...

# ---------------- UI ----------------
st.set_page_config(page_title="MADAME CEPHALPOD ORACLE", layout="wide")

# ... (código de la UI y lógica principal) ...

with st.sidebar:
    st.header("Entrada")
    st.caption("Usando archivo local")
    st.code("PITONISA.xlsx")
    alpha = st.number_input("Suavizado de Laplace (alpha)", min_value=0.0, value=1.0, step=0.5)
    st.subheader("Parametros")
    use_dow = st.toggle("Condicionar por dia de semana", value=True)
    use_wom = st.toggle("Condicionar por semana del mes", value=True)
    window_days = st.number_input("Ventana (dias hacia atras)", min_value=0, value=120, step=7, help="0 = usar todo el historial")
    _p = Path(__file__).parent / "PITONISA.xlsx"
    _t_def, _m_def, _am_on, _am_frac, _item_filter_def = (0.0, 1, True, 0.5, "")
    try:
        if _p.exists():
            _t_def, _m_def, _am_on, _am_frac, _item_filter_def = read_cfg_defaults(_read_bytes_retry(_p))
    except Exception:
        _t_def, _m_def, _am_on, _am_frac, _item_filter_def = (0.0, 1, True, 0.5, "")
    target_stock = st.number_input(
        "Target de stock al cierre",
        min_value=0.0,
        value=float(_t_def),
        step=1.0,
        help="Stock objetivo al cierre del día (0 = terminar justo).",
    )
    order_multiple = st.number_input(
        "Multiplo de pedido",
        min_value=1,
        value=int(_m_def),
        step=1,
        help="Tamaño de caja/pack (≥1); se redondea hacia arriba a este múltiplo.",
    )
    enable_am = st.toggle("Cobertura mañana (AM)", value=bool(_am_on), help="Si está activo, cubre una fracción del esperado del día siguiente como stock objetivo.")
    am_frac = st.number_input("Fracción AM (0-1)", min_value=0.0, max_value=1.0, value=float(_am_frac), step=0.05)
    item_filter = st.text_input("Filtro por ítem (opcional)", value=str(_item_filter_def or ""))
    enable_blocks = st.toggle("Modo bloques", value=False, help="Activa el cálculo de pedidos por bloques de entrega. Si está desactivado, los pedidos se calculan diariamente.")
    USE_IMPUTED = st.toggle(
        "Usar Demanda Imputada (quiebres)",
        value=False,
        help="Reemplaza ventas observadas por ventas imputadas al entrenar/KPI; no altera el cálculo de pedidos."
    )
    with st.form("cfg_form", clear_on_submit=False):
        st.subheader("Calendario semanal de entregas")
        st.caption("1 = hay entrega; 0 = sin entrega")
        days_names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        cols_week = st.columns(7)
        week_bits_selected: List[int] = []
         # Redundant, but guarantees definition
        for i, name in enumerate(days_names):
            with cols_week[i]:
                val = st.toggle(name, value=bool(week_bits_default[i]), key=f"dow_{i}")
                week_bits_selected.append(1 if val else 0)
        colA, colB = st.columns(2)
        save = colA.form_submit_button("Guardar CFG")
        reset = colB.form_submit_button("Reset CFG")
        if save:
            ok = write_cfg_to_file(
                _p,
                target_stock,
                order_multiple,
                enable_am=enable_am,
                am_frac=am_frac,
                item_filter=item_filter,
                delivery_week=week_bits_selected,
            )
            st.success("CFG guardado" if ok else "No se pudo guardar CFG")
            st.session_state["do_recalc"] = False
            st.info("Cambios guardados. Pulsa 'Aplicar y recalcular'.")
        if reset:
            ok = write_cfg_to_file(
                _p,
                0.0,
                1,
                enable_am=True,
                am_frac=0.5,
                item_filter="",
                delivery_week=[1, 1, 1, 1, 1, 1, 1],
            )
            st.success("CFG reseteado" if ok else "No se pudo guardar CFG")
            st.session_state["do_recalc"] = False
            st.info("CFG por defecto cargado. Pulsa 'Aplicar y recalcular'.")

    if st.button("Aplicar y recalcular", type="primary"):
        st.session_state["do_recalc"] = True
    btn_run = st.button("Calcular prediccion")
    st.divider()
    st.subheader("CSV (opcional)")
    st.caption("Importa un CSV (Fecha, Item, Esperado, Stock inicial opcional).")
    csv_file = st.file_uploader("Subir CSV", type=["csv"])
    if 'df_ventas_train' in locals() and not df_ventas_train.empty and st.button("Exportar dataset de entrenamiento"):
        p = "export/training_ventas.csv"
        Path("export").mkdir(exist_ok=True)
        df_ventas_train.to_csv(p, index=False, encoding="utf-8")
        st.success(f"Dataset exportado: {p}")

if st.button("⚙️ Generar/actualizar ESPERADO_DIA (auto)"):
    gen = _esperado_auto(diario_df, df_imput)
    if gen.empty:
        st.error("No pude generar ESPERADO_DIA: DIARIO vacío.")
    else:
        path_csv = _save_esperado_csv(gen, "import/ESPERADO_DIA.csv", keep_existing=True)
        st.success(f"ESPERADO_DIA actualizado: {path_csv}")
        st.dataframe(gen.head(200), use_container_width=True)

if st.button("📦 Exportar baseline → ESPERADO_DIA"):
    # leer modelo
    try:
        with open("models/model_ewma.json","r",encoding="utf-8") as f:
            model = json.load(f)
        pred = predict_next(model)  # Item, Esperado_dia
        if pred.empty:
            st.warning("Modelo vacío; entrená primero.")
        else:
            # Fecha = día siguiente al último presente en df_ventas_train
            ult = pd.to_datetime(df_ventas_train["Fecha"], dayfirst=True, errors="coerce").max()
            next_day = (ult + pd.Timedelta(days=1)).strftime("%d/%m/%Y") if pd.notna(ult) else pd.Timestamp.today().strftime("%d/%m/%Y")
            pred.insert(0, "Fecha", next_day)
            path_csv = _save_esperado_csv(pred, "import/ESPERADO_DIA.csv", keep_existing=True)
            st.success(f"Predicción baseline exportada a {path_csv} (Fecha {next_day})")
            st.dataframe(pred, use_container_width=True)
    except FileNotFoundError:
        st.error("No encontré models/model_ewma.json — entrená el baseline primero.")
    with st.sidebar.expander(" Notas / Especificación"):
        st.markdown(load_notes_md(), unsafe_allow_html=False)

xlsx_path = Path(__file__).parent / "PITONISA.xlsx"
if not xlsx_path.exists():
    st.error(f"No se encontro el archivo {xlsx_path.name} en la carpeta de la app.")
    st.stop()

st.success(f"Excel local detectado: {xlsx_path.name}")
try:
    if "do_recalc" not in st.session_state:
        st.session_state["do_recalc"] = True

    file_bytes = _read_bytes_retry(xlsx_path)
    _xls = pd.ExcelFile(io.BytesIO(file_bytes))
    d2 = try_read_pedido_date(_xls)
    default_date = (d2.date() if d2 is not None else (pd.Timestamp.today() + pd.Timedelta(days=1)).date())
    target_date = st.date_input("Fecha objetivo", value=default_date)
    target_ts = pd.Timestamp(target_date)

    weekly_grid = pd.DataFrame(); weekly_orders = pd.DataFrame(); probs_used = pd.DataFrame(); ventas_df = pd.DataFrame(); diario_df = pd.DataFrame(); maps = {}; stock_df = pd.DataFrame(); Q = pd.DataFrame()
    if st.session_state.get("do_recalc", True):
        with st.spinner("Recalculando predicción y pedidos..."):
            ventas_df, diario_df, maps, probs_used, counts_used, stock_df, Q = detect_and_prepare_strict(
                file_bytes, alpha, target_ts, window_days, use_dow, use_wom
            )
            st.subheader("Matriz de probabilidades")
            st.caption("Orden de estados fijo: " + ", ".join(STATES))
            st.dataframe(sanitize_for_display(probs_used.round(2)))

            try:
                xls_view = pd.ExcelFile(io.BytesIO(file_bytes))
                if "PRED_VENTA" in xls_view.sheet_names:
                    pv = xls_view.parse("PRED_VENTA", header=None)
                    pv_view = pv.iloc[1:13, 0:9]
                    st.subheader("PREDICCION DE VENTA SEMANAL")
                    st.dataframe(sanitize_for_display(pv_view))
            except Exception:
                pass

            weekly_grid = build_weekly_grid(Q, file_bytes, stock_df)
            weekly_orders = build_weekly_orders(
                Q,
                file_bytes,
                stock_df,
                order_multiple=order_multiple,
                target=target_stock,
                enable_am=enable_am,
                am_frac=am_frac,
            )
        st.session_state["do_recalc"] = False
    else:
        st.info("Cambios guardados. Pulsa 'Aplicar y recalcular' para actualizar resultados.")

    if 'diario_df' in locals() and diario_df is not None and not diario_df.empty:
        df_diario_impute = diario_df.copy()
        cols_map = {"fecha":"Fecha", "hora":"Hora", "item":"Item", "stock":"Stock", "ventas":"Ventas"}
        for k, v in cols_map.items():
            if k in df_diario_impute.columns and v not in df_diario_impute.columns:
                df_diario_impute.rename(columns={k:v}, inplace=True)

        required_cols_impute = ["Fecha", "Item", "Hora", "Stock", "Ventas"]
        if all(col in df_diario_impute.columns for col in required_cols_impute):
            rows = []
            for (f, it), g in df_diario_impute.groupby(["Fecha", "Item"], dropna=False, sort=False):
                g = g.sort_values("Hora")
                ventas_imp, perdidas, spans = impute_day(
                    g, time_col="Hora", stock_col="Stock", sales_col="Ventas",
                    expected_total=None
                )
                ventas_obs = float(g["Ventas"].fillna(0).sum())
                rows.append({
                    "Fecha": f, "Item": it,
                    "Ventas_obs": ventas_obs, "Perdidas_est": perdidas,
                    "Ventas_imp": ventas_imp, "Spans": len(spans)
                })
            df_imput = pd.DataFrame(rows)

            with st.expander("🧩 Imputación por quiebres — Resumen"):
                st.dataframe(df_imput.sort_values(["Fecha", "Item"]).head(300), use_container_width=True)
                st.caption("Usar imputación: " + ("ON" if USE_IMPUTED else "OFF"))

            if not df_imput.empty:
                topq = df_imput.sort_values("Perdidas_est", ascending=False).head(20)
                with st.expander("🔥 Top quiebres (perdidas estimadas)"):
                    st.dataframe(topq, use_container_width=True)
        else:
            with st.expander("🧩 Imputación por quiebres — Resumen"):
                st.warning("No se pudo generar el resumen de imputación. Faltan columnas en la hoja DIARIO (se esperan: Fecha, Hora, Item, Stock, Ventas).")
            df_imput = pd.DataFrame(columns=["Fecha", "Item", "Ventas_obs", "Perdidas_est", "Ventas_imp", "Spans"])
    else:
        df_imput = pd.DataFrame(columns=["Fecha", "Item", "Ventas_obs", "Perdidas_est", "Ventas_imp", "Spans"])

    df_ventas_train = get_ventas_train(df_imput, USE_IMPUTED)
    df_esperado_dia = load_esperado_dia()
    with st.expander("📄 Fuente ESPERADO_DIA"):
        st.dataframe(df_esperado_dia.head(200), use_container_width=True)
        st.caption("Origen: CSV import/ESPERADO_DIA.csv si existe; si no, hoja Excel ESPERADO_DIA.")

    if not df_esperado_dia.empty and not df_ventas_train.empty:
        df_join = pd.merge(df_ventas_train, df_esperado_dia, on=["Fecha","Item"], how="inner")
        df_kpi = df_join.copy()
        df_kpi["ae"] = (df_kpi["Ventas_train"] - df_kpi["Esperado_dia"]).abs()
        WAPE = float(df_kpi["ae"].sum()) / max(float(df_kpi["Esperado_dia"].sum()), 1e-9)
        Bias = float((df_kpi["Ventas_train"] - df_kpi["Esperado_dia"]).mean())
        df_kpi["fill"] = (df_kpi[["Ventas_train","Esperado_dia"]].min(axis=1) / df_kpi["Esperado_dia"]).clip(0,1)
        Fill = float(df_kpi["fill"].mean())
        with st.expander("📊 KPIs (Ventas_train vs Esperado_dia)"):
            st.write({"WAPE": round(WAPE,4), "Bias": round(Bias,2), "FillRate": round(Fill,4)})
    else:
        st.info("Falta ESPERADO_DIA (Excel/CSV) o resumen de ventas para calcular KPIs.")

    if not weekly_grid.empty:
        st.subheader("Prediccion semanal por item (unidades)")
        st.dataframe(sanitize_for_display(weekly_grid))
    if not weekly_orders.empty:
        st.subheader("Pedido sugerido semanal (unidades)")
        st.dataframe(sanitize_for_display(weekly_orders))
    else:
        st.info("No hay pedido semanal para mostrar (revisa PRED_VENTA y que los items de PEDIDO_CP coincidan con DIARIO).")

    # ... (resto del código)
except Exception as e:
    st.error(f"Error procesando el archivo: {e}")
    st.stop()