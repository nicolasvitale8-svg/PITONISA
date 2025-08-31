# Notas / Especificación – PITONISA

**Fecha:** 2025-08-30
**Autor:** Codex
**Origen:** contenido pegado por el usuario

## Contenido

### OBJETIVO – Parseo estricto de fechas

- Parsear SIEMPRE como D/M/YYYY (día/mes/año) sin heurísticas ni dateutil.
- Quitar el warning "Could not infer format..." y acelerar el parseo.
- Mantener salida consistente dd/mm/aaaa para mostrar.

#### Helpers estrictos
```
DATE_FMT = "%d/%m/%Y"
DATE_REGEX = re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{4}\s*$")

def parse_fecha_dmY(s: pd.Series) -> pd.Series:
    t = _clean_str_series(s)
    ok = t.str.match(DATE_REGEX)
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    if ok.any():
        out.loc[ok] = pd.to_datetime(t[ok], format=DATE_FMT, errors="coerce")
    return out

def format_fecha_dmY(dt_series: pd.Series) -> pd.Series:
    return dt_series.dt.strftime("%d/%m/%Y")
```

### OBJETIVO – Auditoría de formato de fechas

- Auditar que las fechas cumplan D/M/YYYY (1–2 dígitos día/mes; 4 dígitos año).
- Mostrar aviso en UI con conteo por hoja/columna y ejemplos.
- Permitir descargar CSV con todos los casos detectados para corrección.
- Reusar DATE_REGEX/DATE_FMT existentes y NO cambiar el parseo estricto.

#### Helpers de auditoría
```
def _is_dmY_series(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.strip()
    return t.str.match(DATE_REGEX)

def audit_dmY(df: pd.DataFrame, sheet_name: str, col_name: str, max_examples: int = 5) -> dict:
    if col_name not in df.columns:
        return {"sheet": sheet_name, "column": col_name, "total": 0, "bad": 0, "examples": []}
    col = df[col_name]
    ok = _is_dmY_series(col)
    bad_mask = (~ok).fillna(True)
    bad_count = int(bad_mask.sum())
    total = int(len(col))
    examples = col[bad_mask].dropna().astype(str).unique().tolist()[:max_examples]
    return {"sheet": sheet_name, "column": col_name, "total": total, "bad": bad_count, "examples": examples}
```

---
