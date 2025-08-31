import math, datetime as dt, pandas as pd


def _ceil(x, m):
    m = max(1, int(m or 1))
    return 0 if x <= 0 else int(math.ceil(float(x) / m) * m)


def _tauto(i, E, on, frac):
    return (frac * float(E[i + 1])) if on and i + 1 < len(E) and E[i + 1] is not None else 0.0


def _teff(i, fechas, E, ovr, cfg):
    f = fechas[i]
    k = (f.date() if isinstance(f, dt.datetime) else f).isoformat()
    return (
        float(ovr[k])
        if (ovr and k in ovr and ovr[k] is not None)
        else max(float(cfg.CFG_TARGET), _tauto(i, E, cfg.CFG_ENABLE_AM_COVERAGE, cfg.CFG_AM_FRAC))
    )


def _blocks(fechas, cal):
    n = len(fechas)
    if not cal:
        return [(i, i + 1) for i in range(n)]

    def isdel(d):
        d0 = d.date() if isinstance(d, dt.datetime) else d
        return int(cal.is_delivery(d0)) == 1

    starts = [i for i in range(n) if isdel(fechas[i])] or [0]
    bl = [(s, (starts[j + 1] if j + 1 < len(starts) else n)) for j, s in enumerate(starts)]
    if bl[0][0] != 0:
        bl = [(0, bl[0][0])] + bl
    return bl


def build_weekly_orders_smoke(df, cfg, overrides, cal):
    df = df.sort_values("fecha").reset_index(drop=True)
    fechas = pd.to_datetime(df["fecha"]).tolist()
    E = [float(x or 0.0) for x in df["esperado"].tolist()]
    si = float(df["stock_ini"].iloc[0])  # inventario real del primer día
    # auto: bloques si el calendario tiene algún 0 o excepciones; si no, diario
    use_blocks = (cal is not None) and (
        (hasattr(cal, "week_bits") and 0 in getattr(cal, "week_bits")) or getattr(cal, "exceptions", {})
    )
    pedido = [0] * len(E)
    sf = [0.0] * len(E)
    s = si
    if use_blocks:
        for s0, e in _blocks(fechas, cal):
            dem = sum(E[j] for j in range(s0, e))
            tb = _teff(e - 1, fechas, E, overrides, cfg)
            pb = _ceil(max(0.0, tb + dem - s), cfg.CFG_MULTIPLO)
            if pb > 0:
                pedido[s0] = pb
            for i in range(s0, e):
                add = pb if i == s0 else 0
                s = s + add - E[i]
                sf[i] = s
    else:
        for i in range(len(E)):
            te = _teff(i, fechas, E, overrides, cfg)
            p = _ceil(max(0.0, te + E[i] - s), cfg.CFG_MULTIPLO)
            pedido[i] = p
            s = s + p - E[i]
            sf[i] = s
    df["pedido"] = pedido
    df["stock_fin"] = sf
    df["stock_ini"] = [si] + [sf[i - 1] for i in range(1, len(sf))]
    return df

