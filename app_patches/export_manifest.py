import json
import datetime as dt


def export_manifest(cfg: dict, mode: str, formulas: dict, path: str = "manifest/sample.manifest.json"):
    doc = {
        "version": "0.2.0",
        "generated_at": dt.datetime.utcnow().isoformat(),
        "cfg": {
            "CFG_TARGET": cfg.get("CFG_TARGET", 0),
            "CFG_MULTIPLO": cfg.get("CFG_MULTIPLO", 1),
            "CFG_ENABLE_AM_COVERAGE": bool(cfg.get("CFG_ENABLE_AM_COVERAGE", True)),
            "CFG_AM_FRAC": float(cfg.get("CFG_AM_FRAC", 0.5)),
            "CFG_ITEM_FILTER": cfg.get("CFG_ITEM_FILTER", ""),
            "CFG_DELIVERY_WEEK": [int(x) for x in cfg.get("CFG_DELIVERY_WEEK", [1, 1, 1, 1, 1, 1, 1])],
        },
        "calculated_logic": {
            "mode": mode,
            "formulas": formulas,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return path


FORMULAS = {
    "daily": {
        "Pedido_bruto": "MAX(0, Target_ef + Esperado - Stock_ini)",
        "Pedido": "ROUNDUP(Pedido_bruto/MULT) * MULT",
        "Stock_fin": "Stock_ini + Pedido - Esperado",
        "Target_auto": "AM_ON ? AM_FRAC * Esperado[d+1] : 0",
        "Target_ef": "override(Target_dia) || MAX(CFG_TARGET, Target_auto)",
    },
    "blocks": {
        "Demanda_B": "SUM(Esperado[d..prox-1])",
        "Target_prox": "Target_ef[prox-1]",
        "Pedido_bruto_B": "MAX(0, Target_prox + Demanda_B - Stock_ini[d])",
        "Pedido_B": "ROUNDUP(Pedido_bruto_B/MULT) * MULT (solo en d)",
    },
}

