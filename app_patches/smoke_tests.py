def run_smokes(calc_fn, seed_df_builder):
    """
    calc_fn(df_item, cfg, target_override, calendar) -> df_item con 'pedido'
    seed_df_builder() -> (df_item, cfg, target_override, calendar)
    """
    df, cfg, overrides, cal = seed_df_builder(multiplo=1)
    out = calc_fn(df, cfg, overrides, cal)
    s = int(out["pedido"].sum())
    assert s == 91, f"Smoke A fail: {s}"

    df, cfg, overrides, cal = seed_df_builder(multiplo=6)
    out = calc_fn(df, cfg, overrides, cal)
    s = int(out["pedido"].sum())
    assert s == 96, f"Smoke B fail: {s}"

    # bloques M/MI/V
    df, cfg, overrides, cal = seed_df_builder(multiplo=1, week_bits=[1, 0, 1, 1, 1, 1, 1])
    out = calc_fn(df, cfg, overrides, cal)
    s = int(out["pedido"].sum())
    assert s == 91, f"Smoke C fail: {s}"

    df, cfg, overrides, cal = seed_df_builder(multiplo=6, week_bits=[1, 0, 1, 1, 1, 1, 1])
    out = calc_fn(df, cfg, overrides, cal)
    s = int(out["pedido"].sum())
    assert s == 96, f"Smoke D fail: {s}"
    return "OK"

