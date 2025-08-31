# Lógica de cálculo

- Modo diario: pedido cada día, redondeo a múltiplo, arrastre stock.
- Modo bloques: pedido solo en inicio de bloque; Demanda_B + Target_prox; redondeo único; arrastre.
- Target efectivo: override TARGET_DIA si existe; si no, max(CFG_TARGET, Cobertura AM).
- Cobertura AM: CFG_ENABLE_AM_COVERAGE y CFG_AM_FRAC × esperado del día siguiente.
