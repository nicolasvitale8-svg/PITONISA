# ADR-0004: Target override por día

- Decisión: hoja TARGET_DIA permite fijar Target por fecha.
- Motivo: eventos/picos que requieren cubrir más/menos.
- Consecuencia: Target_efectivo = override si existe; si no, max(CFG_TARGET, AM).
