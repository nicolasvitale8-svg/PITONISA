# ADR-0001: Fechas D/M/YYYY estrictas

- Decisión: parseo solo D/M/YYYY; no heurísticas ni dateutil.
- Motivo: consistencia, performance, evitar warnings.
- Consecuencia: valores fuera de formato quedan NaT y se auditan.
