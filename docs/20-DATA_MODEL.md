# Modelo de datos

## CFG (hoja)
- B2: CFG_TARGET (≥0)
- B3: CFG_MULTIPLO (≥1)
- B4: CFG_ENABLE_AM_COVERAGE (bool)
- B5: CFG_AM_FRAC ∈ [0,1]
- B6: CFG_ITEM_FILTER (texto)
- B7:H7 encabezados [Lun..Dom]; B8:H8 bits 1/0 → CFG_DELIVERY_WEEK (named range)

## TARGET_DIA (opcional)
- Cols: Fecha (dd/mm/aaaa), Target

## ENTREGAS_EXC (opcional)
- Cols: Fecha (dd/mm/aaaa), Entrega ∈ {0,1}

## PEDIDOS (tabla base)
- Fecha, Ítem, Esperado, Stock_ini, Pedido, Stock_fin, …
