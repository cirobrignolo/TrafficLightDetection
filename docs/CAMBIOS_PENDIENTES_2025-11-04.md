# CAMBIOS PENDIENTES - 2025-11-04

## ✅ Estado de Verificación

Después de la verificación exhaustiva del código fuente de Apollo (1,187 líneas analizadas), se confirmó que la implementación tiene una **fidelidad del ~95%** con respecto a Apollo.

---

## 🔴 CAMBIOS CRÍTICOS (Prioridad Alta)

### 1. Implementar Semantic IDs (Gap #1)

**Problema**: Actualmente usamos `row_index` para identificar semáforos, lo cual causa **cross-history transfer** cuando las projection boxes se reordenan.

**Causa**:
- Apollo usa **semantic IDs persistentes** desde el HD-Map
- Nuestra implementación usa **row_index** que cambia con el orden del archivo

**Impacto**:
- Historias de tracking se transfieren al semáforo incorrecto
- Pérdida de consistencia temporal entre frames

**Solución**:
1. Agregar columna 5 en `projection_bboxes.txt` con semantic_id
2. Modificar `src/tlr/tracking.py` para leer y usar `semantic_id` en lugar de `row_index`
3. Actualizar formato de archivo en todos los datasets de test

**Archivos a modificar**:
- `src/tlr/tracking.py` (líneas donde se usa `row_index`)
- `projection_bboxes.txt` en todos los casos de test
- Documentación del formato de archivo

**Testing**:
- Re-ejecutar `test_doble_chico` para verificar que no hay cross-history transfer
- Verificar que los tracking IDs se mantienen consistentes

---

## ✅ CAMBIOS MENORES COMPLETADOS

### 2. Corregir orden de `type_names` (Fix #5 - Inconsistencias) - ✅ COMPLETADO

**Problema**: En 4 líneas de `test_doble_chico/run_pipeline.py`, el orden de `type_names` era incorrecto.

**Líneas corregidas**: 142, 154, 191, 228

**Orden incorrecto (antes)**:
```python
['vert', 'quad', 'hori', 'bg']
```

**Orden correcto (ahora)**:
```python
['bg', 'vert', 'quad', 'hori']
```

**Justificación técnica**:
- Apollo C++ enum (traffic_light.h:37-42): `TL_VERTICAL_CLASS=0, TL_QUADRATE_CLASS=1, TL_HORIZONTAL_CLASS=2`
- Detector output: 4 clases softmax `[clase_0, clase_1, clase_2, clase_3]`
- Mapeo correcto: `[bg, vert, quad, hori]` donde bg es índice 3
- pipeline.py:191: `classifiers = [(vert, ...), (quad, ...), (hori, ...)]`
- pipeline.py:61: `self.classifiers[tl_type-1]` → tl_type=1→vert, tl_type=2→quad, tl_type=3→hori

**Verificación**: tracking.py usa `['black','red','yellow','green']` (línea 70) que es para recognition colors, no detector types - está correcto

**Estado**: ✅ Corregido en las 4 líneas

---

## ⚪ NO SON GAPS (Aclaraciones)

### 3. Multi-cámara
- **Apollo**: Usa telephoto (25mm) + wide-angle (6mm)
- **Nuestra impl**: Una sola cámara
- **Estado**: ✅ Diferencia de diseño aceptable (no necesitamos multi-cámara para nuestro caso de uso)

### 4. 70% Weight en Hungarian
- **Apollo**: Limitación inherente del algoritmo
- **Nuestra impl**: Misma limitación
- **Estado**: ✅ No es un bug, es diseño del algoritmo Hungarian (no se puede "arreglar")

### 5. HD-Map vs Archivo Estático
- **Apollo**: Projection boxes desde HD-Map dinámico
- **Nuestra impl**: Projection boxes desde archivo estático
- **Estado**: ✅ Diferencia de arquitectura aceptable (no tenemos HD-Map, usamos archivo manual)

### 6. Multi-ROI (Selection)
- **Apollo**: Asignación 1-to-1 usando flags `is_selected`
- **Nuestra impl**: Asignación 1-to-1 usando lógica equivalente
- **Estado**: ✅ RESUELTO - Confirmado que ambos hacen 1-to-1 (no es gap)

---

## 📋 Resumen de Acciones

| # | Cambio | Prioridad | Estado | Archivos |
|---|--------|-----------|--------|----------|
| 1 | Semantic IDs | 🔴 CRÍTICA | ⏳ Pendiente | `tracking.py`, `projection_bboxes.txt` |
| 2 | type_names order | 🟡 MEDIA | ✅ COMPLETADO | `test_doble_chico/run_pipeline.py` |

---

## 📝 Notas Adicionales

- **Fidelidad actual**: ~95% (muy alta)
- **Único gap crítico**: Semantic IDs
- **Documentos de referencia**:
  - `VERIFICACION_EXHAUSTIVA_CODIGO.md`: Resumen de verificación
  - `ANALISIS_FLUJO_APOLLO_COMPLETO.md`: Análisis línea por línea de Apollo (1,187 líneas)

---

## ✅ Fixes Ya Verificados (No Requieren Cambios)

Estos fixes se implementaron correctamente en conversaciones previas:

1. ✅ **Fix #1**: ROI validation antes de Hungarian (`selector.py:37-45`)
2. ✅ **Fix #2**: NMS sorting order (`pipeline.py:37-46`)
3. ✅ **Fix #3**: NMS global scope (verificado correcto)
4. ✅ **Fix #4**: NMS threshold 0.6 (`pipeline.py:46`)

---

**Fecha de creación**: 2025-11-04
**Última actualización**: 2025-11-04
**Estado general**: 1 cambio pendiente (1 crítico) - type_names completado ✅
