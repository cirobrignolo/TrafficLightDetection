# Propuesta: Implementación de Semantic IDs para Tracking Robusto

**Fecha**: 2025-11-08
**Autor**: Ciro J.B.
**Para**: Revisión con profesores

---

## 📋 ÍNDICE

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Contexto: Sistema Apollo Original](#contexto-sistema-apollo-original)
3. [Implementación Actual (Row Index)](#implementación-actual-row-index)
4. [Propuesta: Semantic IDs Estáticos](#propuesta-semantic-ids-estáticos)
5. [Manejo de Casos Edge](#manejo-de-casos-edge)
6. [Trade-offs y Limitaciones](#trade-offs-y-limitaciones)
7. [Recomendación Final](#recomendación-final)

---

## 1. RESUMEN EJECUTIVO

### Problema Identificado
El sistema actual de tracking usa **row_index** (posición en archivo) para identificar semáforos, lo cual causa **cross-history transfer** cuando projection boxes se reordenan.

### Solución Propuesta
Adaptar el concepto de **semantic IDs** de Apollo mediante identificadores persistentes almacenados en archivo de texto (columna 5), eliminando la dependencia de infraestructura compleja (HD-Map).

### Impacto
- ✅ **Resuelve**: Cross-history transfer
- ✅ **Mantiene**: Simplicidad del sistema (sin HD-Map)
- ⚠️ **Requiere**: Definición manual de semantic IDs

---

## 2. CONTEXTO: SISTEMA APOLLO ORIGINAL

### 2.1 Arquitectura de Apollo

Apollo utiliza un **HD-Map (High Definition Map)** que contiene información 3D de todos los elementos de la carretera, incluyendo semáforos.

```
Flujo Apollo:
┌─────────────────┐
│   HD-Map Server │ ← Base de datos con semáforos 3D
│   Semáforo 42:  │   (cada semáforo tiene ID único persistente)
│   X=100m, Y=50m │
│   Z=5m          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ GPS + IMU       │ ← Posición del vehículo en tiempo real
│ (lat, lon, θ)   │   (actualizada cada frame)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Proyección 3D→2D│ ← Convierte coordenadas 3D del mapa
│ Semáforo 42:    │   a bbox 2D en imagen usando calibración
│ bbox=(400,150,  │
│       460,220)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Detection + Hun │ ← Detector CNN + Hungarian assignment
│ Asigna det_0    │
│ a semáforo 42   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Tracking        │ ← Usa semantic_id=42 para indexar historia
│ history[42]     │   (SIEMPRE el mismo semáforo)
│ = {color, ...}  │
└─────────────────┘
```

### 2.2 Código Apollo (C++)

**Archivo**: `semantic_decision.cc` líneas 254, 260-261

```cpp
// Obtener semantic ID desde HD-Map
int cur_semantic = light->semantic;  // Ej: ID=42

// Tracking usa semantic ID como key
std::string key = "Semantic_" + std::to_string(cur_semantic);
auto iter = semantic_table_.find(key);  // Buscar historia por ID=42

// Historia persistente:
// Frame 100: semantic_id=42 → bbox=(400,150,460,220), color=GREEN
// Frame 101: semantic_id=42 → bbox=(410,155,470,225), color=GREEN ✅
//            (bbox cambió por movimiento del vehículo, pero ID=42 sigue siendo el mismo semáforo)
```

### 2.3 Ventajas del Sistema Apollo

✅ **Persistencia total**: semantic_id=42 SIEMPRE identifica al mismo semáforo físico
✅ **Dinámico**: Projection boxes se actualizan cada frame según posición del vehículo
✅ **Robusto**: No hay cross-history transfer (historia sigue al semáforo, no a la región espacial)

### 2.4 Limitaciones para Contexto Académico

❌ **Requiere HD-Map server**: Base de datos compleja de toda la ciudad
❌ **Requiere GPS RTK**: Precisión centimeter-level (caro, ~$10,000+ USD)
❌ **Requiere calibración perfecta**: Cámara-GPS-IMU sincronizados
❌ **Infraestructura completa**: No viable para proyecto académico/modular

---

## 3. IMPLEMENTACIÓN ACTUAL (ROW INDEX)

### 3.1 Arquitectura Actual

```
Flujo Actual:
┌─────────────────────┐
│ projection_bboxes   │ ← Archivo de texto estático
│ .txt (4 columnas)   │   (definido manualmente)
│ frame,xmin,ymin,    │
│      xmax,ymax      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Leer archivo        │ ← Lee línea por línea
│ Línea 0 → ROI 0     │   row_index = posición en array (0, 1, 2...)
│ Línea 1 → ROI 1     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Detection + Hun     │ ← Detector CNN + Hungarian assignment
│ Asigna det_0        │
│ a ROI row_index=0   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Tracking            │ ← Usa row_index para indexar historia
│ history[0]          │   (row_index puede CAMBIAR si archivo se reordena)
│ = {color, ...}      │
└─────────────────────┘
```

### 3.2 Código Actual

**Archivo**: `tracking.py` líneas 66-74

```python
for proj_id, det_idx in assignments:  # proj_id = row_index (0, 1, 2...)
    # decidir color actual
    cls = int(max(range(len(recognitions[det_idx])),
                  key=lambda i: recognitions[det_idx][i]))
    color = ["black","red","yellow","green"][cls]

    # obtener o crear estado histórico
    if proj_id not in self.history:  # ← proj_id es row_index
        self.history[proj_id] = SemanticTable(proj_id, frame_ts, color)
```

### 3.3 Problema: Cross-History Transfer

#### Escenario 1: Funcionamiento Normal (SIN reordenamiento)

```
Frame 100:
projection_bboxes.txt:
  Línea 0: 400,150,460,220  ← Semáforo Izquierdo, row_index=0
  Línea 1: 500,150,560,220  ← Semáforo Derecho, row_index=1

Tracking:
  history[0] = {color: GREEN, blink: False}   ✅ Semáforo izquierdo
  history[1] = {color: RED, blink: True}      ✅ Semáforo derecho

Frame 101:
projection_bboxes.txt:
  Línea 0: 400,150,460,220  ← SIGUE siendo izquierdo, row_index=0 ✅
  Línea 1: 500,150,560,220  ← SIGUE siendo derecho, row_index=1 ✅

Tracking:
  history[0] → Semáforo izquierdo ✅ CORRECTO
  history[1] → Semáforo derecho ✅ CORRECTO
```

#### Escenario 2: BUG por Reordenamiento

```
Frame 102:
projection_bboxes.txt (archivo REORDENADO):
  Línea 0: 500,150,560,220  ← Ahora es DERECHO, pero row_index=0 ❌
  Línea 1: 400,150,460,220  ← Ahora es IZQUIERDO, pero row_index=1 ❌

Tracking:
  history[0] → Semáforo DERECHO recibe historia del IZQUIERDO ❌
               (color: GREEN, blink: False) cuando debería ser (RED, True)

  history[1] → Semáforo IZQUIERDO recibe historia del DERECHO ❌
               (color: RED, blink: True) cuando debería ser (GREEN, False)

RESULTADO: ¡Cross-history transfer! Las historias se intercambiaron.
```

### 3.4 Cuándo Ocurre el Problema

1. **Perspective shift del vehículo**: Semáforos cambian de orden espacial (izquierdo→derecho)
2. **Reordenamiento manual del archivo**: Al editar `projection_bboxes.txt`
3. **Generación programática**: Scripts que ordenan ROIs (ej: por coordenada X)

### 3.5 Por Qué NO Pasa en Apollo

Apollo usa **projection boxes dinámicas** que se actualizan cada frame:

```
Frame 100: Vehículo en posición A
  → HD-Map query → Semáforo ID=42 proyecta a (400,150,460,220)
  → Tracking: history[42] = ...

Frame 101: Vehículo se movió a posición B
  → HD-Map query → MISMO semáforo ID=42 proyecta a (410,155,470,225)  ← CAMBIÓ bbox
  → Tracking: SIGUE usando history[42] ✅
  → NO hay cross-history porque el ID=42 es PERSISTENTE
```

---

## 4. PROPUESTA: SEMANTIC IDS ESTÁTICOS

### 4.1 Concepto

Adaptar el sistema de semantic IDs de Apollo mediante **identificadores persistentes almacenados en archivo de texto**, sin necesidad de HD-Map.

### 4.2 Arquitectura Propuesta

```
Flujo Propuesto:
┌─────────────────────┐
│ projection_bboxes   │ ← Archivo de texto con COLUMNA 5 (semantic_id)
│ .txt (5 columnas)   │   (definido manualmente una sola vez)
│ frame,xmin,ymin,    │
│      xmax,ymax,ID   │ ← NUEVA columna
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Leer archivo        │ ← Lee línea por línea
│ Línea 0 → ROI 0,    │   row_index=0, semantic_id=10
│           ID=10     │
│ Línea 1 → ROI 1,    │   row_index=1, semantic_id=20
│           ID=20     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Detection + Hun     │ ← Detector CNN + Hungarian assignment
│ Asigna det_0        │   RETORNA (semantic_id=10, det_idx=0)
│ a ROI ID=10         │   (NO row_index)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Tracking            │ ← Usa semantic_id para indexar historia
│ history[10]         │   (semantic_id NUNCA cambia)
│ = {color, ...}      │
└─────────────────────┘
```

### 4.3 Formato de Archivo

#### ANTES (4 columnas):
```
frame_0000.jpg,400,150,460,220
frame_0000.jpg,500,150,560,220
```

#### DESPUÉS (5 columnas):
```
frame_0000.jpg,400,150,460,220,10
frame_0000.jpg,500,150,560,220,20
```

**Columna 5**: semantic_id (10, 20, 30, ...)
- Valores arbitrarios (pueden ser 10, 20, 30 o 100, 200, 300)
- **REGLA CRÍTICA**: Mismo semáforo físico = mismo ID en TODOS los frames

### 4.4 Cambios de Código Necesarios

#### Cambio 1: `selector.py` (retornar semantic_id)

**Antes:**
```python
# Líneas 62-63
final_assignment1s.append(proj_idx)  # row_index (0, 1, 2...)
final_assignment2s.append(det_idx)

return torch.stack([torch.tensor(final_assignment1s),
                    torch.tensor(final_assignment2s)]).transpose(1, 0)
```

**Después:**
```python
# Líneas 62-63
semantic_id = projections[proj_idx].semantic_id  # ← Leer de projection
final_assignment1s.append(semantic_id)  # ← Ahora retorna semantic_id
final_assignment2s.append(det_idx)

return torch.stack([torch.tensor(final_assignment1s),
                    torch.tensor(final_assignment2s)]).transpose(1, 0)
```

#### Cambio 2: `tracking.py` (usar semantic_id)

**Antes:**
```python
# Línea 66
for proj_id, det_idx in assignments:  # proj_id = row_index
    if proj_id not in self.history:
        self.history[proj_id] = SemanticTable(proj_id, ...)
```

**Después:**
```python
# Línea 66
for semantic_id, det_idx in assignments:  # semantic_id de archivo (10, 20, ...)
    if semantic_id not in self.history:
        self.history[semantic_id] = SemanticTable(semantic_id, ...)
```

**⚠️ NOTA**: `tracking.py` ya está preparado (acepta semantic_id en constructor)

#### Cambio 3: Leer columna 5 del archivo

**Ubicación**: `run_pipeline.py` (o script que lee `projection_bboxes.txt`)

```python
# ANTES
parts = line.strip().split(',')
xmin, ymin, xmax, ymax = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
projection = ProjectionBox(xmin, ymin, xmax, ymax)

# DESPUÉS
parts = line.strip().split(',')
xmin, ymin, xmax, ymax = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
semantic_id = int(parts[5]) if len(parts) > 5 else row_index  # ← Leer columna 5
projection = ProjectionBox(xmin, ymin, xmax, ymax, semantic_id=semantic_id)
```

### 4.5 Ejemplo Completo: Reordenamiento CON Semantic IDs

```
Frame 100:
projection_bboxes.txt:
  Línea 0: 400,150,460,220,10  ← Izquierdo, row_index=0, semantic_id=10
  Línea 1: 500,150,560,220,20  ← Derecho, row_index=1, semantic_id=20

Tracking:
  history[10] = {color: GREEN, blink: False}  ✅ Semáforo ID=10 (izquierdo)
  history[20] = {color: RED, blink: True}     ✅ Semáforo ID=20 (derecho)

Frame 101 (archivo REORDENADO):
projection_bboxes.txt:
  Línea 0: 500,150,560,220,20  ← Derecho, row_index=0, semantic_id=20 ← ID NO cambió
  Línea 1: 400,150,460,220,10  ← Izquierdo, row_index=1, semantic_id=10 ← ID NO cambió

Tracking:
  history[20] → Semáforo DERECHO ✅ CORRECTO (ID=20 siempre es el derecho)
                (color: RED, blink: True)

  history[10] → Semáforo IZQUIERDO ✅ CORRECTO (ID=10 siempre es el izquierdo)
                (color: GREEN, blink: False)

RESULTADO: ✅ NO hay cross-history transfer! Las historias siguen al semáforo correcto.
```

---

## 5. MANEJO DE CASOS EDGE

### 5.1 Caso: MÁS ROIs que Semáforos Físicos

**Escenario**:
```
Semáforos reales: 2 (izquierdo, derecho)

projection_bboxes.txt:
  frame_0000.jpg,400,150,460,220,10  ← Semáforo izquierdo (ID=10)
  frame_0000.jpg,500,150,560,220,20  ← Semáforo derecho (ID=20)
  frame_0000.jpg,600,150,660,220,30  ← ❌ NO HAY SEMÁFORO (false ROI, ID=30)
```

**¿Qué pasa?**

```python
# Hungarian crea matriz 3×2 (3 ROIs, 2 detections):
costs = [
  [0.95, 0.20],  # ROI 0 (ID=10) vs [det_0, det_1]
  [0.25, 0.90],  # ROI 1 (ID=20) vs [det_0, det_1]
  [0.05, 0.10]   # ROI 2 (ID=30) vs [det_0, det_1] ← Scores muy bajos (lejos)
]

# Hungarian maximiza:
assignments = [
  (10, 0),  # semantic_id=10 → detection 0 ✅
  (20, 1),  # semantic_id=20 → detection 1 ✅
  # ROI ID=30 NO tiene assignment (no hay detection válida) ✅
]

# Tracking:
history[10] = ... ✅  # Semáforo ID=10 tiene data
history[20] = ... ✅  # Semáforo ID=20 tiene data
history[30]         # ❌ NO se crea (no hay assignment para ID=30)
```

**✅ Resultado**: Hungarian + validación de ROI **automáticamente descarta** ROIs inválidas.

### 5.2 Caso: MÁS Detections que ROIs (false positives)

**Escenario**:
```
ROIs definidas: 2

Detections encontradas: 3
  - det_0: Cerca de ROI 0 ✅
  - det_1: Cerca de ROI 1 ✅
  - det_2: Lejos de ambas (false positive) ❌
```

**¿Qué pasa?**

```python
costs = [
  [0.94, 0.15, 0.02],  # ROI 0 vs 3 detections
  [0.20, 0.92, 0.05]   # ROI 1 vs 3 detections
]

# Hungarian maximiza (1-to-1):
assignments = [
  (10, 0),  # ROI ID=10 → detection 0 ✅
  (20, 1)   # ROI ID=20 → detection 1 ✅
  # detection_2 queda SIN asignar ✅
]

# En CSVs:
# detection_0 → proj_id=10, status=VALID
# detection_1 → proj_id=20, status=VALID
# detection_2 → proj_id=-1, status=INVALID (no asignada, ignorada)
```

**✅ Resultado**: Detection sin ROI cercana → ID=-1 → ignorada por tracking.

### 5.3 Caso: ROI Fuera de Imagen

**Escenario**:
```
projection_bboxes.txt:
  frame_0000.jpg,400,150,460,220,10   ← Dentro de imagen ✅
  frame_0000.jpg,2000,150,2060,220,20 ← Fuera de imagen (x > 1920) ❌
```

**¿Qué pasa?**

```python
# En selector.py líneas 37-45:
# ROI validation ANTES de Hungarian
if coors[0] > det_box[0] or coors[1] < det_box[2] or ...:
    costs[row, col] = 0.0  # ← Score = 0 para detections fuera de ROI

# Si TODA la fila tiene cost=0 → Hungarian NO asigna nada a esa ROI

# Tracking:
history[10] = ... ✅  # ROI dentro de imagen
history[20]         # ❌ NO se crea (ROI fuera de imagen, sin assignments)
```

**✅ Resultado**: ROI fuera de imagen → sin assignments → no entra al tracking.

---

## 6. TRADE-OFFS Y LIMITACIONES

### 6.1 Comparación: Apollo vs Propuesta

| Aspecto | Apollo Original | Implementación Actual | Propuesta Semantic IDs |
|---------|-----------------|----------------------|------------------------|
| **Fuente de IDs** | HD-Map database | Row index (0, 1, 2...) | Archivo columna 5 (10, 20, ...) |
| **Persistencia** | ✅ Absoluta (GPS + HD-Map) | ❌ Ninguna (depende orden) | ✅ Manual (definida por usuario) |
| **Dinámico** | ✅ SÍ (actualiza cada frame) | ❌ NO (estático) | ❌ NO (estático) |
| **Cross-history** | ❌ NO ocurre | ✅ SÍ ocurre | ❌ NO ocurre |
| **Infraestructura** | ❌ Compleja (HD-Map, GPS RTK) | ✅ Simple (archivo .txt) | ✅ Simple (archivo .txt) |
| **Setup inicial** | ❌ Requiere mapeo de ciudad | ✅ Manual (1 frame base) | ✅ Manual (1 frame base) |
| **Mantenimiento** | ✅ Automático (GPS actualiza) | ⚠️ Manual (propagar boxes) | ⚠️ Manual (propagar boxes) |

### 6.2 Limitaciones de la Propuesta

#### Limitación 1: Projection Boxes Estáticas

**Problema**:
```
Frame 100: Vehículo en posición A
  → projection_bboxes.txt define ROI ID=10 en (400,150,460,220)
  → Semáforo físico ESTÁ ahí ✅

Frame 200: Vehículo se movió mucho (nueva posición B)
  → projection_bboxes.txt SIGUE definiendo ROI ID=10 en (400,150,460,220) ❌
  → Semáforo físico AHORA está en (600,200,660,280) ← DESINCRONIZADO
  → Hungarian (70% peso en distancia):
      - ROI ID=10 @ (400,150) vs Sem físico @ (600,200) → distancia=200px
      - Score bajo → puede NO asignarse ❌
```

**Solución parcial**: Propagación manual o semi-automática de projection boxes frame a frame.

**Diferencia con Apollo**: Apollo actualiza projection boxes AUTOMÁTICAMENTE cada frame usando GPS + HD-Map.

#### Limitación 2: 70% Peso en Distancia Espacial

**Problema inherente** (APOLLO TAMBIÉN LO TIENE):
```
# Hungarian usa 70% distance, 30% confidence
costs[row, col] = 0.3 * detection_score + 0.7 * gaussian_distance

Escenario problemático:
  ROI ID=10 @ (400, 150) ← Desincronizada (semáforo se movió)

  Detection A: score=0.95, posición=(600, 200) → distance=200px → gaussian≈0.1
    → cost = 0.3*0.95 + 0.7*0.1 = 0.285 + 0.07 = 0.355

  Detection B: score=0.60, posición=(405, 155) → distance=5px → gaussian≈0.95
    → cost = 0.3*0.60 + 0.7*0.95 = 0.18 + 0.665 = 0.845

  Hungarian elige Detection B (menor confianza pero más cerca) ❌
```

**Conclusión**: Si projection boxes se dessincronizan, semantic IDs NO resuelven el problema de distancia espacial.

**Diferencia con Apollo**: Apollo mantiene projection boxes sincronizadas con GPS → problema no ocurre.

### 6.3 Cuándo Semantic IDs SÍ Resuelven el Problema

✅ **Reordenamiento de archivo**: Si projection boxes se reordenan pero ESTÁN en posiciones correctas
✅ **Perspective shifts leves**: Semáforos cambian de orden espacial (izq↔der) pero projection boxes siguen siendo precisas
✅ **Generación programática**: Scripts que ordenan ROIs alfabéticamente/por coordenada

### 6.4 Cuándo Semantic IDs NO Resuelven el Problema

❌ **Projection boxes muy desincronizadas**: Semáforo real está >100px lejos de ROI definida
❌ **Movimiento significativo del vehículo**: Perspectiva cambia radicalmente
❌ **GPS drift sin HD-Map**: Sin sistema de actualización automática de ROIs

**Solución completa**: Requiere projection boxes dinámicas (HD-Map + GPS) como Apollo original.

---

## 7. RECOMENDACIÓN FINAL

### 7.1 Enfoque Sugerido: Sistema Dual para Tesis

#### FASE 1: Sistema con Row Index (ACTUAL)
**Propósito**: Demostrar empíricamente el problema de cross-history transfer

```
Tests a ejecutar:
- right/problematic: Archivo con projection boxes reordenadas
- left/problematic: Archivo con projection boxes reordenadas
- Resultado esperado: ❌ Cross-history transfer visible en CSVs
```

**Output**:
- CSVs mostrando historias transferidas incorrectamente
- Documentación del problema identificado

#### FASE 2: Sistema con Semantic IDs (PROPUESTA)
**Propósito**: Demostrar que semantic IDs resuelven el problema

```
Tests a ejecutar:
- MISMOS tests (right/problematic, left/problematic)
- MISMA configuración de projection boxes reordenadas
- ÚNICA diferencia: Usar semantic IDs (columna 5) para tracking
- Resultado esperado: ✅ NO hay cross-history transfer
```

**Output**:
- CSVs mostrando historias correctamente asignadas
- Comparación lado a lado: Fase 1 vs Fase 2

### 7.2 Estructura de Carpetas Propuesta

```
TrafficLightDetection/
├── test_doble_chico/                    # FASE 1: row_index
│   ├── run_pipeline.py                  # Sin modificar
│   ├── projection_bboxes_master.txt     # Sin columna 5 (o ignorada)
│   └── outputs/                         # Resultados con BUG
│
├── test_doble_chico_semantic/           # FASE 2: semantic_id
│   ├── run_pipeline.py                  # Modificado (usar columna 5)
│   ├── projection_bboxes_master.txt     # CON columna 5
│   └── outputs/                         # Resultados SIN BUG
│
└── docs/
    ├── COMPARACION_ROW_VS_SEMANTIC.md   # Análisis comparativo
    └── PROPUESTA_SEMANTIC_IDS.md        # Este documento
```

### 7.3 Contribución para la Tesis

**NO es**: "Inventé semantic IDs" (Apollo ya los usa)

**SÍ es**:
1. **Demostración empírica del problema row_index**: Test controlado que aísla cross-history transfer
2. **Adaptación de Apollo a contexto simplificado**: Semantic IDs desde archivo estático (sin HD-Map)
3. **Validación experimental**: Comparación cuantitativa Fase 1 vs Fase 2
4. **Análisis de limitaciones**: Documentación de trade-offs (projection boxes estáticas vs dinámicas)

### 7.4 Narrativa Sugerida para la Tesis

> **Capítulo 4: Identificación y Resolución de Cross-History Transfer**
>
> Apollo utiliza semantic IDs persistentes provenientes del HD-Map para garantizar que la historia de tracking sigue al semáforo físico correcto, independientemente de cambios en la perspectiva o posición del vehículo. Sin embargo, este enfoque requiere infraestructura compleja (HD-Map server, GPS RTK, calibración perfecta) no viable en contextos académicos.
>
> En este trabajo, se identificó empíricamente el problema de cross-history transfer al usar row_index como identificador de semáforos (Sistema Fase 1). Mediante tests controlados que reordenan projection boxes, se demostró que las historias de tracking se transfieren incorrectamente entre semáforos físicos distintos.
>
> Se propuso una adaptación del concepto de semantic IDs mediante identificadores persistentes almacenados en archivo de texto estático (columna 5), eliminando la dependencia de HD-Map dinámico. Los mismos tests aplicados al Sistema Fase 2 (con semantic IDs) demostraron la eliminación completa del cross-history transfer.
>
> Se documentaron las limitaciones de este enfoque simplificado (projection boxes estáticas, vulnerabilidad a desincronización espacial) y se identificó como trabajo futuro la integración con sistemas de localización para actualización dinámica de ROIs.

### 7.5 Decisión a Tomar con Profesores

**Opción A: Implementar ambas fases**
- ✅ Demuestra problema + solución
- ✅ Contribución clara y validada
- ⚠️ Requiere tiempo de implementación (~2-4 horas)

**Opción B: Solo documentar el problema (Fase 1)**
- ✅ Más rápido (no requiere código nuevo)
- ✅ Identifica gap con Apollo
- ❌ No demuestra solución

**Opción C: Solo implementar Fase 2 (semantic IDs)**
- ✅ Sistema final más robusto
- ❌ No hay baseline para comparar
- ❌ Menor impacto académico (sin demostración empírica del problema)

---

## ANEXO: Referencias

### Código Apollo Verificado
- `semantic_decision.cc` líneas 254, 260-261 (semantic ID usage)
- `select.cc` líneas 95-120 (Hungarian assignment post-processing)
- `detection.cc` líneas 351-354 (`is_detected` flag handling)

### Documentos del Proyecto
- `VERIFICACION_EXHAUSTIVA_CODIGO.md`: Análisis línea por línea de fidelidad con Apollo
- `ANALISIS_FLUJO_APOLLO_COMPLETO.md`: Análisis exhaustivo de 1,187 líneas de C++
- `CAMBIOS_PENDIENTES_2025-11-04.md`: Lista de cambios identificados

### Papers Relacionados
- Baidu Apollo Auto-Calibration System (arXiv:1808.10134)
- California DMV Autonomous Vehicle Disengagement Reports (2018-2023)

---

**FIN DEL DOCUMENTO**

**Próximos pasos**:
1. Revisar con profesores
2. Decidir entre Opción A/B/C
3. Ejecutar implementación según decisión
4. Documentar resultados para tesis
