# Flujo del Sistema Propio de Detección de Semáforos

**Documento:** Explicación narrativa del pipeline completo de detección y reconocimiento de semáforos
**Sistema:** Implementación propia basada en PyTorch
**Última actualización:** 2025-01-18

---

## Introducción

Este documento describe el flujo completo del sistema de detección de semáforos implementado en Python/PyTorch. A diferencia del sistema original de Apollo que usa HD-Map y proyección 3D→2D dinámica, **nuestro sistema utiliza projection boxes definidas manualmente** que representan las regiones de interés donde se espera encontrar semáforos.

### Diferencias Clave con Apollo Original

| Característica | Apollo Original | Nuestro Sistema |
|----------------|-----------------|-----------------|
| **Entrada de ROIs** | HD-Map con posiciones 3D georreferenciadas | Projection boxes manualmente definidas `[x1,y1,x2,y2,signal_id]` |
| **Proyección 3D→2D** | Sí (usa pose del vehículo + calibración) | No (boxes ya están en coordenadas 2D) |
| **Semantic IDs** | Diseñado pero no implementado (siempre 0) | No aplica |
| **Selección de Cámara** | Sí (telephoto vs wide-angle) | No (single camera) |
| **Tracking Temporal** | Por signal ID individual | Por signal ID individual (similar) |
| **Query HD-Map** | Dinámico (consulta continua según pose) | Estático (boxes cargadas desde archivo) |

---

## Conceptos Clave

### Projection Boxes y Signal ID

En nuestro sistema, una **projection box** es una caja delimitadora `[x1, y1, x2, y2, signal_id]` que define:
- `x1, y1, x2, y2`: Coordenadas del rectángulo en la imagen (píxeles)
- `signal_id`: Identificador del semáforo físico (persiste entre frames)

El `signal_id` es crítico para el tracking: el historial de estados se asocia al semáforo físico, no a la posición en el array.

**Ejemplo:**
```python
boxes = [
    [850, 300, 890, 380, 0],   # signal_id=0 → "signal_0"
    [1050, 280, 1090, 360, 1], # signal_id=1 → "signal_1"
    [1200, 320, 1235, 390, 2], # signal_id=2 → "signal_2"
]
```

### ProjectionROI Object

Internamente, cada box se convierte a un objeto `ProjectionROI` con dos tipos de ID:

```python
class ProjectionROI:
    def __init__(self, x, y, w, h, proj_id=None, signal_id=None):
        # Geometría
        self.x, self.y, self.w, self.h = x, y, w, h
        self.center_x = int((x + x + w) / 2)
        self.center_y = int((y + y + h) / 2)

        # IDs para tracking
        self.proj_id = proj_id      # ID temporal en este frame (auto-generado)
        self.signal_id = signal_id  # ID del semáforo físico (persistente)
```

- **`proj_id`**: Identificador temporal de la projection box en el frame actual (ej: `"proj_0"`, `"proj_1"`)
- **`signal_id`**: Identificador del semáforo físico que persiste entre frames (ej: `"signal_0"`, `"signal_1"`)

El tracking usa `signal_id` para mantener el historial asociado al semáforo físico.

---

## Estructura de Datos: Tensores en Cada Etapa

### 1. **Input: Projection Boxes** (Python list)
```python
boxes = [
    [850, 300, 890, 380, 0],   # [x1, y1, x2, y2, signal_id]
    [1050, 280, 1090, 360, 1],
    [1200, 320, 1235, 390, 2]
]
# M boxes (M = 3 en este ejemplo)
```

### 2. **Detections** (Tensor N×9)
```python
detections = torch.Tensor([
    # [score, x1, y1, x2, y2, bg_prob, vert_prob, quad_prob, hori_prob]
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],
])
# N detections después de NMS y filtros
```

### 3. **Recognitions** (Tensor N×4)
```python
recognitions = torch.Tensor([
    # [black_prob, red_prob, yellow_prob, green_prob] - ONE-HOT
    [0.0, 1.0, 0.0, 0.0],  # Detection #0 → RED
    [0.0, 0.0, 0.0, 1.0],  # Detection #1 → GREEN
])
```

### 4. **Assignments** (Tensor K×2)
```python
assignments = torch.Tensor([
    [0, 0],  # proj_idx=0 → det_idx=0
    [1, 1],  # proj_idx=1 → det_idx=1
])
# K assignments (K ≤ min(M, N))
```

### 5. **Revised States** (dict, si tracking habilitado)
```python
revised = {
    "signal_0": ('red', False),    # signal_id → (color, blink)
    "signal_1": ('green', False),
}
# Dict con signal_id (string) como keys
```

---

## Arquitectura del Pipeline

El punto de entrada principal es `load_pipeline()` en `src/tlr/pipeline.py`:

```python
from tlr.pipeline import load_pipeline

# Carga del pipeline (incluye tracker por defecto)
pipeline = load_pipeline('cuda:0')  # o 'cpu'

# Uso
valid_detections, recognitions, assignments, invalid_detections, revised = \
    pipeline(image, boxes, frame_ts=timestamp)
```

**Componentes cargados:**
1. **Detector** (`TLDetector`): Red SSD-style para detectar semáforos
2. **Classifiers** (3 `Recognizer`s): Clasificadores por orientación (vertical, quad, horizontal)
3. **Hungarian Optimizer** (`HungarianOptimizer`): Para asignación óptima projection→detection
4. **Tracker** (`TrafficLightTracker`): Para revisión temporal del estado (habilitado por defecto)

---

## ETAPA 1: PREPROCESAMIENTO POR PROJECTION

**Archivo:** `src/tlr/tools/utils.py` (funciones `crop`, `preprocess4det`, `boxes2projections`)
**Entrada:** Imagen completa + una projection box
**Salida:** Crop normalizado de 270×270 píxeles

### ¿Qué hace?

Por cada projection box `[x1, y1, x2, y2, signal_id]`:

1. **Convierte a ProjectionROI** (`box2projection` en `utils.py:308-333`):
   ```python
   # Extrae signal_id del 5to elemento
   signal_id = f"signal_{int(box[4])}" if len(box) > 4 else None

   # Crea ProjectionROI con geometría y IDs
   projection = ProjectionROI(x, y, w, h, proj_id=None, signal_id=signal_id)
   ```

2. **Calcula ROI expandido** (`crop()` en `utils.py:219-240`):
   ```python
   crop_scale = 2.5      # Factor de expansión (igual que Apollo)
   min_crop_size = 270   # Tamaño mínimo

   # Tamaño del crop (siempre cuadrado)
   resize = crop_scale * max(projection.w, projection.h)
   resize = max(resize, min_crop_size)
   resize = min(resize, width, height)  # No exceder imagen

   # Centrar en la projection
   xl = projection.center_x - resize/2 + 1
   yt = projection.center_y - resize/2 + 1
   # + ajustes si excede bordes de imagen
   ```

3. **Extrae y resize** (`preprocess4det` en `utils.py:242-247`):
   ```python
   xl, xr, yt, yb = crop(image.shape, projection)
   src = image[yt:yb, xl:xr]
   dst = torch.zeros(270, 270, 3, device=src.device)
   resized = ResizeGPU(src, dst, means)  # Interpolación bilineal GPU
   ```

4. **Normalización:**
   - Resta medias del detector: `[102.98, 115.95, 122.77]` (BGR)

### Cardinalidad
```
M projection boxes → M crops de 270×270
```

---

## ETAPA 2: DETECCIÓN CNN

**Archivo:** `src/tlr/pipeline.py:26-76` (método `detect`)
**Entrada:** Imagen completa + M projection boxes
**Salida:** Tensor N×9 con detecciones (después de NMS y filtros)

### ¿Qué hace?

1. **Preprocesa cada projection:**
   ```python
   projections = boxes2projections(boxes)
   for projection in projections:
       input = preprocess4det(image, projection, self.means_det)
       bboxes = self.detector(input.unsqueeze(0).permute(0, 3, 1, 2))
       detected_boxes.append(bboxes)
   ```

2. **Restaura coordenadas a imagen completa** (`restore_boxes_to_full_image`):
   - Escala coordenadas de 270×270 al tamaño real del crop
   - Suma offsets del crop para obtener coordenadas globales

3. **Sort por score + NMS:**
   ```python
   # Ordenar por score de clasificación (max de [bg, vert, quad, hori])
   scores = torch.max(detections[:, 5:9], dim=1).values
   sorted_indices = torch.argsort(scores, descending=True)
   detections_sorted = detections[sorted_indices]

   # NMS con threshold 0.6 (igual que Apollo)
   idxs = nms(detections_sorted[:, 1:5], 0.6)
   detections = detections_sorted[idxs]
   ```

4. **Filtro de tamaño y aspect ratio** (Apollo-style):
   ```python
   MIN_SIZE = 5      # Mínimo 5 píxeles
   MAX_SIZE = 300    # Máximo 300 píxeles
   MIN_ASPECT = 0.5  # Aspect ratio mínimo (h/w)
   MAX_ASPECT = 8.0  # Aspect ratio máximo

   for det in detections:
       w = det[3] - det[1]
       h = det[4] - det[2]
       aspect = h / w

       # Rechazar si tamaño inválido o aspect ratio extremo
       if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
           valid = False
       if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
           valid = False
   ```

### Arquitectura del Detector (SSD-style)

```
Input [1,3,270,270]
  ↓
FeatureNet (VGG-like backbone)
  ├→ rpn_cls_prob_reshape [1,30,34,34]
  ├→ rpn_bbox_pred [1,60,34,34]
  └→ ft_add_left_right [1,490,34,34]
  ↓
RPNProposalSSD (~52 proposals)
  → rois [52,5]
  ↓
DFMBPSROIAlign
  → psroi_rois [52,10,7,7]
  ↓
Flatten + FC(2048) + ReLU
  ↓
┌──────────┬──────────┐
│ cls_score│ bbox_pred│
│ FC(4)    │ FC(16)   │
└──────────┴──────────┘
  ↓
RCNNProposal (NMS interno)
  → bboxes [N, 9]
```

**Output por detection:**
```
[score, x1, y1, x2, y2, bg_prob, vert_prob, quad_prob, hori_prob]
```

### Cardinalidad
```
M projections → K detecciones brutas → N detecciones después de NMS y filtros
```

---

## ETAPA 3: FILTRADO POR TIPO Y ASIGNACIÓN

**Archivo:** `src/tlr/pipeline.py:158-171` + `src/tlr/selector.py`
**Entrada:** Detecciones N×9 + projection boxes originales
**Salida:** Valid detections + Assignments (K×2)

### ¿Qué hace?

1. **Filtro de confidence mínima:**
   ```python
   MIN_CONFIDENCE = 0.3
   confidence_scores = torch.max(detections[:, 5:9], dim=1).values
   confidence_mask = confidence_scores >= MIN_CONFIDENCE
   detections = detections[confidence_mask]
   ```

2. **Clasifica tipo de semáforo:**
   ```python
   tl_types = torch.argmax(detections[:, 5:], dim=1)
   # 0=background, 1=vertical, 2=quadrant, 3=horizontal
   ```

3. **Filtra detecciones válidas:**
   ```python
   valid_mask = tl_types != 0  # Elimina background
   valid_detections = detections[valid_mask]
   invalid_detections = detections[~valid_mask]
   ```

4. **Asignación Húngara** (`select_tls` en `selector.py:8-68`):

   **Matriz de costos:**
   ```python
   for row, projection in enumerate(projections):
       center_hd = [projection.center_x, projection.center_y]
       coors = crop(item_shape, projection)  # ROI expandido

       for col, detection in enumerate(detections):
           # Score de distancia (Gaussiana 2D, sigma=100)
           center_det = [(det[3]+det[1])/2, (det[4]+det[2])/2]
           distance_score = exp(-0.5 * ((dx/100)² + (dy/100)²))

           # Score de confianza (clipped a 0.9)
           detection_score = min(max(det[5:]), 0.9)

           # Score combinado: 70% distancia, 30% confianza
           costs[row, col] = 0.3 * detection_score + 0.7 * distance_score

           # Validación: detection debe estar dentro del crop ROI
           if detection_outside_crop_roi:
               costs[row, col] = 0.0
   ```

   **Hungarian Algorithm:**
   ```python
   assignments = hungarian_optimizer.maximize(costs)
   # Resultado: lista de pares [proj_idx, det_idx]
   ```

### Cardinalidad
```
M projections + N valid_detections → K assignments (K ≤ min(M, N))
```

**Nota importante:** `proj_idx` es el índice en la lista `projections`, NO el `signal_id`. El `signal_id` se obtiene de `projections[proj_idx].signal_id`.

---

## ETAPA 4: RECONOCIMIENTO DE COLOR

**Archivo:** `src/tlr/pipeline.py:78-120` (método `recognize`)
**Entrada:** Imagen completa + valid_detections + tipos de semáforo
**Salida:** Tensor N×4 con clasificaciones one-hot

### ¿Qué hace?

Por cada detección válida:

1. **Selecciona clasificador según tipo:**
   ```python
   tl_type = tl_types[i]  # 1=vert, 2=quad, 3=hori
   recognizer, shape = self.classifiers[tl_type - 1]
   # classifiers = [
   #   (vert_recognizer, (96, 32, 3)),   # tipo 1
   #   (quad_recognizer, (64, 64, 3)),   # tipo 2
   #   (hori_recognizer, (32, 96, 3))    # tipo 3
   # ]
   ```

2. **Preprocesa ROI de la detección:**
   ```python
   det_box = detection[1:5].type(torch.long)  # [x1, y1, x2, y2]
   input = preprocess4rec(img, det_box, shape, self.means_rec)
   # means_rec = [66.56, 66.58, 69.06] (BGR order)
   ```

3. **Escala y ejecuta inferencia:**
   ```python
   input_scaled = input.permute(2, 0, 1).unsqueeze(0)  # HWC → NCHW
   input_scaled = input_scaled * 0.01  # Apollo's scale factor
   output_probs = recognizer(input_scaled)[0]  # [4]
   ```

4. **Apollo's Prob2Color logic:**
   ```python
   max_prob, max_idx = torch.max(output_probs, dim=0)
   threshold = 0.5  # Apollo's classify_threshold

   if max_prob > threshold:
       color_id = max_idx.item()  # 0=BLACK, 1=RED, 2=YELLOW, 3=GREEN
   else:
       color_id = 0  # Forzar a BLACK (baja confianza)

   # Crear resultado one-hot
   result = torch.zeros(4)
   result[color_id] = 1.0
   ```

### Arquitectura del Recognizer

```
Input [1,3,H,W]  (H,W según orientación)
  ↓
Conv1(3→32) + BN + Scale + ReLU + MaxPool
  ↓
Conv2(32→64) + BN + Scale + ReLU + MaxPool
  ↓
Conv3(64→128) + BN + Scale + ReLU + MaxPool
  ↓
Conv4(128→128) + BN + Scale + ReLU + MaxPool
  ↓
Conv5(128→128) + BN + Scale + ReLU + AvgPool(orientación-específico)
  ↓
FC(128) + BN + Scale + ReLU
  ↓
FC(4) → Softmax
  → [black_prob, red_prob, yellow_prob, green_prob]
```

### Cardinalidad
```
N valid_detections → N recognitions (one-to-one)
```

---

## ETAPA 5: TRACKING TEMPORAL

**Archivo:** `src/tlr/tracking.py` + `src/tlr/pipeline.py:180-192`
**Entrada:** frame_ts, assignments, recognitions, projections
**Salida:** dict {signal_id → (revised_color, blink)}

### Concepto Clave: Tracking por Signal ID

El tracking mantiene el historial **por signal_id** (el semáforo físico), NO por índice de projection. Esto significa que si un semáforo cambia de posición en el array de boxes, su historial se mantiene.

```python
# El tracker recibe projections para acceder al signal_id
revised = self.tracker.track(frame_ts, assigns_list, recs_list, projections)

# Internamente:
for proj_idx, det_idx in assignments:
    signal_id = projections[proj_idx].signal_id  # ej: "signal_0"
    # El historial se guarda en self.history[signal_id]
```

### Parámetros de Configuración

```python
REVISE_TIME_S = 1.5              # Ventana temporal de revisión
BLINK_THRESHOLD_S = 0.55         # Umbral para detectar blink
HYSTERETIC_THRESHOLD_COUNT = 1   # Frames para confirmar cambio desde BLACK
```

### Lógica de Revisión Temporal (SemanticDecision.update)

**Por cada assignment `(proj_idx, det_idx)`:**

1. **Obtener signal_id y color actual:**
   ```python
   signal_id = projections[proj_idx].signal_id  # ej: "signal_0"
   cls = argmax(recognitions[det_idx])
   cur_color = ["black", "red", "yellow", "green"][cls]
   ```

2. **Obtener o crear historial:**
   ```python
   if signal_id not in self.history:
       self.history[signal_id] = SemanticTable(signal_id, frame_ts, cur_color)
       # Nuevo semáforo → aceptar color inmediatamente
       return
   st = self.history[signal_id]
   ```

3. **Calcular tiempo transcurrido:**
   ```python
   dt = frame_ts - st.time_stamp
   ```

4. **Aplicar reglas según ventana temporal:**

   **Si `dt <= 1.5s` (dentro de ventana):**

   ```python
   if cur_color == "yellow":
       if st.color == "red":
           # REGLA DE SEGURIDAD: Yellow después de Red → mantener Red
           # "Any yellow after red is reset to red for safety until green displays"
           # NO cambiar st.color
           pass
       else:
           # Yellow después de Green/Black → aceptar
           st.color = cur_color
           st.last_dark_time = frame_ts

   elif cur_color in ("red", "green"):
       st.color = cur_color

       # DETECCIÓN DE BLINK
       if (frame_ts - st.last_bright_time > BLINK_THRESHOLD_S and
           st.last_dark_time > st.last_bright_time):
           st.blink = True
       else:
           st.blink = False

       st.last_bright_time = frame_ts

   elif cur_color == "black":
       st.last_dark_time = frame_ts
       if st.color in ("unknown", "black"):
           st.color = cur_color
       # Si estaba encendido → mantener color anterior
   ```

   **Si `dt > 1.5s` (ventana expirada):**
   ```python
   # Resetear historial y aceptar color actual sin validación
   st.time_stamp = frame_ts
   st.color = cur_color
   st.blink = False
   ```

### Estructura SemanticTable

```python
class SemanticTable:
    semantic_id: str        # signal_id del semáforo
    time_stamp: float       # Último timestamp procesado
    color: str              # Color actual ("black", "red", "yellow", "green")
    last_bright_time: float # Último timestamp con color bright (red/green)
    last_dark_time: float   # Último timestamp con color dark (yellow/black)
    blink: bool             # Flag de parpadeo detectado
    hysteretic_color: str   # Color candidato para histéresis
    hysteretic_count: int   # Contador de histéresis
```

### Cardinalidad
```
K assignments → K revised states (por signal_id)
```

### Ejemplo de Tracking

**Frame t=10.25:**
```python
# Historial previo (t=10.15):
self.history = {
    "signal_0": SemanticTable(color="red", last_bright_time=10.15),
    "signal_1": SemanticTable(color="green", last_bright_time=10.15),
}

# Reconocimientos actuales:
# signal_0 → RED, signal_1 → GREEN

# Resultado:
revised = {
    "signal_0": ('red', False),
    "signal_1": ('green', False),
}
```

**Frame t=10.40 (Yellow después de Green):**
```python
# signal_1 detecta YELLOW
dt = 10.40 - 10.25 = 0.15s (dentro de ventana)
st.color era "green" → Yellow después de Green es VÁLIDO
st.color = "yellow"

revised = {
    "signal_1": ('yellow', False),
}
```

**Frame t=10.45 (Yellow después de Red - INVÁLIDO):**
```python
# signal_0 detecta YELLOW (pero estaba en RED)
dt = 10.45 - 10.25 = 0.20s (dentro de ventana)
st.color era "red" → Yellow después de Red es INVÁLIDO
# Mantener st.color = "red" (regla de seguridad)

revised = {
    "signal_0": ('red', False),  # Mantiene RED
}
```

---

## ETAPA 6: OUTPUT FINAL

**Archivo:** `src/tlr/pipeline.py:122-193` (método `forward`)

### Estructura de Datos de Salida

```python
valid_detections, recognitions, assignments, invalid_detections, revised = \
    pipeline(image, boxes, frame_ts=10.25)
```

**1. `valid_detections` (Tensor N×9):**
```python
torch.Tensor([
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],
])
# [score, x1, y1, x2, y2, bg, vert, quad, hori]
```

**2. `recognitions` (Tensor N×4):**
```python
torch.Tensor([
    [0.0, 1.0, 0.0, 0.0],  # RED (one-hot)
    [0.0, 0.0, 0.0, 1.0],  # GREEN (one-hot)
])
```

**3. `assignments` (Tensor K×2):**
```python
torch.Tensor([
    [0, 0],  # proj_idx=0 → det_idx=0
    [1, 1],  # proj_idx=1 → det_idx=1
])
```

**4. `invalid_detections` (Tensor L×9):**
```python
torch.Tensor([
    [0.60, 1205, 328, 1230, 382, 0.85, 0.10, 0.03, 0.02],
])
# Detecciones clasificadas como background
```

**5. `revised` (dict):**
```python
{
    "signal_0": ('red', False),
    "signal_1": ('green', False),
}
# signal_id (string) → (color, blink)
```

### Uso Típico del Output

```python
# Dibujar bounding boxes con color revisado
for (proj_idx, det_idx) in assignments:
    det = valid_detections[det_idx]
    box = [det[1], det[2], det[3], det[4]]

    # Obtener signal_id de la projection
    signal_id = projections[proj_idx].signal_id  # ej: "signal_0"

    if revised is not None and signal_id in revised:
        color, blink = revised[signal_id]
    else:
        # Sin tracking, usar recognition directo
        rec = recognitions[det_idx]
        color_idx = torch.argmax(rec).item()
        color = ["black", "red", "yellow", "green"][color_idx]
        blink = False

    draw_box(image, box, color, blink)
```

---

## RESUMEN DE CARDINALIDADES

| Etapa | Input | Output | Relación |
|-------|-------|--------|----------|
| **1. Preproc** | M projection boxes | M crops 270×270 | 1:1 |
| **2. Detección** | M crops | N detections (después filtros) | M:N |
| **3. Filtrado** | N detections | N' valid + invalid | Split |
| **4. Asignación** | M projections + N' valid | K assignments | M:N' → K |
| **5. Reconocimiento** | N' valid detections | N' recognitions | 1:1 |
| **6. Tracking** | K assignments | K revised states | 1:1 |

**Ejemplo numérico:**
```
M = 3 projection boxes
↓ Detección + NMS + Filtros de tamaño
N = 3 detections
↓ Filtro de confidence (≥0.3) + Filtrado tipo
N' = 2 valid detections
↓ Asignación Húngara
K = 2 assignments
↓ Reconocimiento
N' = 2 recognitions (one-hot)
↓ Tracking
K = 2 revised states (por signal_id)
```

---

## DIAGRAMA DE BLOQUES

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                        │
│ • Image (Tensor [H,W,3] BGR)                                │
│ • Projection Boxes (List [[x1,y1,x2,y2,signal_id], ...])   │
│ • Frame Timestamp (float, para tracking)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 1: PREPROCESAMIENTO                                   │
│ • boxes2projections() → ProjectionROI con signal_id         │
│ • Por cada projection: crop + expand 2.5× + resize 270×270  │
│ • Normalización: img - [102.98, 115.95, 122.77]            │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 2: DETECCIÓN CNN                                      │
│ • TLDetector (SSD + RPN + DFMB)                             │
│ • Restaurar coords: scale → offset                          │
│ • Sort by score + NMS (IoU=0.6)                             │
│ • Filtros: tamaño (5-300px), aspect ratio (0.5-8.0)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 3: FILTRADO + ASIGNACIÓN                              │
│ • Filtro confidence (≥0.3)                                  │
│ • Clasificar tipo: argmax → 0/1/2/3                         │
│ • Filtrar válidos: tipo != 0 (background)                   │
│ • Hungarian: proj → det (70% dist + 30% conf)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 4: RECONOCIMIENTO                                     │
│ • Crop ROI + resize según tipo (96×32 / 64×64 / 32×96)     │
│ • Normalización: img - [66.56, 66.58, 69.06] (BGR)         │
│ • Scale: × 0.01                                             │
│ • Recognizer → softmax → Prob2Color (threshold=0.5)         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 5: TRACKING TEMPORAL                                  │
│ • Historial por signal_id (no por índice)                   │
│ • Ventana temporal: 1.5s                                    │
│ • Regla: Yellow después de Red → mantener Red               │
│ • Blink detection: bright→dark→bright < 0.55s              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT                                                       │
│ • valid_detections: [N',9]                                  │
│ • recognitions: [N',4] (one-hot)                            │
│ • assignments: [K,2]                                        │
│ • invalid_detections: [L,9]                                 │
│ • revised: {signal_id: (color, blink)}                      │
└─────────────────────────────────────────────────────────────┘
```

---

## PARÁMETROS CLAVE DEL SISTEMA

| Parámetro | Valor | Ubicación |
|-----------|-------|-----------|
| Factor de expansión ROI | 2.5× | `utils.py:222` |
| Tamaño mínimo de crop | 270px | `utils.py:223` |
| Medias detector (BGR) | [102.98, 115.95, 122.77] | `pipeline.py:200` |
| Medias recognizer (BGR) | [66.56, 66.58, 69.06] | `pipeline.py:203` |
| Scale factor recognizer | 0.01 | `pipeline.py:94` |
| Threshold NMS | 0.6 IoU | `pipeline.py:46` |
| Threshold Prob2Color | 0.5 | `pipeline.py:101` |
| Min confidence | 0.3 | `pipeline.py:151` |
| Min detection size | 5px | `pipeline.py:53` |
| Max detection size | 300px | `pipeline.py:54` |
| Min aspect ratio | 0.5 | `pipeline.py:55` |
| Max aspect ratio | 8.0 | `pipeline.py:56` |
| Peso distancia (asignación) | 0.7 | `selector.py:33` |
| Peso confianza (asignación) | 0.3 | `selector.py:34` |
| Gaussian sigma | 100.0 | `selector.py:25` |
| Ventana temporal tracking | 1.5s | `tracking.py:12` |
| Threshold blink | 0.55s | `tracking.py:16` |
| Hysteretic threshold | 1 | `tracking.py:21` |

---

## ARCHIVOS DE CONFIGURACIÓN Y PESOS

### Pesos de Modelos
```
src/tlr/weights/
├── tl.torch        # Detector SSD (270×270 → detections)
├── vert.torch      # Recognizer vertical (96×32 → 4 clases)
├── quad.torch      # Recognizer quad (64×64 → 4 clases)
└── hori.torch      # Recognizer horizontal (32×96 → 4 clases)
```

### Configuraciones JSON
```
src/tlr/confs/
├── bbox_reg_param.json
├── detection_output_ssd_param.json
├── dfmb_psroi_pooling_param.json
├── rcnn_bbox_reg_param.json
└── rcnn_detection_output_ssd_param.json
```

---

## EJEMPLO COMPLETO DE EJECUCIÓN

```python
import torch
from tlr.pipeline import load_pipeline

# Cargar pipeline
pipeline = load_pipeline('cuda:0')

# Imagen (1920×1080 BGR)
image = torch.Tensor(...)  # [1080, 1920, 3]

# Projection boxes con signal_id
boxes = [
    [850, 300, 890, 380, 0],    # signal_id=0
    [1050, 280, 1090, 360, 1],  # signal_id=1
    [1200, 320, 1235, 390, 2],  # signal_id=2
]

# Timestamp para tracking
frame_ts = 10.25

# Ejecutar
valid_dets, recs, assigns, invalid_dets, revised = \
    pipeline(image, boxes, frame_ts=frame_ts)

# Resultado
print(revised)
# {
#     "signal_0": ('red', False),
#     "signal_1": ('green', False),
# }
```

---

## REFERENCIAS DE CÓDIGO

### Archivos Principales
- `src/tlr/pipeline.py` - Pipeline principal y load_pipeline()
- `src/tlr/detector.py` - TLDetector (SSD)
- `src/tlr/recognizer.py` - Recognizer (CNN clasificador)
- `src/tlr/selector.py` - select_tls() y Hungarian assignment
- `src/tlr/tracking.py` - TrafficLightTracker y SemanticDecision
- `src/tlr/hungarian_optimizer.py` - HungarianOptimizer
- `src/tlr/tools/utils.py` - Funciones de preprocesamiento, NMS, ProjectionROI

### Funciones Clave
| Función | Archivo | Líneas |
|---------|---------|--------|
| `load_pipeline()` | pipeline.py | 197-251 |
| `Pipeline.detect()` | pipeline.py | 26-76 |
| `Pipeline.recognize()` | pipeline.py | 78-120 |
| `Pipeline.forward()` | pipeline.py | 122-193 |
| `select_tls()` | selector.py | 8-68 |
| `SemanticDecision.update()` | tracking.py | 59-184 |
| `boxes2projections()` | utils.py | 336-350 |
| `box2projection()` | utils.py | 308-333 |
| `crop()` | utils.py | 219-240 |
| `preprocess4det()` | utils.py | 242-247 |
| `preprocess4rec()` | utils.py | 249-259 |
| `restore_boxes_to_full_image()` | utils.py | 261-302 |
| `nms()` | utils.py | 82-158 |

---

**Fin del documento.**
