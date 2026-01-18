# Infraestructura del Sistema de Detección de Semáforos (PyTorch)

## 1. Visión General

Este sistema es una implementación en PyTorch del pipeline de detección y reconocimiento de semáforos basado en la arquitectura Apollo TLR. Procesa cada frame de video en **5 etapas secuenciales**:

```
Frame de Cámara + Projection Boxes (archivo)
                    │
                    ▼
        ┌───────────────────────┐
        │  1. PREPROCESAMIENTO  │  Projection boxes → crops 270×270
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │     2. DETECCIÓN      │  CNN (Faster R-CNN) encuentra semáforos
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │     3. ASIGNACIÓN     │  Hungarian Algorithm: detections → projections
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │   4. RECONOCIMIENTO   │  CNN clasifica color (Black/R/Y/G)
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │      5. TRACKING      │  Consistencia temporal + reglas de seguridad
        └───────────┬───────────┘
                    ▼
            Resultado Final
```

A diferencia de Apollo que usa HD-Map con proyección 3D→2D dinámica, **nuestro sistema utiliza projection boxes pre-definidas en coordenadas 2D**, lo que lo hace ideal para procesamiento de video offline y testing.

---

## 2. Conceptos Fundamentales

### 2.1 Projection Boxes y Signal ID

Una **projection box** define una región de interés donde se espera encontrar un semáforo:

```python
box = [x1, y1, x2, y2, signal_id]
# Ejemplo: [850, 300, 890, 380, 0]
```

- `x1, y1, x2, y2`: Coordenadas del rectángulo en píxeles
- `signal_id`: Identificador del semáforo físico (persiste entre frames)

El `signal_id` es crítico: el sistema de tracking mantiene el historial asociado al semáforo físico, no a la posición en el array. Internamente se convierte a string: `"signal_0"`, `"signal_1"`, etc.

### 2.2 ProjectionROI Object

Cada box se convierte internamente a un objeto `ProjectionROI`:

```python
class ProjectionROI:
    # Geometría
    x, y, w, h: int           # Posición y tamaño
    center_x, center_y: int   # Centro calculado

    # IDs para tracking
    proj_id: str      # ID temporal en el frame (auto-generado: "proj_0")
    signal_id: str    # ID del semáforo físico (persistente: "signal_0")
```

### 2.3 Modelos de Machine Learning

El sistema utiliza **4 modelos** de redes neuronales:

| Modelo | Archivo | Propósito | Input | Output |
|--------|---------|-----------|-------|--------|
| **Detector** | `tl.torch` | Detectar semáforos en ROIs | 270×270×3 | Bboxes + tipo |
| **Recognizer Vertical** | `vert.torch` | Clasificar color | 96×32×3 | [Black, R, Y, G] |
| **Recognizer Horizontal** | `hori.torch` | Clasificar color | 32×96×3 | [Black, R, Y, G] |
| **Recognizer Quad** | `quad.torch` | Clasificar color | 64×64×3 | [Black, R, Y, G] |

**Detector (Faster R-CNN):**
- Arquitectura: RPN + RCNN con DFMB-PSROIAlign
- Triple NMS interno: RPN (IoU=0.7) → RCNN (IoU=0.5) → Global (IoU=0.6)
- Salida por detección: `[score, x1, y1, x2, y2, bg_prob, vert_prob, quad_prob, hori_prob]`

**Recognizers (CNN):**
- 5 capas convolucionales + BatchNorm + pooling adaptativo
- Salida: probabilidades `[Black, Red, Yellow, Green]`

---

## 3. Las 5 Etapas del Pipeline

### 3.1 Etapa 1: Preprocesamiento

**Archivos:** `src/tlr/tools/utils.py`, `src/tlr/pipeline.py`

**Entrada:**
- Imagen BGR (ej: 1920×1080×3)
- Lista de projection boxes `[[x1,y1,x2,y2,id], ...]`

**Proceso:**

1. **Conversión a ProjectionROI:**
   ```python
   projections = boxes2projections(boxes)
   # Cada box → ProjectionROI con signal_id
   ```

2. **Cálculo del crop expandido:**
   ```python
   crop_scale = 2.5
   min_crop_size = 270

   resize = crop_scale × max(width, height)
   resize = max(resize, 270)  # Mínimo 270px
   # Crop siempre CUADRADO, centrado en la projection
   ```

3. **Extracción y resize:**
   ```python
   crop = image[yt:yb, xl:xr]  # Extraer región
   resized = ResizeGPU(crop, 270×270)  # Interpolación bilineal
   normalized = resized - means_det  # [102.98, 115.95, 122.77] BGR
   ```

**Salida:** M crops de 270×270×3 normalizados

---

### 3.2 Etapa 2: Detección

**Archivo:** `src/tlr/pipeline.py` (método `detect`)

**Entrada:**
- M crops de 270×270
- Imagen original (para restaurar coordenadas)

**Proceso:**

1. **Inferencia CNN por cada crop:**
   ```python
   for projection in projections:
       input = preprocess4det(image, projection, means_det)
       bboxes = detector(input)  # Faster R-CNN
       detected_boxes.append(bboxes)
   ```

2. **Restaurar coordenadas a imagen original:**
   - Escalar de 270×270 al tamaño real del crop
   - Sumar offsets del crop

3. **Sort por score + NMS global:**
   ```python
   scores = torch.max(detections[:, 5:9], dim=1).values
   sorted_indices = torch.argsort(scores, descending=True)
   idxs = nms(detections_sorted[:, 1:5], threshold=0.6)
   ```

4. **Filtros de validación:**
   - Tamaño: 5px ≤ size ≤ 300px
   - Aspect ratio: 0.5 ≤ h/w ≤ 8.0
   - Confidence: ≥ 0.3

**Salida:** N detecciones `[score, x1, y1, x2, y2, bg, vert, quad, hori]`

---

### 3.3 Etapa 3: Asignación (Hungarian Algorithm)

**Archivo:** `src/tlr/selector.py`

**Entrada:**
- M projections (con signal_id)
- N detecciones válidas (sin identidad)

**Proceso:**

1. **Construir matriz de costos M×N:**
   ```python
   for each (projection_i, detection_j):
       # Score de distancia (Gaussiana 2D, σ=100)
       distance_score = exp(-0.5 × ((dx/100)² + (dy/100)²))

       # Score de confianza (clipped a 0.9)
       detection_score = min(max_class_prob, 0.9)

       # Combinación: 70% distancia, 30% confianza
       cost[i,j] = 0.7 × distance_score + 0.3 × detection_score

       # Penalizar si detection fuera del crop_roi
       if detection_outside_crop:
           cost[i,j] = 0.0
   ```

2. **Ejecutar Hungarian Algorithm:**
   ```python
   assignments = hungarian_optimizer.maximize(costs)
   # Retorna: [[proj_idx, det_idx], ...]
   ```

**Salida:** K asignaciones `[[proj_idx, det_idx], ...]`

---

### 3.4 Etapa 4: Reconocimiento

**Archivo:** `src/tlr/pipeline.py` (método `recognize`)

**Entrada:**
- Imagen original
- Detecciones válidas con tipo asignado

**Proceso:**

1. **Seleccionar recognizer según tipo:**
   ```python
   # tipo 1 → vert (96×32)
   # tipo 2 → quad (64×64)
   # tipo 3 → hori (32×96)
   recognizer, shape = classifiers[tl_type - 1]
   ```

2. **Preprocesar crop de la detección:**
   ```python
   crop = image[y1:y2, x1:x2]
   resized = resize(crop, shape)
   normalized = (resized - means_rec) × 0.01
   # means_rec = [66.56, 66.58, 69.06] BGR
   ```

3. **Inferencia + Prob2Color:**
   ```python
   probs = recognizer(input)  # [black, red, yellow, green]
   max_prob, max_idx = torch.max(probs)

   if max_prob > 0.5:
       color_id = max_idx  # 0=BLACK, 1=RED, 2=YELLOW, 3=GREEN
   else:
       color_id = 0  # Forzar BLACK (baja confianza)

   # Output one-hot
   result = [0, 0, 0, 0]
   result[color_id] = 1.0
   ```

**Salida:** N reconocimientos one-hot `[[0,1,0,0], ...]`

---

### 3.5 Etapa 5: Tracking Temporal

**Archivo:** `src/tlr/tracking.py`

**Entrada:**
- Timestamp del frame
- Asignaciones (proj_idx, det_idx)
- Reconocimientos
- Projections (para obtener signal_id)

**Proceso:**

El tracking mantiene historial **por signal_id** y aplica reglas de consistencia temporal:

1. **Ventana temporal (≤1.5s):**

   ```python
   if cur_color == "yellow":
       if prev_color == "red":
           # REGLA DE SEGURIDAD: Yellow después de Red → mantener Red
           # (secuencia inválida en semáforos reales)
           keep_previous_color()
       else:
           accept_yellow()

   elif cur_color in ("red", "green"):
       accept_color()
       # Detectar blink: bright→dark→bright en <0.55s
       if time_since_bright > 0.55s and had_dark_between:
           blink = True

   elif cur_color == "black":
       if prev_color in ("red", "green", "yellow"):
           keep_previous_color()  # Mantener último color conocido
       else:
           accept_black()
   ```

2. **Ventana expirada (>1.5s):**
   ```python
   # Reset completo, aceptar color sin validación
   color = cur_color
   blink = False
   ```

**Salida:** Dict `{signal_id: (color, blink)}`

---

## 4. Flujo de Datos

### Evolución de los datos a través del pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ INPUT                                                                        │
│  image: Tensor [H,W,3] BGR                                                  │
│  boxes: [[850,300,890,380,0], [1050,280,1090,360,1], ...]                   │
│  frame_ts: 10.25 (segundos)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DESPUÉS DE ETAPA 2 (Detección)                                              │
│                                                                              │
│  detections: Tensor [N, 9]                                                  │
│  [[0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # vertical          │
│   [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90]] # horizontal       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DESPUÉS DE ETAPA 3 (Asignación)                                             │
│                                                                              │
│  assignments: Tensor [K, 2]                                                 │
│  [[0, 0],   # proj_idx=0 (signal_0) → det_idx=0                            │
│   [1, 1]]   # proj_idx=1 (signal_1) → det_idx=1                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DESPUÉS DE ETAPA 4 (Reconocimiento)                                         │
│                                                                              │
│  recognitions: Tensor [N, 4] (one-hot)                                      │
│  [[0.0, 1.0, 0.0, 0.0],   # det_idx=0 → RED                                │
│   [0.0, 0.0, 0.0, 1.0]]   # det_idx=1 → GREEN                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DESPUÉS DE ETAPA 5 (Tracking)                                               │
│                                                                              │
│  revised: Dict[str, Tuple[str, bool]]                                       │
│  {                                                                           │
│      "signal_0": ("red", False),                                            │
│      "signal_1": ("green", False)                                           │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Uso del Sistema

### 5.1 Carga del Pipeline

```python
from tlr.pipeline import load_pipeline

# Cargar en GPU o CPU
pipeline = load_pipeline('cuda:0')  # o 'cpu'
```

### 5.2 Procesamiento de un Frame

```python
import torch

# Imagen BGR (ej: de cv2.imread o video)
image = torch.from_numpy(frame).to('cuda:0')

# Projection boxes con signal_id
boxes = [
    [850, 300, 890, 380, 0],   # signal_id=0
    [1050, 280, 1090, 360, 1], # signal_id=1
]

# Timestamp para tracking
frame_ts = frame_number / fps  # segundos

# Ejecutar pipeline
valid_dets, recognitions, assignments, invalid_dets, revised = \
    pipeline(image, boxes, frame_ts=frame_ts)

# Usar resultados
for proj_idx, det_idx in assignments:
    signal_id = f"signal_{boxes[proj_idx][4]}"
    color, blink = revised[signal_id]
    bbox = valid_dets[det_idx][1:5]  # x1, y1, x2, y2
    print(f"{signal_id}: {color} (blink={blink}) at {bbox}")
```

### 5.3 Formato de Projection Boxes (archivo)

```
# projection_bboxes_master.txt
# x1 y1 x2 y2 signal_id
850 300 890 380 0
1050 280 1090 360 1
1200 320 1235 390 2
```

---

## 6. Parámetros del Sistema

### 6.1 Preprocesamiento

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `crop_scale` | 2.5 | Factor de expansión del ROI |
| `min_crop_size` | 270 | Tamaño mínimo del crop (px) |
| `means_det` | [102.98, 115.95, 122.77] | Medias BGR para detector |

### 6.2 Detección

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `nms_threshold` | 0.6 | IoU threshold para NMS global |
| `min_confidence` | 0.3 | Confianza mínima para detecciones |
| `min_size` | 5 | Tamaño mínimo de detección (px) |
| `max_size` | 300 | Tamaño máximo de detección (px) |
| `min_aspect` | 0.5 | Aspect ratio mínimo (h/w) |
| `max_aspect` | 8.0 | Aspect ratio máximo (h/w) |

### 6.3 Asignación

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `gaussian_sigma` | 100.0 | Sigma para score de distancia |
| `distance_weight` | 0.7 | Peso de distancia en score combinado |
| `detection_weight` | 0.3 | Peso de confianza en score combinado |
| `max_detection_score` | 0.9 | Clip máximo para detection score |

### 6.4 Reconocimiento

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `means_rec` | [66.56, 66.58, 69.06] | Medias BGR para recognizers |
| `scale_factor` | 0.01 | Factor de escala post-normalización |
| `prob2color_threshold` | 0.5 | Umbral para clasificar color |
| `vert_shape` | (96, 32) | Tamaño input recognizer vertical |
| `hori_shape` | (32, 96) | Tamaño input recognizer horizontal |
| `quad_shape` | (64, 64) | Tamaño input recognizer quad |

### 6.5 Tracking

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `revise_time_s` | 1.5 | Ventana temporal de revisión (s) |
| `blink_threshold_s` | 0.55 | Umbral para detectar blink (s) |
| `hysteretic_threshold` | 1 | Frames para confirmar cambio desde BLACK |

---

## 7. Estructura de Archivos

```
src/tlr/
├── pipeline.py              # Pipeline principal y load_pipeline()
├── detector.py              # TLDetector (Faster R-CNN)
├── recognizer.py            # Recognizer (CNN clasificador)
├── selector.py              # select_tls() y Hungarian assignment
├── tracking.py              # TrafficLightTracker y SemanticDecision
├── hungarian_optimizer.py   # Algoritmo Húngaro (Munkres)
├── faster_rcnn.py           # Componentes Faster R-CNN
├── tools/
│   └── utils.py             # Preprocesamiento, NMS, ProjectionROI
├── weights/
│   ├── tl.torch             # Pesos detector
│   ├── vert.torch           # Pesos recognizer vertical
│   ├── hori.torch           # Pesos recognizer horizontal
│   └── quad.torch           # Pesos recognizer quad
└── confs/
    ├── bbox_reg_param.json
    ├── detection_output_ssd_param.json
    ├── dfmb_psroi_pooling_param.json
    ├── rcnn_bbox_reg_param.json
    └── rcnn_detection_output_ssd_param.json
```

---

## 8. Compatibilidad con Apollo

Este sistema es **funcionalmente idéntico** al pipeline Apollo TLR en todas las etapas críticas:

| Aspecto | Estado |
|---------|--------|
| Arquitectura detector | ✅ Idéntica (Faster R-CNN) |
| Parámetros de preprocesamiento | ✅ Idénticos |
| Algoritmo de asignación | ✅ Idéntico (Hungarian) |
| Lógica de reconocimiento | ✅ Idéntica (Prob2Color) |
| Reglas de tracking | ✅ Idénticas (todas las reglas de seguridad) |
| Parámetros numéricos | ✅ Idénticos (thresholds, weights, means) |

**Simplificaciones respecto a Apollo:**
- No requiere HD-Map ni proyección 3D→2D (usa boxes pre-definidas)
- No soporta multi-cámara (mono-cámara)
- Semantic voting no implementado (Apollo tampoco lo usa en la práctica)

Estas simplificaciones hacen el sistema ideal para **procesamiento offline** y **testing con datasets estáticos**.
