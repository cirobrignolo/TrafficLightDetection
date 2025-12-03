# Flujo del Sistema Propio de Detección de Semáforos

**Documento:** Explicación narrativa del pipeline completo de detección y reconocimiento de semáforos
**Sistema:** Implementación propia basada en PyTorch
**Fecha:** 2025-11-26

---

## Introducción

Este documento describe el flujo completo del sistema de detección de semáforos implementado en Python/PyTorch. A diferencia del sistema original de Apollo que usa HD-Map y semantic IDs, **nuestro sistema utiliza projection boxes manualmente definidas** que representan las regiones de interés donde se espera encontrar semáforos.

### Diferencias Clave con Apollo Original

| Característica | Apollo Original | Nuestro Sistema |
|----------------|-----------------|-----------------|
| **Entrada de ROIs** | HD-Map con posiciones 3D georreferenciadas | Projection boxes manualmente definidas `[x1,y1,x2,y2,id]` |
| **Proyección 3D→2D** | Sí (usa pose del vehículo + calibración) | No (boxes ya están en coordenadas 2D) |
| **Semantic IDs** | Sí (agrupa semáforos relacionados) | No (cada projection box tiene su propio `id`) |
| **Selección de Cámara** | Sí (telephoto vs wide-angle) | No (single camera) |
| **Tracking Temporal** | Por semantic group con voting | Por projection ID individual |
| **Query HD-Map** | Dinámico (consulta continua según pose) | Estático (boxes cargadas desde archivo) |

### Concepto Clave: Projection Boxes

En nuestro sistema, una **projection box** es una caja delimitadora `[x1, y1, x2, y2, id]` que define:
- `x1, y1, x2, y2`: Coordenadas del rectángulo en la imagen (píxeles)
- `id`: Identificador único de esa región (análogo al `id` del semáforo en Apollo, pero sin semantic grouping)

Estas boxes se definen manualmente usando herramientas como `select_projection_and_append.py` y se guardan en archivos YAML por video/secuencia.

**Ejemplo:**
```yaml
# frames_labeled/video_01/projection_boxes.yaml
- [850, 300, 890, 380, 1]  # Semáforo izquierdo
- [1050, 280, 1090, 360, 2]  # Semáforo derecho
- [1200, 320, 1235, 390, 3]  # Semáforo lejano
```

---

## Estructura de Datos: Tensores en Cada Etapa

A diferencia del objeto `TrafficLight` de Apollo, nuestro sistema trabaja con **tensores de PyTorch** en cada etapa:

### 1. **Input: Projection Boxes** (Python list)
```python
boxes = [
    [850, 300, 890, 380, 1],  # [x1, y1, x2, y2, id]
    [1050, 280, 1090, 360, 2],
    [1200, 320, 1235, 390, 3]
]
# M boxes (M = 3 en este ejemplo)
```

### 2. **Detections** (Tensor N×9)
```python
detections = torch.Tensor([
    # [score, x1, y1, x2, y2, bg_prob, vert_prob, quad_prob, hori_prob]
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # Detected TL #0
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],  # Detected TL #1
    [0.82, 1202, 325, 1233, 385, 0.03, 0.84, 0.08, 0.05],  # Detected TL #2
    # ... más detecciones si hay múltiples por projection
])
# N detections (puede ser N > M si detecta varios por projection)
```

### 3. **Recognitions** (Tensor N×4)
```python
recognitions = torch.Tensor([
    # [black_prob, red_prob, yellow_prob, green_prob]
    [0.0, 0.98, 0.01, 0.01],  # Detection #0 → RED
    [0.0, 0.02, 0.03, 0.95],  # Detection #1 → GREEN
    [0.0, 0.85, 0.10, 0.05],  # Detection #2 → RED
])
# Mismo número de filas que valid_detections
```

### 4. **Assignments** (Tensor K×2)
```python
assignments = torch.Tensor([
    [0, 0],  # projection_idx=0 → detection_idx=0
    [1, 1],  # projection_idx=1 → detection_idx=1
    [2, 2],  # projection_idx=2 → detection_idx=2
])
# K assignments (idealmente K ≤ M, puede ser K < M si algunas projections no tienen detección)
```

### 5. **Revised States** (dict, solo si tracking habilitado)
```python
revised = {
    1: ('red', False),     # projection_id=1 → (color, blink)
    2: ('green', False),   # projection_id=2 → (color, blink)
    3: ('red', False),     # projection_id=3 → (color, blink)
}
# Dict con projection IDs como keys (viene del 5to elemento de cada box)
```

---

## Arquitectura del Pipeline

El punto de entrada principal es `load_pipeline()` en `src/tlr/pipeline.py`:

```python
# Carga del pipeline
pipeline = load_pipeline('cuda:0')  # o 'cpu'

# Uso
valid_detections, recognitions, assignments, invalid_detections, revised = \
    pipeline(image, boxes, frame_ts=timestamp)
```

**Componentes cargados:**
1. **Detector** (`TLDetector`): Red SSD-style para detectar semáforos
2. **Classifiers** (3 `Recognizer`s): Clasificadores por orientación (vertical, quad, horizontal)
3. **Hungarian Optimizer** (`HungarianOptimizer`): Para asignación óptima projection→detection
4. **Tracker** (`TrafficLightTracker`): Para revisión temporal del estado

---

## ETAPA 1: PREPROCESAMIENTO POR PROJECTION

**Archivo:** `src/tlr/tools/utils.py:238-243` (función `preprocess4det`)
**Entrada:** Imagen completa + una projection box
**Salida:** Crop normalizado de 270×270 píxels

### ¿Qué hace?

Por cada projection box `[x1, y1, x2, y2, id]`:

1. **Calcula ROI expandido** (`crop()` en `utils.py:215-236`):
   - Centro: `center_x = (x1 + x2) / 2`, `center_y = (y1 + y2) / 2`
   - Tamaño base: `max(width, height)` de la projection box
   - Expansión: `crop_scale = 2.5` (igual que Apollo)
   - Tamaño mínimo: `min_crop_size = 270`
   - **Resultado:** Crop cuadrado centrado en la projection, expandido 2.5× para capturar contexto

2. **Extrae región de la imagen:**
   ```python
   xl, xr, yt, yb = crop(image.shape, projection)
   src = image[yt:yb, xl:xr]
   ```

3. **Resize a 270×270 píxels:**
   - Usa interpolación bilineal GPU-acelerada (`ResizeGPU()` en `utils.py:173-213`)
   - **Normalización:** Resta las medias del dataset `[102.98, 115.95, 122.77]` (BGR)
   - Similar a Apollo's `DataProvider::GetImage()`

4. **Formato de salida:**
   - Tensor `[270, 270, 3]` (HWC format)
   - Valores normalizados (imagen - medias)

### Cardinalidad
```
M projection boxes → M crops de 270×270
```

### Estructura de Datos en Esta Etapa

**Input (Python list):**
```python
boxes = [
    [850, 300, 890, 380, 1],  # ✅ x1, y1, x2, y2, id (definidos manualmente)
]
```

**Output (Tensor):**
```python
preprocessed_crop = torch.Tensor([270, 270, 3])  # ✅ Crop normalizado listo para detector
# Valores: imagen_bgr - [102.98, 115.95, 122.77]
```

**Diferencia con Apollo:**
Apollo calcula estos ROIs dinámicamente desde HD-Map (Query HD-Map → Proyección 3D→2D → Crop). Nosotros los tenemos **pre-definidos en coordenadas 2D**.

---

## ETAPA 2: DETECCIÓN CNN

**Archivo:** `src/tlr/pipeline.py:26-49` (método `detect`)
**Entrada:** Imagen completa + M projection boxes
**Salida:** Tensor N×9 con detecciones (después de NMS)

### ¿Qué hace?

1. **Preprocesa cada projection** (ver Etapa 1):
   ```python
   projections = boxes2projections(boxes)  # Convierte a objetos ProjectionROI
   for projection in projections:
       input = preprocess4det(image, projection, self.means_det)  # [270,270,3]
       bboxes = self.detector(input.unsqueeze(0).permute(0, 3, 1, 2))  # Inference
       detected_boxes.append(bboxes)
   ```

2. **Inference CNN** (`TLDetector` en `src/tlr/detector.py:24-40`):

   **Arquitectura (SSD-style):**
   ```
   Input [1,3,270,270]
     ↓
   FeatureNet (VGG-like backbone)
     ├→ rpn_cls_prob_reshape [1,30,34,34]  # RPN classification
     ├→ rpn_bbox_pred [1,60,34,34]         # RPN box regression
     └→ ft_add_left_right [1,490,34,34]    # Feature maps
     ↓
   RPNProposalSSD (generates ~52 proposals)
     → rois [52,5]  # [batch_idx, x1, y1, x2, y2]
     ↓
   DFMBPSROIAlign (Deformable Position-Sensitive ROI Align)
     → psroi_rois [52,10,7,7]
     ↓
   Flatten + FC(2048) + ReLU
     → inner_rois [52,2048]
     ↓
   ┌──────────┬──────────┐
   │ cls_score│ bbox_pred│
   │ FC(4)    │ FC(16)   │
   └──────────┴──────────┘
       ↓            ↓
   Softmax   BBox Regression
     [52,4]       [52,16]
       └────────┬────────┘
                ↓
   RCNNProposal (NMS interno, threshold=0.3)
     → bboxes [~3-10, 9]
   ```

   **Output por projection:**
   ```python
   bboxes = [
       # [score, x1, y1, x2, y2, bg_prob, vert_prob, quad_prob, hori_prob]
       [0.95, 125, 80, 145, 155, 0.01, 0.92, 0.05, 0.02],
       [0.82, 130, 85, 148, 160, 0.05, 0.88, 0.04, 0.03],
       # ... (coordenadas relativas al crop 270×270)
   ]
   ```

3. **Restaura coordenadas a imagen completa** (`restore_boxes_to_full_image` en `utils.py:261-302`):

   **Bug fix implementado (Apollo-style):**
   - **Problema original:** Coordenadas del detector (270×270) se sumaban directamente al offset del crop
   - **Solución correcta (Apollo):**
     1. Escalar coordenadas del detector al tamaño real del crop:
        ```python
        crop_width = xr - xl + 1
        crop_height = yb - yt + 1
        scale_x = crop_width / 270
        scale_y = crop_height / 270

        detection[:, 1:5] *= [scale_x, scale_y, scale_x, scale_y]  # Escalar
        ```
     2. Sumar offsets del crop:
        ```python
        detection[:, 1:5] += [xl, yt, xl, yt]  # Trasladar
        ```

   **Referencia Apollo:** `detection.cc:329-356`

4. **Concatena detecciones de todas las projections:**
   ```python
   detections = torch.vstack(detected_boxes).reshape(-1, 9)
   # Puede haber múltiples detecciones por projection
   ```

5. **Sort + NMS global** (pipeline.py:37-47):

   **APOLLO FIX:** Ordenar por score ANTES de NMS (como Apollo en `detection.cc:381-390`):
   ```python
   scores = detections[:, 0]  # detect_score
   sorted_indices = torch.argsort(scores, descending=True)
   detections_sorted = detections[sorted_indices]

   # NMS con threshold 0.6 (mismo que Apollo: detection.h:87)
   idxs = nms(detections_sorted[:, 1:5], 0.6)
   detections = detections_sorted[idxs]
   ```

### Cardinalidad
```
M projections → K detecciones brutas → N detecciones después de NMS
(K ≈ 3-10 por projection, N ≤ K después de NMS)
```

### Estructura de Datos en Esta Etapa

**Input (per projection):**
```python
preprocessed_crop = torch.Tensor([270, 270, 3])  # ✅ Crop normalizado
```

**Output del detector (per projection, coordenadas en crop):**
```python
bboxes_crop = torch.Tensor([
    [0.95, 125, 80, 145, 155, 0.01, 0.92, 0.05, 0.02],  # ✅ score, box, class_probs
    # Coordenadas relativas a crop 270×270
])
```

**Output restaurado (coordenadas en imagen completa):**
```python
detections_global = torch.Tensor([
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # ✅ Coordenadas globales
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],
    # ... más detecciones
])
```

**Output final (después de NMS):**
```python
detections = torch.Tensor([
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # ✅ Top detections
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],  # ✅ No redundantes
    # Detecciones superpuestas eliminadas por NMS
])
```

---

## ETAPA 3: FILTRADO POR TIPO Y ASIGNACIÓN

**Archivo:** `src/tlr/pipeline.py:122-126` + `src/tlr/selector.py`
**Entrada:** Detecciones N×9 + projection boxes originales
**Salida:** Valid detections + Assignments (M×2)

### ¿Qué hace?

1. **Clasifica tipo de semáforo** (pipeline.py:122):
   ```python
   tl_types = torch.argmax(detections[:, 5:], dim=1)
   # 0=background, 1=vertical, 2=quadrant, 3=horizontal
   ```

2. **Filtra detecciones válidas** (pipeline.py:123-125):
   ```python
   valid_mask = tl_types != 0  # Elimina background
   valid_detections = detections[valid_mask]
   invalid_detections = detections[~valid_mask]
   ```

3. **Asignación Húngara** (`select_tls` en `selector.py:8-68`):

   **Objetivo:** Asignar cada projection box a **máximo una** detección válida.

   **Matriz de costos:**
   Para cada par (projection_i, detection_j):
   ```python
   # Centro de projection (Apollo usa esto también)
   center_projection = [(x1+x2)/2, (y1+y2)/2]

   # Centro de detection
   center_detection = [(det_x1+det_x2)/2, (det_y1+det_y2)/2]

   # Score basado en distancia (Gaussiana 2D)
   gaussian_score = 100.0  # sigma
   distance_score = exp(-0.5 * (
       (cx_p - cx_d)² / σ² + (cy_p - cy_d)² / σ²
   ))

   # Score basado en confianza del detector
   detection_score = max(detect_score, 0.9)  # Cap a 0.9

   # Score combinado (70% distancia, 30% confianza)
   cost[i, j] = 0.3 * detection_score + 0.7 * distance_score
   ```

   **APOLLO FIX:** Validar que la detección está dentro del crop ROI (selector.py:38-45):
   ```python
   coors = crop(item_shape, projection)  # [xl, xr, yt, yb]
   det_box = detection[1:5]  # [x1, y1, x2, y2]

   # Si la detección está fuera del crop → score = 0
   if coors[0] > det_box[0] or \
      coors[1] < det_box[2] or \
      coors[2] > det_box[1] or \
      coors[3] < det_box[3]:
       costs[i, j] = 0.0
   ```

   **Referencia Apollo:** `select.cc:76-83`

   **Algoritmo Húngaro:**
   ```python
   assignments = hungarian_optimizer.maximize(costs)
   # Resultado: lista de pares [projection_idx, detection_idx]
   ```

   **Post-procesamiento** (selector.py:50-63):
   - Eliminar duplicados (una projection no puede tener 2 detections)
   - Verificar índices válidos
   - Resultado final: Tensor K×2

### Cardinalidad
```
M projections + N valid_detections → K assignments (K ≤ min(M, N))
```

### Estructura de Datos en Esta Etapa

**Input (detecciones después de NMS):**
```python
detections = torch.Tensor([
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # ✅ score, box, class_probs
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],
    [0.60, 1205, 328, 1230, 382, 0.85, 0.10, 0.03, 0.02],  # background!
])
```

**Después de filtrado:**
```python
tl_types = torch.Tensor([1, 3, 0])  # ✅ 1=vert, 3=hori, 0=bg

valid_detections = torch.Tensor([
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # ✅ Tipo vertical
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],  # ✅ Tipo horizontal
])

invalid_detections = torch.Tensor([
    [0.60, 1205, 328, 1230, 382, 0.85, 0.10, 0.03, 0.02],  # ❌ Background
])
```

**Matriz de costos (ejemplo M=3 projections, N=2 valid detections):**
```python
costs = torch.Tensor([
    [0.92, 0.15],  # proj 0: alta afinidad con det 0, baja con det 1
    [0.18, 0.89],  # proj 1: baja con det 0, alta con det 1
    [0.05, 0.08],  # proj 2: baja con ambas (sin detección cercana)
])
```

**Output de asignación:**
```python
assignments = torch.Tensor([
    [0, 0],  # ✅ projection_idx=0 → detection_idx=0
    [1, 1],  # ✅ projection_idx=1 → detection_idx=1
    # projection 2 sin asignación (score muy bajo)
])
```

**Nota:** `projection_idx` se refiere al índice en la lista `boxes`, NO al `id` del 5to elemento. El `id` se usa después en tracking.

---

## ETAPA 4: RECONOCIMIENTO DE ESTADO

**Archivo:** `src/tlr/pipeline.py:51-93` (método `recognize`)
**Entrada:** Imagen completa + valid_detections + tipos de semáforo
**Salida:** Tensor N×4 con probabilidades por clase

### ¿Qué hace?

**Por cada detección válida:**

1. **Selecciona clasificador según tipo** (pipeline.py:61):
   ```python
   tl_type = tl_types[i]  # 1=vert, 2=quad, 3=hori
   recognizer, shape = self.classifiers[tl_type - 1]
   # classifiers = [
   #   (vert_recognizer, (96, 32, 3)),   # idx 0 → tipo 1
   #   (quad_recognizer, (64, 64, 3)),   # idx 1 → tipo 2
   #   (hori_recognizer, (32, 96, 3))    # idx 2 → tipo 3
   # ]
   ```

2. **Preprocesa ROI de la detección** (`preprocess4rec` en `utils.py:245-256`):
   ```python
   det_box = detection[1:5].type(torch.long)  # [x1, y1, x2, y2]
   src = image[y1:y2, x1:x2]  # Crop exacto de la detección

   # Resize según orientación:
   # - Vertical: 96×32
   # - Quad: 64×64
   # - Horizontal: 32×96

   input = preprocess4rec(img, det_box, shape, self.means_rec)
   # Resta medias: [69.06, 66.58, 66.56] (diferentes al detector!)
   ```

3. **Inference del clasificador** (`Recognizer` en `src/tlr/recognizer.py:41-63`):

   **Arquitectura:**
   ```
   Input [1,3,H,W]  (H,W según orientación)
     ↓
   Conv1(3→32) + BN + Scale + ReLU + MaxPool(3×3, stride=2)
     ↓
   Conv2(32→64) + BN + Scale + ReLU + MaxPool(3×3, stride=2)
     ↓
   Conv3(64→128) + BN + Scale + ReLU + MaxPool(3×3, stride=2)
     ↓
   Conv4(128→128) + BN + Scale + ReLU + MaxPool(3×3, stride=2)
     ↓
   Conv5(128→128) + BN + Scale + ReLU + AvgPool(orientación-específico)
     ↓
   FC(128) + BN + Scale + ReLU
     ↓
   FC(4) → Softmax
     → [black_prob, red_prob, yellow_prob, green_prob]
   ```

4. **Apollo's Prob2Color Logic** (pipeline.py:63-90):

   **Preprocesamiento adicional:**
   ```python
   input_scaled = input.permute(2, 0, 1).unsqueeze(0)  # HWC → NCHW
   input_scaled = input_scaled * 0.01  # Apollo's scale factor
   ```

   **Decisión de color:**
   ```python
   output_probs = recognizer(input_scaled)[0]  # [4]
   max_prob, max_idx = torch.max(output_probs, dim=0)
   threshold = 0.5  # Apollo's classify_threshold

   # Si probabilidad < threshold → forzar a BLACK
   if max_prob > threshold:
       color_id = max_idx.item()  # 0=BLACK, 1=RED, 2=YELLOW, 3=GREEN
   else:
       color_id = 0  # Forzar a BLACK (desconocido)

   # Crear one-hot result (Apollo style)
   result = torch.zeros(4)
   result[color_id] = 1.0
   ```

   **Referencia Apollo:** `recognition.cc:48-65` (función `Prob2Color`)

5. **Concatena resultados:**
   ```python
   recognitions = torch.vstack(recognitions).reshape(-1, 4)
   ```

### Cardinalidad
```
N valid_detections → N recognitions (one-to-one)
```

### Estructura de Datos en Esta Etapa

**Input:**
```python
valid_detections = torch.Tensor([
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # ✅ Detección válida
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],
])

tl_types = torch.Tensor([1, 3])  # ✅ 1=vertical, 3=horizontal
```

**Crops preprocesados (diferentes tamaños):**
```python
# Detección 0 (vertical):
input_vert = torch.Tensor([96, 32, 3])  # ✅ Crop 96×32 normalizado

# Detección 1 (horizontal):
input_hori = torch.Tensor([32, 96, 3])  # ✅ Crop 32×96 normalizado
```

**Output de clasificadores (raw probabilities):**
```python
# Detección 0:
probs_0 = torch.Tensor([0.02, 0.87, 0.08, 0.03])  # [black, red, yellow, green]
max_prob = 0.87 > 0.5 → color_id = 1 (RED)

# Detección 1:
probs_1 = torch.Tensor([0.01, 0.03, 0.02, 0.94])
max_prob = 0.94 > 0.5 → color_id = 3 (GREEN)
```

**Output final (one-hot):**
```python
recognitions = torch.Tensor([
    [0.0, 1.0, 0.0, 0.0],  # ✅ RED (detección 0)
    [0.0, 0.0, 0.0, 1.0],  # ✅ GREEN (detección 1)
])
```

**Caso especial (baja confianza):**
```python
probs_uncertain = torch.Tensor([0.30, 0.35, 0.25, 0.10])
max_prob = 0.35 < 0.5 → color_id = 0 (BLACK, forzado)

result = torch.Tensor([1.0, 0.0, 0.0, 0.0])  # ✅ BLACK (desconocido)
```

---

## ETAPA 5: TRACKING TEMPORAL (OPCIONAL)

**Archivo:** `src/tlr/tracking.py` + `src/tlr/pipeline.py:136-145`
**Entrada:** frame_ts, assignments, recognitions
**Salida:** dict {projection_id → (revised_color, blink)}

### ¿Qué hace?

**Diferencia clave con Apollo:**
Apollo hace tracking por **semantic groups** (múltiples semáforos comparten historial). Nuestro sistema hace tracking **por projection ID individual**.

**Flujo:**

1. **Conversión de tensores a listas** (pipeline.py:142-143):
   ```python
   assigns_list = assignments.cpu().tolist()  # [[0, 0], [1, 1], ...]
   recs_list = recognitions.cpu().tolist()    # [[0,1,0,0], [0,0,0,1], ...]
   ```

2. **Llamada al tracker** (pipeline.py:144):
   ```python
   revised = self.tracker.track(frame_ts, assigns_list, recs_list)
   ```

3. **Procesamiento por projection ID** (`SemanticDecision.update` en `tracking.py:53-123`):

   **Por cada assignment `(proj_id, det_idx)`:**

   a. **Determinar color actual:**
   ```python
   cls = int(max(range(4), key=lambda i: recognitions[det_idx][i]))
   color = ["black", "red", "yellow", "green"][cls]
   ```

   b. **Obtener o crear historial:**
   ```python
   if proj_id not in self.history:
       self.history[proj_id] = SemanticTable(proj_id, frame_ts, color)
   st = self.history[proj_id]  # Estado histórico
   ```

   c. **APOLLO'S HYSTERESIS LOGIC** (tracking.py:78-107):

   **Regla 1: Yellow Blink Detection**
   ```python
   dt = frame_ts - st.time_stamp
   if color == "yellow" and dt < self.blink_threshold_s:  # 0.55s
       # Parpadeo amarillo → tratarlo como ROJO por seguridad
       st.blink = True
       color = "red"  # Override
   else:
       st.blink = False
   ```

   **Regla 2: Sequence Safety**
   ```python
   # "Cualquier amarillo después de rojo se resetea a rojo por seguridad"
   if color == "yellow" and st.color == "red":
       color = "red"  # Mantener rojo hasta que aparezca verde
   ```

   **Regla 3: Histéresis desde BLACK**
   ```python
   if st.color == "black":
       # Conservador: necesita evidencia para salir del estado desconocido
       if st.hysteretic_color == color:
           st.hysteretic_count += 1
       else:
           st.hysteretic_color = color
           st.hysteretic_count = 1

       # Solo cambiar DESDE black con suficiente evidencia
       if st.hysteretic_count > self.hysteretic_threshold:  # > 1
           st.color = color
           st.hysteretic_count = 0
   else:
       # Entre estados conocidos (red/green/yellow), actualizar inmediatamente
       st.color = color
       st.hysteretic_count = 0
   ```

   d. **Actualizar timestamps:**
   ```python
   st.time_stamp = frame_ts
   if color in ("red", "green"):
       st.last_bright_time = frame_ts
   else:
       st.last_dark_time = frame_ts
   ```

   e. **Resetear histéresis si pasa la ventana temporal:**
   ```python
   if frame_ts - st.time_stamp > self.revise_time_s:  # 1.5s
       st.hysteretic_count = 0
   ```

   f. **Guardar resultado:**
   ```python
   results[proj_id] = (st.color, st.blink)
   ```

4. **Return revised states:**
   ```python
   return results  # dict {proj_id: (color, blink)}
   ```

### Cardinalidad
```
K assignments → K revised states (one per assigned projection)
```

### Estructura de Datos en Esta Etapa

**Input (del pipeline):**
```python
frame_ts = 10.25  # ✅ Timestamp en segundos

assignments = torch.Tensor([
    [0, 0],  # ✅ proj_idx=0 (proj_id=1) → det_idx=0
    [1, 1],  # ✅ proj_idx=1 (proj_id=2) → det_idx=1
])

recognitions = torch.Tensor([
    [0.0, 1.0, 0.0, 0.0],  # ✅ det_idx=0 → RED
    [0.0, 0.0, 0.0, 1.0],  # ✅ det_idx=1 → GREEN
])

# Necesitamos recuperar los projection IDs de las boxes originales:
boxes = [
    [850, 300, 890, 380, 1],  # proj_idx=0 → proj_id=1
    [1050, 280, 1090, 360, 2],  # proj_idx=1 → proj_id=2
    [1200, 320, 1235, 390, 3],  # proj_idx=2 → proj_id=3 (sin asignación)
]
```

**Historial interno (SemanticTable por projection ID):**
```python
# Frame anterior (t=10.15):
self.history = {
    1: SemanticTable(
        semantic_id=1,
        time_stamp=10.15,
        color="red",
        last_bright_time=10.15,
        last_dark_time=9.80,
        blink=False,
        hysteretic_color="red",
        hysteretic_count=0
    ),
    2: SemanticTable(
        semantic_id=2,
        time_stamp=10.15,
        color="green",
        last_bright_time=10.15,
        last_dark_time=9.50,
        blink=False,
        hysteretic_color="green",
        hysteretic_count=0
    ),
}
```

**Procesamiento frame actual (t=10.25):**

Para `proj_id=1` (assignment [0, 0]):
```python
# Color detectado: RED
current_color = "red"
dt = 10.25 - 10.15 = 0.10s

# ¿Es yellow blink? No (es red, no yellow)
# ¿Es yellow después de red? No (es red)
# ¿Color anterior era black? No (era red)
# → Actualizar inmediatamente: st.color = "red"

st.time_stamp = 10.25
st.last_bright_time = 10.25  # (porque es red)
```

Para `proj_id=2` (assignment [1, 1]):
```python
# Color detectado: GREEN
current_color = "green"
dt = 10.25 - 10.15 = 0.10s

# ¿Es yellow blink? No (es green)
# ¿Es yellow después de red? No (es green)
# ¿Color anterior era black? No (era green)
# → Actualizar inmediatamente: st.color = "green"

st.time_stamp = 10.25
st.last_bright_time = 10.25  # (porque es green)
```

**Output:**
```python
revised = {
    1: ('red', False),    # ✅ proj_id=1 → RED, no blink
    2: ('green', False),  # ✅ proj_id=2 → GREEN, no blink
}
```

**Caso especial: Yellow Blink**

Supongamos frame siguiente (t=10.40):
```python
# proj_id=2 detecta YELLOW
current_color = "yellow"
st.color = "green"  # Estado anterior
dt = 10.40 - 10.25 = 0.15s

# ¿Es yellow blink? 0.15s < 0.55s → SÍ!
# → st.blink = True
# → color = "red"  # Override por seguridad (Apollo safety rule)

revised = {
    2: ('red', True),  # ✅ Forzado a RED por yellow blink
}
```

**Caso especial: Histéresis desde BLACK**

Supongamos projection nueva (t=10.50):
```python
# proj_id=3 aparece por primera vez
current_color = "green"
st.color = "black"  # Estado inicial (sin historial)

# ¿Color anterior era black? SÍ
# → Histéresis activada
# st.hysteretic_color = "green"
# st.hysteretic_count = 1

# ¿hysteretic_count > 1? NO (es 1, threshold=1)
# → NO cambiar todavía, mantener "black"

revised = {
    3: ('black', False),  # ✅ Esperando más evidencia
}

# Frame siguiente (t=10.60) con green otra vez:
# st.hysteretic_count = 2 > 1 → AHORA cambiar
revised = {
    3: ('green', False),  # ✅ Ahora sí acepta green
}
```

---

## ETAPA 6: OUTPUT FINAL

**Archivo:** `src/tlr/pipeline.py:95-146` (método `forward`)
**Return del pipeline**

### Estructura de Datos de Salida

```python
# Llamada:
valid_detections, recognitions, assignments, invalid_detections, revised = \
    pipeline(image, boxes, frame_ts=10.25)

# Retorno completo:
```

**1. `valid_detections` (Tensor N×9):**
```python
torch.Tensor([
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # ✅ Detección válida #0
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],  # ✅ Detección válida #1
    # ... (solo semáforos, no background)
])
```

**2. `recognitions` (Tensor N×4):**
```python
torch.Tensor([
    [0.0, 1.0, 0.0, 0.0],  # ✅ Det #0 → RED
    [0.0, 0.0, 0.0, 1.0],  # ✅ Det #1 → GREEN
])
```

**3. `assignments` (Tensor K×2):**
```python
torch.Tensor([
    [0, 0],  # ✅ proj_idx=0 → det_idx=0
    [1, 1],  # ✅ proj_idx=1 → det_idx=1
])
```

**4. `invalid_detections` (Tensor L×9):**
```python
torch.Tensor([
    [0.60, 1205, 328, 1230, 382, 0.85, 0.10, 0.03, 0.02],  # ❌ Background
    # ... (detecciones clasificadas como fondo)
])
```

**5. `revised` (dict o None):**
```python
# Si tracker habilitado:
{
    1: ('red', False),    # ✅ proj_id=1 → color revisado, no blink
    2: ('green', False),  # ✅ proj_id=2 → color revisado, no blink
}

# Si tracker deshabilitado:
None
```

### Uso Típico del Output

```python
# Dibujar bounding boxes con color revisado (si tracking habilitado)
for (proj_idx, det_idx) in assignments:
    det = valid_detections[det_idx]
    box = [det[1], det[2], det[3], det[4]]  # x1, y1, x2, y2

    # Obtener proj_id del 5to elemento
    proj_id = int(boxes[proj_idx][4])

    if revised is not None:
        color, blink = revised[proj_id]
    else:
        # Sin tracking, usar recognition directo
        rec = recognitions[det_idx]
        color_idx = torch.argmax(rec).item()
        color = ["black", "red", "yellow", "green"][color_idx]
        blink = False

    # Dibujar
    draw_box(image, box, color, blink)
```

---

## RESUMEN DE CARDINALIDADES

| Etapa | Input | Output | Relación |
|-------|-------|--------|----------|
| **1. Preproc** | M projection boxes | M crops 270×270 | 1:1 |
| **2. Detección** | M crops | N detections (después NMS) | M:N (N puede ser > M) |
| **3. Filtrado** | N detections | N' valid + (N-N') invalid | 1:1 split |
| **4. Asignación** | M projections + N' valid dets | K assignments | M:N' → K (K ≤ min(M,N')) |
| **5. Reconocimiento** | N' valid detections | N' recognitions | 1:1 |
| **6. Tracking** | K assignments | K revised states | 1:1 |

**Ejemplo numérico:**
```
M = 3 projection boxes
↓ Detección
N = 5 detections (varias por projection)
↓ NMS
N = 3 detections (después de eliminar redundantes)
↓ Filtrado
N' = 2 valid detections (1 era background)
↓ Asignación Húngara
K = 2 assignments (las 2 válidas se asignan a 2 projections)
↓ Reconocimiento
N' = 2 recognitions (one-to-one con valid dets)
↓ Tracking
K = 2 revised states (one-to-one con assignments)
```

---

## COMPARACIÓN DETALLADA: APOLLO vs NUESTRO SISTEMA

| Aspecto | Apollo Original | Nuestro Sistema |
|---------|-----------------|-----------------|
| **Lenguaje** | C++ (Caffe → inferencia) | Python + PyTorch |
| **Framework** | CyberRT (ROS-like) | Standalone pipeline |
| **Entrada de ROIs** | HD-Map (consulta dinámica según pose) | YAML con projection boxes manuales |
| **Proyección 3D→2D** | Sí (calibración cámara + pose vehículo) | No (boxes ya en 2D) |
| **Semantic IDs** | Sí (agrupa semáforos relacionados) | No (cada projection es independiente) |
| **Selección Cámara** | Sí (telephoto 25mm vs wide-angle 6mm) | No (single camera) |
| **Crop Expansion** | 2.5× (crop_scale) | 2.5× (mismo) |
| **Tamaño Crop** | 270×270 (min_crop_size) | 270×270 (mismo) |
| **Normalización Det** | Medias: [102.98, 115.95, 122.77] | Medias: [102.98, 115.95, 122.77] (mismo) |
| **Normalización Rec** | Medias específicas por modelo | Medias: [69.06, 66.58, 66.56] |
| **Detector** | SSD + RPN + DFMB ROI Align | SSD + RPN + DFMB ROI Align (mismo) |
| **Recognizers** | 3 modelos (hori, vert, quad) | 3 modelos (hori, vert, quad) (mismo) |
| **NMS Threshold** | 0.6 (iou_thresh) | 0.6 (mismo) |
| **NMS Sorting** | Ascendente (procesa desde atrás) | Descendente (procesa desde adelante) |
| **Asignación** | Hungarian con Gaussian scoring | Hungarian con Gaussian scoring (mismo) |
| **Distance Weight** | 70% distancia, 30% confianza | 70% distancia, 30% confianza (mismo) |
| **Gaussian Sigma** | 100.0 | 100.0 (mismo) |
| **Prob2Color Threshold** | 0.5 (classify_threshold) | 0.5 (mismo) |
| **Tracking Scope** | Por semantic group (N semáforos) | Por projection ID (individual) |
| **Revise Time** | 1.5s (revise_time_second) | 1.5s (mismo) |
| **Blink Threshold** | 0.4s (blink_threshold_second) | 0.55s (ajustado) |
| **Hysteretic Threshold** | 1 frame (hysteretic_threshold_count) | 1 frame (mismo) |
| **Yellow Blink Safety** | Sí (YELLOW blink → RED) | Sí (mismo) |
| **Sequence Safety** | Sí (YELLOW after RED → RED) | Sí (mismo) |
| **Output Format** | Protobuf (`TrafficLightDetection`) | Tensors + dict |
| **Coordenadas Bug Fix** | Original correcto | Fixed (scale + offset) |

---

## REFERENCIAS DE CÓDIGO

### Apollo Original (C++)
- `traffic_light_region_proposal_component.cc:80-132` - Query HD-Map
- `tl_preprocessor.cc:105-182` - Proyección 3D→2D
- `detection.cc:329-356` - Restauración de coordenadas
- `detection.cc:381-410` - Sort + NMS
- `select.cc:38-97` - Hungarian assignment
- `recognition.cc:48-65` - Prob2Color
- `semantic_decision.cc:151-237` - Tracking temporal

### Nuestro Sistema (Python)
- `src/tlr/pipeline.py:150-201` - `load_pipeline()` entry point
- `src/tlr/pipeline.py:26-49` - `detect()` método
- `src/tlr/pipeline.py:51-93` - `recognize()` método
- `src/tlr/pipeline.py:95-146` - `forward()` método principal
- `src/tlr/detector.py:24-40` - `TLDetector` inference
- `src/tlr/recognizer.py:41-63` - `Recognizer` inference
- `src/tlr/selector.py:8-68` - `select_tls()` Hungarian
- `src/tlr/tracking.py:36-123` - `SemanticDecision` clase
- `src/tlr/tools/utils.py:215-236` - `crop()` función
- `src/tlr/tools/utils.py:238-243` - `preprocess4det()` función
- `src/tlr/tools/utils.py:245-256` - `preprocess4rec()` función
- `src/tlr/tools/utils.py:261-302` - `restore_boxes_to_full_image()` bug fix
- `src/tlr/tools/utils.py:82-158` - `nms()` implementación

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
├── bbox_reg_param.json                     # RPN box regression params
├── detection_output_ssd_param.json         # RPN NMS params
├── dfmb_psroi_pooling_param.json           # DFMB ROI Align params
├── rcnn_bbox_reg_param.json                # RCNN box regression params
└── rcnn_detection_output_ssd_param.json    # RCNN NMS params
```

### Projection Boxes (YAML)
```
frames_labeled/video_01/
├── projection_boxes.yaml   # [x1, y1, x2, y2, id] por frame
└── frames/
    ├── frame_0000.jpg
    ├── frame_0001.jpg
    └── ...
```

---

## FLUJO SIMPLIFICADO (DIAGRAMA DE BLOQUES)

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                        │
│ • Image (Tensor [H,W,3] BGR)                                │
│ • Projection Boxes (List [[x1,y1,x2,y2,id], ...])          │
│ • Frame Timestamp (float, opcional para tracking)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 1: PREPROCESAMIENTO                                   │
│ • Por cada projection: crop + expand 2.5× + resize 270×270  │
│ • Normalización: img - [102.98, 115.95, 122.77]            │
│                                                              │
│ INPUT:  M projection boxes                                  │
│ OUTPUT: M crops [270,270,3]                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 2: DETECCIÓN CNN                                      │
│ • TLDetector (SSD + RPN + DFMB)                             │
│ • Restaurar coords: scale → offset                          │
│ • Sort by score + NMS (IoU threshold 0.6)                   │
│                                                              │
│ INPUT:  M crops [270,270,3]                                 │
│ OUTPUT: N detections [N,9] después NMS                      │
│         [score, x1, y1, x2, y2, bg, vert, quad, hori]       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 3: FILTRADO + ASIGNACIÓN                              │
│ • Clasificar tipo: argmax(class_probs) → 0/1/2/3           │
│ • Filtrar válidos: tipo != 0 (background)                   │
│ • Hungarian: proj → det (Gaussian + confidence scoring)     │
│                                                              │
│ INPUT:  N detections [N,9]                                  │
│ OUTPUT: N' valid_dets [N',9] + K assignments [K,2]         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 4: RECONOCIMIENTO                                     │
│ • Por cada valid detection:                                 │
│   - Crop ROI + resize (96×32 / 64×64 / 32×96)              │
│   - Recognizer → [black, red, yellow, green]                │
│   - Prob2Color: max_prob > 0.5 ? use_max : force_black     │
│                                                              │
│ INPUT:  N' valid_dets [N',9]                                │
│ OUTPUT: N' recognitions [N',4] (one-hot)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ ETAPA 5: TRACKING TEMPORAL (OPCIONAL)                       │
│ • Por cada assignment (proj_id, det_idx):                   │
│   - Histéresis desde BLACK (threshold=1)                    │
│   - Yellow blink detection (< 0.55s → force RED)            │
│   - Sequence safety (yellow after red → keep RED)           │
│   - Revise time window (1.5s)                               │
│                                                              │
│ INPUT:  K assignments [K,2] + N' recognitions [N',4]       │
│ OUTPUT: dict {proj_id: (color, blink)}                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT                                                       │
│ • valid_detections: [N',9]                                  │
│ • recognitions: [N',4]                                      │
│ • assignments: [K,2]                                        │
│ • invalid_detections: [N-N',9]                              │
│ • revised: {proj_id: (color, blink)} o None                 │
└─────────────────────────────────────────────────────────────┘
```

---

## EJEMPLO COMPLETO DE EJECUCIÓN

### Input
```python
import torch
from tlr.pipeline import load_pipeline

# Cargar pipeline
pipeline = load_pipeline('cuda:0')

# Imagen (1920×1080 BGR)
image = torch.Tensor(...)  # [1080, 1920, 3]

# Projection boxes manuales
boxes = [
    [850, 300, 890, 380, 1],   # Semáforo izquierdo (ID=1)
    [1050, 280, 1090, 360, 2],  # Semáforo derecho (ID=2)
    [1200, 320, 1235, 390, 3],  # Semáforo lejano (ID=3)
]

# Timestamp (para tracking)
frame_ts = 10.25  # segundos
```

### Ejecución
```python
valid_dets, recs, assigns, invalid_dets, revised = \
    pipeline(image, boxes, frame_ts=frame_ts)
```

### Output
```python
# valid_dets: [2, 9]
tensor([
    [0.95, 852, 305, 888, 375, 0.01, 0.92, 0.05, 0.02],  # Det #0 (vertical)
    [0.88, 1052, 285, 1088, 355, 0.02, 0.05, 0.03, 0.90],  # Det #1 (horizontal)
])

# recs: [2, 4]
tensor([
    [0.0, 1.0, 0.0, 0.0],  # Det #0 → RED
    [0.0, 0.0, 0.0, 1.0],  # Det #1 → GREEN
])

# assigns: [2, 2]
tensor([
    [0, 0],  # proj_idx=0 (ID=1) → det_idx=0
    [1, 1],  # proj_idx=1 (ID=2) → det_idx=1
])

# invalid_dets: [1, 9]
tensor([
    [0.60, 1205, 328, 1230, 382, 0.85, 0.10, 0.03, 0.02],  # Background (de proj #2)
])

# revised: dict
{
    1: ('red', False),    # ID=1 → RED (sin blink)
    2: ('green', False),  # ID=2 → GREEN (sin blink)
    # ID=3 sin assignment (no detectado o score muy bajo)
}
```

### Interpretación
- **Projection 0 (ID=1):** Detectó semáforo vertical en ROJO
- **Projection 1 (ID=2):** Detectó semáforo horizontal en VERDE
- **Projection 2 (ID=3):** Detectó algo pero clasificado como background (falso positivo)

---

## CONCLUSIÓN

Este sistema implementa la arquitectura core de Apollo TLR en PyTorch, con las siguientes simplificaciones:

**Conservado:**
- ✅ Arquitectura CNN completa (detector + 3 recognizers)
- ✅ Pipeline completo (detect → assign → recognize → track)
- ✅ Hungarian algorithm para asignación óptima
- ✅ Tracking temporal con hysteresis y safety rules
- ✅ Crop expansion 2.5×, NMS threshold 0.6, etc.

**Simplificado:**
- ❌ No HD-Map (projection boxes manuales en 2D)
- ❌ No semantic grouping (tracking individual por projection ID)
- ❌ No selección de cámara (single camera)
- ❌ No proyección 3D→2D (boxes ya en píxeles)

**Beneficios:**
- Pipeline más simple y directo
- No requiere HD-Map ni calibración compleja
- Más fácil de debuggear (todo en Python/PyTorch)
- Ideal para datasets con projection boxes annotadas manualmente

**Limitaciones:**
- Requiere definición manual de projection boxes por video
- No se adapta automáticamente a nuevas posiciones de semáforos
- Sin multi-camera fusion
- Tracking menos robusto (sin semantic voting entre múltiples semáforos)

---

**Fin del documento.**
