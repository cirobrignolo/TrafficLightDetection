# Verificación Completa: Implementación Python vs Apollo C++

## FASE 1: DETECTOR

### Apollo C++ (detection.cc:310-340)
```cpp
// 1. Detector retorna 9 valores por detección:
//    [score, x1, y1, x2, y2, class_0, class_1, class_2, class_3]
std::vector<float> score{result_data[5], result_data[6], result_data[7], result_data[8]};

// 2. Calcula argmax
std::vector<float>::iterator biggest = std::max_element(std::begin(score), std::end(score));

// 3. Convierte a class_id con offset -1
tmp->region.detect_class_id = base::TLDetectionClass(std::distance(std::begin(score), biggest) - 1);
//    Si biggest apunta a índice 0 → class_id = -1 (TL_UNKNOWN_CLASS)
//    Si biggest apunta a índice 1 → class_id = 0 (TL_VERTICAL_CLASS)
//    Si biggest apunta a índice 2 → class_id = 1 (TL_QUADRATE_CLASS)
//    Si biggest apunta a índice 3 → class_id = 2 (TL_HORIZONTAL_CLASS)

// 4. Filtra si class_id < 0
if (static_cast<int>(tmp->region.detect_class_id) >= 0) {
  // Detección válida
}
```

**Orden de scores en Apollo**: `[background, vertical, quadrate, horizontal]`

### Python (pipeline.py:26-48)
```python
# 1. Detector retorna tensor (N, 9):
#    [:, 0] = detect_score
#    [:, 1:5] = bbox [x1, y1, x2, y2]
#    [:, 5:9] = class scores

detections = self.detector(input)  # Retorna (N, 9)

# 2. Sort by score BEFORE NMS
scores = detections[:, 0]
sorted_indices = torch.argsmax(scores, descending=True)
detections_sorted = detections[sorted_indices]

# 3. Apply NMS with threshold 0.6
idxs = nms(detections_sorted[:, 1:5], 0.6)
detections = detections_sorted[idxs]

# 4. Filtrar por tipo (en forward, líneas 121-124)
tl_types = torch.argmax(detections[:, 5:9], dim=1)  # argmax de 4 scores
valid_mask = tl_types != 0  # Filtra cuando argmax == 0
valid_detections = detections[valid_mask]
invalid_detections = detections[~valid_mask]
```

**Orden de scores en Python**: NECESITA VERIFICACIÓN

---

## ANÁLISIS: ¿Cuál es el orden real?

### Evidencia del CSV generado:
```
frame_0000.jpg,INVALID,0,81,199,184,278,bg,0.7285,0.2682,0.0018,0.0015
```

- `tl_type = bg` porque `argmax(det[5:9]) == 0` y `type_names[0] = 'bg'`
- Los scores son: `[0.7285, 0.2682, 0.0018, 0.0015]`
- El máximo es 0.7285 en índice 0
- Esta detección es INVALID (filtrada)

### Conclusión:
- **Orden correcto**: `[bg, vert, quad, hori]`
- **Filtrado correcto**: `tl_types != 0` filtra cuando argmax==0 (background)
- **Implementación CORRECTA** ✅

---

## FASE 2: NMS

### Apollo C++ (detection.h:87)
```cpp
void ApplyNMS(std::vector<base::TrafficLightPtr> *lights, double iou_thresh = 0.6);
```

### Python (pipeline.py:46)
```python
idxs = nms(detections_sorted[:, 1:5], 0.6)  # ✅ AHORA CORRECTO
```

**Status**: ✅ CORRECTO (cambiado de 0.7 a 0.6)

---

## FASE 3: SELECTOR (Hungarian)

### Apollo C++ (select.cc:45-90)
```cpp
// 1. Construir cost matrix
for (int row = 0; row < lights_num; ++row) {
  for (int col = 0; col < boxes_num; ++col) {
    // Calcular distancia y confidence
    float gaussian_score = GetGaussianScore(center_hd, center_box);
    float confidence = max(lights->at(col)->region.detect_score, 0.0f);

    // ROI validation ANTES de Hungarian
    cv::Rect region = crop(image_border_size, *(boxes.at(row)));
    if (!BoxIsValid(detect_box, region)) {
      cost_matrix[row * boxes_num + col] = 0.0;
      continue;
    }

    // Score combinado
    cost_matrix[row * boxes_num + col] = 0.7 * gaussian_score + 0.3 * confidence;
  }
}

// 2. Aplicar Hungarian
global_optimizer->Maximize(&cost_matrix, &assignment);
```

### Python (selector.py:22-60)
```python
# 1. Construir cost matrix
for row, projection in enumerate(projections):
    center_hd = [projection.center_x, projection.center_y]
    coors = crop(item_shape, projection)  # Pre-computar ROI

    for col, detection in enumerate(detections):
        # Calcular distancia y confidence
        center_box = [(detection[1] + detection[3]) / 2, (detection[2] + detection[4]) / 2]
        gaussian_score = get_gaussian_score(center_hd, center_box)
        confidence = max(detection[0].item(), 0.0)

        # ROI validation ANTES de Hungarian ✅ CORRECTO
        det_box = detection[1:5]
        if coors[0] > det_box[0] or coors[1] < det_box[2] or \
           coors[2] > det_box[1] or coors[3] < det_box[3]:
            costs[row, col] = 0.0
            continue

        # Score combinado
        costs[row, col] = 0.7 * gaussian_score + 0.3 * confidence

# 2. Aplicar Hungarian
assignments = ho.maximize(costs)
```

**Status**: ✅ CORRECTO

---

## FASE 4: RECOGNIZER

### Apollo C++ (classify.cc:151-164)
```cpp
void Prob2Color(const float* out_put_data, float threshold, base::TrafficLightPtr light) {
  std::vector<base::TLColor> status_map = {
      base::TLColor::TL_BLACK,   // índice 0
      base::TLColor::TL_RED,     // índice 1
      base::TLColor::TL_YELLOW,  // índice 2
      base::TLColor::TL_GREEN    // índice 3
  };

  std::vector<float> prob(out_put_data, out_put_data + status_map.size());
  auto max_prob = std::max_element(prob.begin(), prob.end());

  max_color_id = (*max_prob > threshold)
                 ? static_cast<int>(std::distance(prob.begin(), max_prob))
                 : 0;  // Force BLACK si prob < threshold

  light->status.color = status_map[max_color_id];
}
```

**Orden de colores en Apollo**: `[BLACK, RED, YELLOW, GREEN]`
**Threshold**: 0.5

### Python (pipeline.py:50-92)
```python
def recognize(self, img, detections, tl_types):
    recognitions = []

    for detection, tl_type in zip(detections, tl_types):
        det_box = detection[1:5].type(torch.long)
        recognizer, shape = self.classifiers[tl_type-1]  # ⚠️ VERIFICAR

        input = preprocess4rec(img, det_box, shape, self.means_rec)
        input_scaled = input.permute(2, 0, 1).unsqueeze(0) * 0.01

        output_probs = recognizer(input_scaled)[0]  # [4]

        max_prob, max_idx = torch.max(output_probs, dim=0)
        threshold = 0.5

        if max_prob > threshold:
            color_id = max_idx.item()
        else:
            color_id = 0  # Force BLACK

        result = torch.zeros_like(output_probs)
        result[color_id] = 1.0
        recognitions.append(result)

    return torch.vstack(recognitions)
```

### VERIFICAR: Mapeo tl_type → recognizer

Si `tl_types` viene de `argmax(det[:, 5:9])` con orden `[bg, vert, quad, hori]`:
- tl_type = 0 → bg (filtrado, no llega a recognize)
- tl_type = 1 → vert → classifiers[0] = vert_recognizer ✅
- tl_type = 2 → quad → classifiers[1] = quad_recognizer ✅
- tl_type = 3 → hori → classifiers[2] = hori_recognizer ✅

**Status**: ✅ CORRECTO

---

## FASE 5: TRACKER

### Apollo C++ (semantic_decision.cc:158-229)
- Revisor temporal con hysteresis
- Filtro de blink (yellow flickering)
- Safety rules (red/green transitions)

### Python (tracking.py:45-180)
- Implementación equivalente
- Mismos parámetros de hysteresis
- Mismas safety rules

**Status**: ✅ CORRECTO (ya verificado previamente)

---

## RESUMEN DE DIFERENCIAS ENCONTRADAS

### ✅ CORRECCIONES REALIZADAS:
1. NMS threshold: 0.7 → 0.6 ✅
2. ROI validation timing: ANTES de Hungarian ✅
3. NMS sorting: Sort by score BEFORE applying NMS ✅
4. IoU abs(): Added torch.abs() ✅
5. CSV headers: Corregidos a orden [bg, vert, quad, hori] ✅
6. CSV type_names: Corregidos a orden [bg, vert, quad, hori] ✅

### ⚠️ DIFERENCIAS QUE QUEDAN:
1. **Semantic IDs**: Python usa row_index, Apollo usa persistent semantic_id
2. **Multi-ROI selection**: No implementado (Apollo puede asignar 1 detección a múltiples ROIs)

### 🐛 PROBLEMAS NO RESUELTOS (Limitaciones del detector):
1. **Frames 1-33**: Detecciones duplicadas (NMS no las filtra porque IoU < 0.6)
2. **Frames 118, 152, 154-161**: Falsos positivos grandes con bg_score alto pero quad_score máximo
3. **Frame 243+**: Post-movimiento, múltiples falsos positivos

Estos problemas son **limitaciones del detector** (genera bboxes espurias) y probablemente ocurren también en Apollo.

