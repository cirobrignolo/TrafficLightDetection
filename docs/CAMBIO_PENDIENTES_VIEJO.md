# 🔧 Informe de Gaps y Roadmap: Sistema Actual → Apollo Original

**Objetivo**: Documentar todas las diferencias entre el sistema actual y Apollo, con plan de implementación detallado para cerrar los gaps.

**Audiencia**: Personal (documentación + futuro trabajo de implementación)

---

## 📋 Índice

1. **Resumen Ejecutivo de Gaps**
2. **Gap #1: Assignment Algorithm (Hungarian → Selection)**
3. **Gap #2: Múltiples Detecciones por ROI**
4. **Gap #3: Projection Boxes Dinámicas**
5. **Gap #4: ID Management (Row Index → Semantic ID)**
6. **Gap #5: Multi-Camera Fusion**
7. **Gap #6: Dependencia Espacial del Recognizer**
8. **Roadmap de Implementación Priorizado**
9. **Plan de Testing y Validación**

---

## 1. 📊 Resumen Ejecutivo de Gaps

### Tabla de Gaps Identificados

| # | Gap | Impacto | Complejidad | Prioridad |
| --- | --- | --- | --- | --- |
| **1** | Hungarian → Selection Algorithm | 🔴 Alto | 🟡 Media | **P0** |
| **2** | Múltiples Detections/ROI no manejadas | 🔴 Alto | 🟡 Media | **P0** |
| **3** | Projection Boxes Estáticas → Dinámicas | 🔴 Alto | 🔴 Alta | **P1** |
| **4** | Row Index → Semantic ID | 🟠 Medio | 🟢 Baja | **P2** |
| **5** | Single Camera → Multi-Camera | 🟡 Bajo | 🔴 Alta | **P3** |
| **6** | Dependencia Espacial Recognizer | 🔴 Alto | 🔴 Alta | **P1** |

### Impacto por Categoría

```
Funcionalidad Core (Detección/Assignment):
├── Hungarian vs Selection .............. 🔴 CRÍTICO
├── Múltiples detections ................ 🔴 CRÍTICO
└── Projection boxes dinámicas .......... 🔴 CRÍTICO

Tracking/IDs:
├── Row index vs Semantic ID ............ 🟠 IMPORTANTE
└── Cross-history transfer .............. 🔴 CRÍTICO (causado por gaps anteriores)

Performance/Robustez:
├── Dependencia espacial ................ �� CRÍTICO
└── Multi-camera ....................... 🟡 NICE-TO-HAVE

```

---

## 2. 🎯 Gap #1: Assignment Algorithm (Hungarian → Selection)

### 2.1 Estado Actual vs Apollo

### **Sistema Actual: Hungarian Algorithm**

```python
# src/tlr/selector.py
def select_tls(ho, detections, projections, item_shape):
    costs = torch.zeros([len(projections), len(detections)])

    # Matriz de costos M×N
    for row, projection in enumerate(projections):
        for col, detection in enumerate(detections):
            distance_score = calc_2d_gaussian_score(...)
            detection_score = torch.max(detection[5:])
            costs[row, col] = 0.3 * detection_score + 0.7 * distance_score

    # Assignment óptimo 1:1
    assignments = ho.maximize(costs)  # [[proj_idx, det_idx], ...]

    return assignments

```

**Características**:

- ✅ Assignment óptimo global (maximiza suma de scores)
- ✅ Garantiza no-conflictos (1 detection → max 1 projection)
- ❌ Solo 2 métricas (distance + confidence)
- ❌ **Constraint 1:1 estricto** (si 2 detections para 1 projection, solo 1 se asigna)
- ❌ No fusiona múltiples detections del mismo semáforo
- ❌ Complejidad O(N³) (costoso para muchas detections)

---

### **Apollo Original: Score-based Selection**

```cpp
// Apollo's selection algorithm (pseudo-código basado en documentación)
struct SelectionCriteria {
    float detection_score;        // 0.4 weight
    float spatial_proximity;      // 0.3 weight
    float shape_consistency;      // 0.2 weight
    float temporal_consistency;   // 0.1 weight
};

for (auto &hd_light : hd_map_lights) {  // Para cada semáforo HD-Map
    vector<Detection> candidates;

    // Encontrar todas las detecciones cercanas
    for (auto &detection : all_detections) {
        if (distance(hd_light.projection, detection.bbox) < threshold) {
            candidates.push_back(detection);
        }
    }

    // Calcular score para cada candidato
    float best_score = -1;
    Detection* best_detection = nullptr;

    for (auto &candidate : candidates) {
        float score = 0.4 * candidate.confidence +
                     0.3 * spatial_score(hd_light, candidate) +
                     0.2 * shape_score(hd_light, candidate) +
                     0.1 * temporal_score(hd_light, candidate);

        if (score > best_score) {
            best_score = score;
            best_detection = &candidate;
        }
    }

    // Asignar mejor detection a este HD-Map light
    if (best_detection != nullptr) {
        assignments[hd_light.id] = best_detection;
    }
}

```

**Características**:

- ✅ **N detections → 1 selección** por semáforo HD-Map
- ✅ **4 métricas** de evaluación (más robusto)
- ✅ Fusiona múltiples detections del mismo objeto (elige la mejor)
- ✅ **Temporal consistency** incluida en selection
- ✅ Complejidad O(N) por semáforo (más eficiente)
- ✅ **Permite detections sin asignar** (no fuerza 1:1)

---

### 2.2 Diferencias Críticas

| Aspecto | Hungarian (Actual) | Selection (Apollo) |
| --- | --- | --- |
| **Objetivo** | Maximizar suma global de scores | Encontrar mejor detection por semáforo |
| **Constraint** | 1:1 estricto | N:1 permitido (múltiples det → 1 semáforo) |
| **Criterios** | 2 (distance + confidence) | 4 (+ shape + temporal) |
| **Múltiples det mismo objeto** | Solo asigna 1, resto → ID -1 | Fusiona (selecciona mejor) |
| **Temporal info** | ❌ No considerada | ✅ Incluida (0.1 weight) |
| **Complejidad** | O(N³) | O(N×M) ≈ O(N) |
| **Robustez** | Baja (solo spatial) | Alta (multi-criterio) |

---

### 2.3 ¿Por Qué Apollo Usa Selection en vez de Hungarian?

### **Razón 1: Múltiples Detections del Mismo Semáforo**

```
Escenario real:
- 1 semáforo físico genera 2-3 bboxes levemente diferentes
- Detector SSD puede producir múltiples proposals para mismo objeto

Hungarian: Solo asigna 1, resto quedan ID -1 (pérdida de información)
Selection: Selecciona la mejor, ignora duplicados (fusión implícita)

```

### **Razón 2: No Requiere Assignment Perfecto**

```
Hungarian: Necesita asignar TODAS las detections (o dejarlas ID -1)
           - Problema si hay N detections pero M<N projections

Selection: Cada HD-Map light busca su mejor detection independientemente
          - No importa si sobran detections (pueden ser false positives)

```

### **Razón 3: Temporal Consistency en Assignment**

```
Apollo: Usa historial para validar si una detection es consistente temporalmente
        - Si semáforo era verde y detection dice rojo → score bajo
        - Si semáforo era verde y detection dice verde → score alto

Sistema actual: Temporal consistency solo DESPUÉS de assignment (en tracker)

```

### **Razón 4: Shape Validation**

```
Apollo: Valida geometría (aspect ratio, tamaño esperado vs detectado)
        - Detections con forma incorrecta → score bajo

Sistema actual: No valida geometría en assignment

```

---

### 2.4 Impacto del Gap

### **Problemas Causados por Hungarian**

**Problema 1: ID -1 Excesivo**

```python
# Escenario: 1 semáforo genera 2 detections
projections = [proj_0]  # 1 projection
detections = [det_A, det_B]  # 2 detections del mismo semáforo

# Hungarian: Solo puede asignar 1:1
assignments = [[0, 0]]  # Asigna det_A a proj_0
# det_B queda ID -1 (perdido)

# Selection Apollo: Evaluaría ambas, seleccionaría la mejor
best = max(det_A, det_B, key=lambda d: score(d))
assignments = [best]  # Solo 1 resultado, pero eligió el mejor

```

**Frecuencia observada**: 5-10% de detections válidas → ID -1

---

**Problema 2: No Considera Temporal Consistency**

```python
# Frame 100: Semáforo proj_0 está en GREEN (history[0].color = 'green')
# Frame 101: 2 detections cerca de proj_0
det_A: clasificado como GREEN (consistente con history)
det_B: clasificado como RED (inconsistente - posible error)

# Hungarian: Solo usa distance + confidence
# Si det_B está más cerca o tiene mayor confidence → se asigna (INCORRECTO)

# Selection Apollo: Usaría temporal_score
temporal_score(det_A) = high (GREEN → GREEN transition OK)
temporal_score(det_B) = low (GREEN → RED sin YELLOW = invalid)
# Resultado: det_A seleccionado (CORRECTO)

```

---

**Problema 3: Performance O(N³)**

```python
# Con muchas detections:
10 projections × 20 detections = Hungarian O(30³) ≈ 27,000 ops
                                 Selection O(10×20) = 200 ops

# Impacto real:
Hungarian: 5-20ms para assignment
Selection: <1ms para assignment

```

---

### 2.5 Implementación Propuesta: Apollo Selection Algorithm

### **Paso 1: Definir Criterios de Scoring**

```python
# src/tlr/apollo_selector.py (NUEVO ARCHIVO)

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

class SelectionCriteria:
    """Apollo-style selection criteria"""

    WEIGHT_DETECTION = 0.4
    WEIGHT_SPATIAL = 0.3
    WEIGHT_SHAPE = 0.2
    WEIGHT_TEMPORAL = 0.1

    @staticmethod
    def detection_score(detection: torch.Tensor) -> float:
        """
        Score basado en confidence del detector
        detection[0] = confidence score
        """
        return float(detection[0])

    @staticmethod
    def spatial_score(projection, detection: torch.Tensor,
                     sigma_x: float = 100, sigma_y: float = 100) -> float:
        """
        Score basado en distancia 2D gaussiana
        Mismo cálculo que Hungarian pero como score independiente
        """
        proj_center_x = (projection.xl + projection.xr) / 2
        proj_center_y = (projection.yt + projection.yb) / 2

        det_center_x = (detection[1] + detection[3]) / 2
        det_center_y = (detection[2] + detection[4]) / 2

        dx = proj_center_x - det_center_x
        dy = proj_center_y - det_center_y

        score = np.exp(-0.5 * ((dx/sigma_x)**2 + (dy/sigma_y)**2))
        return score

    @staticmethod
    def shape_score(projection, detection: torch.Tensor) -> float:
        """
        Score basado en consistencia geométrica
        Valida aspect ratio y tamaño esperado
        """
        # Tamaño esperado de projection
        proj_w = projection.xr - projection.xl
        proj_h = projection.yb - projection.yt
        proj_aspect = proj_w / proj_h if proj_h > 0 else 1.0

        # Tamaño de detection
        det_w = detection[3] - detection[1]
        det_h = detection[4] - detection[2]
        det_aspect = det_w / det_h if det_h > 0 else 1.0

        # Score de aspect ratio (penaliza diferencias grandes)
        aspect_diff = abs(proj_aspect - det_aspect) / max(proj_aspect, det_aspect)
        aspect_score = 1.0 - min(aspect_diff, 1.0)

        # Score de tamaño (penaliza detections muy grandes/pequeñas)
        size_ratio = (det_w * det_h) / (proj_w * proj_h)
        size_score = 1.0 if 0.5 <= size_ratio <= 2.0 else 0.5

        return (aspect_score + size_score) / 2.0

    @staticmethod
    def temporal_score(projection_id: int, detection: torch.Tensor,
                      history: Dict, recognitions: List[List[float]],
                      det_idx: int) -> float:
        """
        Score basado en consistencia temporal
        Valida si el cambio de estado es válido
        """
        if projection_id not in history:
            return 0.5  # Neutral si no hay historial

        prev_color = history[projection_id].color

        # Obtener color actual de esta detection
        if det_idx < len(recognitions):
            curr_cls = int(max(range(len(recognitions[det_idx])),
                             key=lambda i: recognitions[det_idx][i]))
            curr_color = ["black", "red", "yellow", "green"][curr_cls]
        else:
            return 0.0

        # Validar transiciones
        valid_transitions = {
            "black": ["red", "yellow", "green", "black"],  # Desde unknown, todo OK
            "red": ["red", "yellow", "green"],             # Red puede ir a yellow/green
            "yellow": ["yellow", "red"],                   # Yellow solo a red (safety)
            "green": ["green", "yellow"]                   # Green a yellow (normal)
        }

        if curr_color in valid_transitions.get(prev_color, []):
            return 1.0  # Transición válida
        else:
            return 0.2  # Transición inválida (pero no imposible)

```

---

### **Paso 2: Implementar Selection Algorithm**

```python
# src/tlr/apollo_selector.py (continuación)

class ApolloSelector:
    """
    Apollo-style selection algorithm
    Para cada projection, selecciona la mejor detection basándose en múltiples criterios
    """

    def __init__(self):
        self.criteria = SelectionCriteria()

    def select(self,
               detections: torch.Tensor,  # (N, 9) tensor
               projections: List,          # List of ProjectionROI
               history: Dict,              # Tracking history
               recognitions: List[List[float]],  # Recognition results
               distance_threshold: float = 200.0  # Max distance to consider
               ) -> List[Tuple[int, int]]:  # [(proj_id, det_idx), ...]
        """
        Selecciona la mejor detection para cada projection

        Args:
            detections: Todas las detecciones (N×9)
            projections: Lista de projection boxes
            history: Historial de tracking
            recognitions: Resultados de reconocimiento
            distance_threshold: Distancia máxima para considerar candidatos

        Returns:
            Lista de assignments [(proj_id, det_idx), ...]
        """
        assignments = []

        for proj_id, projection in enumerate(projections):
            # Paso 1: Filtrar candidates por distancia
            candidates = []

            for det_idx, detection in enumerate(detections):
                spatial = self.criteria.spatial_score(projection, detection)

                # Convertir score a distancia aproximada para threshold
                # (spatial_score alto = distancia baja)
                if spatial > 0.1:  # Score mínimo (equivale a ~200px)
                    candidates.append((det_idx, detection))

            if len(candidates) == 0:
                continue  # No hay candidates para esta projection

            # Paso 2: Calcular score total para cada candidate
            best_score = -1
            best_det_idx = None

            for det_idx, detection in candidates:
                # 4 componentes del score
                det_score = self.criteria.detection_score(detection)
                spatial_score = self.criteria.spatial_score(projection, detection)
                shape_score = self.criteria.shape_score(projection, detection)
                temporal_score = self.criteria.temporal_score(
                    proj_id, detection, history, recognitions, det_idx
                )

                # Score total ponderado
                total_score = (
                    SelectionCriteria.WEIGHT_DETECTION * det_score +
                    SelectionCriteria.WEIGHT_SPATIAL * spatial_score +
                    SelectionCriteria.WEIGHT_SHAPE * shape_score +
                    SelectionCriteria.WEIGHT_TEMPORAL * temporal_score
                )

                if total_score > best_score:
                    best_score = total_score
                    best_det_idx = det_idx

            # Paso 3: Asignar mejor detection
            if best_det_idx is not None:
                assignments.append((proj_id, best_det_idx))

        return assignments

```

---

### **Paso 3: Integrar en Pipeline**

```python
# src/tlr/pipeline.py (MODIFICAR)

from tlr.apollo_selector import ApolloSelector  # NUEVO

class Pipeline(nn.Module):
    def __init__(self, detector, classifiers, ho, means_det, means_rec,
                 device=None, tracker=None, use_apollo_selector=True):  # NUEVO FLAG
        super().__init__()
        self.detector = detector
        self.classifiers = classifiers
        self.means_det = means_det
        self.means_rec = means_rec
        self.ho = ho  # Mantener para compatibilidad
        self.device = device
        self.tracker = tracker

        # NUEVO: Apollo selector
        self.use_apollo_selector = use_apollo_selector
        if use_apollo_selector:
            self.apollo_selector = ApolloSelector()

    def forward(self, img, boxes, frame_ts=None):
        # ... (detección igual) ...

        detections = self.detect(img, boxes)
        tl_types = torch.argmax(detections[:, 5:], dim=1)
        valid_mask = tl_types != 0
        valid_detections = detections[valid_mask]
        invalid_detections = detections[~valid_mask]

        # MODIFICAR: Selección de assignments
        if self.use_apollo_selector:
            # NUEVO: Apollo-style selection
            # Primero necesitamos recognitions para temporal scoring
            temp_recognitions = []
            if len(valid_detections) > 0:
                temp_recognitions = self.recognize(img, valid_detections,
                                                   tl_types[valid_mask]).cpu().tolist()

            # Obtener historial del tracker
            history = self.tracker.semantic.history if self.tracker else {}

            # Selection algorithm
            assignments = self.apollo_selector.select(
                valid_detections,
                boxes2projections(boxes),
                history,
                temp_recognitions
            )
            assignments = torch.tensor(assignments, device=self.device)
        else:
            # ORIGINAL: Hungarian algorithm
            assignments = select_tls(self.ho, valid_detections,
                                    boxes2projections(boxes), img.shape)

        # ... (resto del pipeline igual) ...

```

---

### 2.6 Plan de Testing

### **Test 1: Múltiples Detections Mismo Semáforo**

```python
# test_apollo_selector.py

def test_multiple_detections_same_light():
    """
    Test: 1 semáforo genera 2 detections
    Esperado: Selection elige la mejor (mayor confidence)
    """
    projections = [
        ProjectionROI(400, 150, 60, 70)  # 1 projection
    ]

    detections = torch.tensor([
        [0.85, 410, 160, 450, 210, 0.01, 0.95, 0.03, 0.01],  # Det A: conf=0.85
        [0.92, 408, 162, 448, 212, 0.01, 0.96, 0.02, 0.01],  # Det B: conf=0.92 (mejor)
    ])

    selector = ApolloSelector()
    assignments = selector.select(detections, projections, {}, [])

    # Verificar que selecciona det_idx=1 (mayor confidence)
    assert len(assignments) == 1
    assert assignments[0] == (0, 1)  # proj_0 → det_1

```

---

### **Test 2: Temporal Consistency**

```python
def test_temporal_consistency():
    """
    Test: Detection inconsistente temporalmente recibe score bajo
    """
    projections = [ProjectionROI(400, 150, 60, 70)]

    # History: Semáforo estaba en GREEN
    history = {
        0: SemanticTable(0, 1.0, 'green')
    }

    detections = torch.tensor([
        [0.90, 410, 160, 450, 210, 0.01, 0.95, 0.03, 0.01],  # Det A
        [0.95, 412, 162, 452, 212, 0.01, 0.94, 0.04, 0.01],  # Det B (mayor conf)
    ])

    recognitions = [
        [0, 0, 1, 0],  # Det A: YELLOW (transición válida GREEN→YELLOW)
        [0, 1, 0, 0],  # Det B: RED (transición INVÁLIDA GREEN→RED)
    ]

    selector = ApolloSelector()
    assignments = selector.select(detections, projections, history, recognitions)

    # Verificar que selecciona Det A (consistente) a pesar de menor confidence
    assert assignments[0] == (0, 0)  # Temporal score compensa

```

---

### **Test 3: Shape Validation**

```python
def test_shape_validation():
    """
    Test: Detection con geometría incorrecta recibe score bajo
    """
    # Projection: aspect ratio vertical (60×70 ≈ 0.86)
    projections = [ProjectionROI(400, 150, 60, 70)]

    detections = torch.tensor([
        [0.90, 410, 160, 450, 210, 0.01, 0.95, 0.03, 0.01],  # Det A: ~1:1.25 (OK)
        [0.92, 410, 160, 480, 180, 0.01, 0.96, 0.02, 0.01],  # Det B: ~3.5:1 (horizontal - MAL)
    ])

    selector = ApolloSelector()
    assignments = selector.select(detections, projections, {}, [])

    # Verificar que selecciona Det A (aspect ratio correcto)
    assert assignments[0] == (0, 0)

```

---

### 2.7 Migración Gradual

### **Fase 1: Implementación Paralela (Semana 1)**

```python
# Correr ambos algoritmos, comparar resultados
assignments_hungarian = select_tls(ho, detections, projections, shape)
assignments_apollo = apollo_selector.select(detections, projections, history, recognitions)

# Log diferencias
if not torch.equal(assignments_hungarian, torch.tensor(assignments_apollo)):
    log_difference(assignments_hungarian, assignments_apollo)

```

### **Fase 2: A/B Testing (Semana 2-3)**

```python
# Alternar entre algoritmos en diferentes frames
if frame_num % 2 == 0:
    assignments = apollo_selector.select(...)
else:
    assignments = select_tls(ho, ...)

# Comparar métricas de performance

```

### **Fase 3: Deployment Completo (Semana 4)**

```python
# Usar Apollo selector por defecto
pipeline = load_pipeline(device, use_apollo_selector=True)

```

---

## 3. 🔄 Gap #2: Múltiples Detecciones por ROI

### 3.1 Problema Actual

### **Comportamiento Observado**

```python
# Frame con 1 semáforo físico
projections = [
    [400, 150, 460, 220, 0]  # 1 projection para semáforo
]

# Detector genera múltiples bboxes
detections_from_detector = [
    [0.95, 410, 160, 450, 210, 0.01, 0.95, 0.03, 0.01],  # Det A: ligeramente arriba
    [0.92, 408, 162, 448, 212, 0.01, 0.96, 0.02, 0.01],  # Det B: ligeramente abajo
    [0.88, 412, 159, 452, 211, 0.01, 0.94, 0.04, 0.01],  # Det C: ligeramente a la derecha
]

# PASO 1: NMS global (threshold=0.7)
# IoU entre Det A y Det B = 0.85 (alta superposición)
# IoU entre Det A y Det C = 0.82
# → NMS elimina Det B y Det C, mantiene Det A

detections_after_nms = [
    [0.95, 410, 160, 450, 210, 0.01, 0.95, 0.03, 0.01]  # Solo Det A
]

# PASO 2: Hungarian assignment
assignments = [[0, 0]]  # proj_0 → det_0

# RESULTADO: ✅ Funciona en este caso (solo 1 detection sobrevive)

```

**Pero...**

```python
# Frame con 2 semáforos físicos DISTINTOS en 1 ROI grande
projections = [
    [300, 100, 600, 400, 0]  # 1 ROI grande (crop_scale=2.5)
]

# Detector encuentra ambos semáforos
detections_from_detector = [
    [0.95, 320, 150, 360, 200, 0.01, 0.95, 0.03, 0.01],  # Semáforo izq
    [0.92, 520, 150, 560, 200, 0.01, 0.96, 0.02, 0.01],  # Semáforo der
]

# PASO 1: NMS global (threshold=0.7)
# IoU entre ambos = 0.0 (no se superponen)
# → NMS mantiene ambos

detections_after_nms = [
    [0.95, 320, 150, 360, 200, ...],  # Det 0
    [0.92, 520, 150, 560, 200, ...]   # Det 1
]

# PASO 2: Hungarian assignment
# ⚠️ PROBLEMA: Solo 1 projection pero 2 detections
# Hungarian fuerza 1:1 → solo asigna 1

assignments = [[0, 0]]  # proj_0 → det_0
# Det 1 queda ID -1 ❌ (semáforo real perdido)

```

---

### 3.2 Diseño de Apollo: Multi-Detection Handling

### **Filosofía de Apollo**

```cpp
// Apollo's multi-detection design principle:
// "Better to detect too many than to miss real traffic lights"

// 1. ROI expansion creates large search areas
float crop_scale = 2.5;  // ROI puede ser 2.5× más grande que projection

// 2. Detector puede encontrar múltiples lights en 1 ROI
vector<Detection> detections_in_roi = detector.Infer(roi);
// detections_in_roi.size() puede ser 0, 1, 2, 3+...

// 3. Selection NO impone límite 1:1
for (auto &hd_light : hd_map_lights) {
    // Cada HD-Map light busca su mejor detection independientemente
    // Múltiples HD-Map lights pueden estar en el mismo ROI
    best_detection = SelectBestDetection(hd_light, detections_in_roi);
}

```

---

### 3.3 ¿Cuándo Ocurre Multi-Detection en 1 ROI?

### **Caso 1: Múltiples Proposals del Mismo Semáforo**

```
Detector SSD genera múltiples bounding boxes para mismo objeto:
    ┌────────┐
    │ ┌────┐ │  ← Det A (confidence 0.95)
    │ │    │ │
    └─┤    ├─┘  ← Det B (confidence 0.92)
      └────┘

NMS elimina duplicados (IoU > 0.7)
Resultado: 1 detection final ✅

```

**Manejo actual**: ✅ Funciona correctamente (NMS hace su trabajo)

---

### **Caso 2: Semáforos Cercanos (Mismo Poste)**

```
ROI grande contiene 2 semáforos físicos distintos:

    ┌─────────────────────┐
    │  ┌──┐          ┌──┐ │
    │  │🔴│          │🟢│ │  ← 2 semáforos reales
    │  └──┘          └──┘ │
    └─────────────────────┘
     ↑                    ↑
   Det A              Det B

NMS NO elimina (IoU ≈ 0)
Resultado: 2 detections válidas

Apollo: Asigna ambas a diferentes HD-Map lights ✅
Sistema actual: Solo asigna 1, otra queda ID -1 ❌

```

**Manejo actual**: ❌ Problema (Hungarian 1:1)

---

### **Caso 3: Intersección Compleja**

```
ROI muy grande en intersección:

    ┌──────────────────────────┐
    │  ┌──┐  ┌──┐       ┌──┐   │
    │  │🔴│  │🟡│       │🟢│   │  ← 3+ semáforos
    │  └──┘  └──┘       └──┘   │
    └──────────────────────────┘
     ↑      ↑           ↑
   TL1    TL2         TL3

Apollo: 3 HD-Map lights, cada uno busca su mejor detection
Sistema actual: 1 projection, solo 1 assignment

```

**Manejo actual**: ❌ Problema crítico

---

### 3.4 Análisis del Gap

| Escenario | Detections | Projections | Apollo | Sistema Actual |
| --- | --- | --- | --- | --- |
| **Duplicados mismo objeto** | 2-3 | 1 | NMS fusiona → 1 | NMS fusiona → 1 ✅ |
| **2 semáforos cercanos** | 2 | 1 | Requiere 2 HD-Map lights | Solo asigna 1 ❌ |
| **Intersección compleja** | 5+ | 1 | Requiere 5+ HD-Map lights | Solo asigna 1 ❌ |

**Root cause**: Sistema actual asume **1 Projection = 1 Semáforo esperado**

---

### 3.5 Solución Propuesta

### **Opción A: Projection Boxes Específicas (Quick Fix)**

```python
# En vez de 1 ROI grande, definir múltiples projections específicas

# ❌ Actual (1 ROI grande):
projections = [
    [300, 100, 600, 400, 0]  # Cubre ambos semáforos
]

# ✅ Propuesto (2 ROIs específicas):
projections = [
    [320, 150, 360, 200, 0],  # Semáforo izquierdo
    [520, 150, 560, 200, 1]   # Semáforo derecho
]

```

**Ventajas**:

- ✅ No requiere cambios en código
- ✅ Funciona con Hungarian 1:1
- ✅ Fácil de implementar

**Desventajas**:

- ❌ Requiere annotación manual muy precisa
- ❌ No escala para intersecciones complejas
- ❌ No maneja semáforos inesperados (fuera de projections)

---

### **Opción B: Detección Iterativa dentro de ROI (Apollo-like)**

```python
# src/tlr/pipeline.py (NUEVA IMPLEMENTACIÓN)

def detect_multi(self, image, boxes):
    """
    Detección multi-semáforo dentro de cada ROI
    Permite múltiples detections por projection
    """
    all_detections = []
    detection_to_projection = []  # Mapeo det_idx → proj_id

    for proj_id, box in enumerate(boxes):
        projection = box2projection(box)

        # Crop y resize ROI
        input = preprocess4det(image, projection, self.means_det)

        # Detector encuentra N semáforos en esta ROI
        bboxes = self.detector(input.unsqueeze(0).permute(0, 3, 1, 2))

        # Restaurar coordenadas
        restored = restore_boxes_to_full_image(image, [bboxes], [projection])[0]

        # Agregar todas las detections de esta ROI
        for det in restored:
            all_detections.append(det)
            detection_to_projection.append(proj_id)  # Recordar de qué ROI vino

    all_detections = torch.vstack(all_detections) if all_detections else torch.empty((0,9))

    # NMS global (elimina duplicados entre ROIs)
    idxs = nms(all_detections[:, 1:5], 0.7)
    final_detections = all_detections[idxs]

    # Mantener mapeo después de NMS
    final_projection_map = [detection_to_projection[i] for i in idxs]

    return final_detections, final_projection_map

```

**Uso con Selection Algorithm**:

```python
def forward(self, img, boxes, frame_ts=None):
    # Detección multi
    detections, det_to_proj_map = self.detect_multi(img, boxes)

    # Filtrado
    tl_types = torch.argmax(detections[:, 5:], dim=1)
    valid_mask = tl_types != 0
    valid_detections = detections[valid_mask]

    # Reconocimiento
    recognitions = self.recognize(img, valid_detections, tl_types[valid_mask])

    # Selection (ya implementado en Gap #1)
    assignments = self.apollo_selector.select(
        valid_detections,
        boxes2projections(boxes),
        history,
        recognitions.cpu().tolist()
    )

    # ⚠️ IMPORTANTE: Verificar que detection pertenece a ROI correcto
    validated_assignments = []
    for proj_id, det_idx in assignments:
        # Solo asignar si detection vino de esta ROI (o ROI cercana)
        original_proj = det_to_proj_map[det_idx]
        if original_proj == proj_id or spatial_distance_ok(original_proj, proj_id):
            validated_assignments.append((proj_id, det_idx))

    return validated_assignments

```

**Ventajas**:

- ✅ Maneja múltiples semáforos por ROI
- ✅ Compatible con Selection Algorithm
- ✅ Más cercano a diseño Apollo

**Desventajas**:

- ⚠️ Requiere lógica de validación adicional
- ⚠️ Más complejo de debuggear

---

### **Opción C: HD-Map Integration (Solución Completa Apollo)**

```python
# src/tlr/hdmap_projections.py (NUEVO MÓDULO)

class HDMapProjector:
    """
    Proyecta semáforos del HD-Map a coordenadas 2D de imagen
    Replica funcionalidad Apollo de projection dinámica
    """

    def __init__(self, hdmap_file: str, camera_calib: dict):
        self.hdmap = self.load_hdmap(hdmap_file)
        self.camera_calib = camera_calib

    def load_hdmap(self, hdmap_file):
        """
        Carga HD-Map con coordenadas 3D de semáforos

        Formato esperado (JSON):
        {
            "traffic_lights": [
                {
                    "id": "TL001",
                    "position_3d": [x, y, z],  # Coordenadas mundo
                    "orientation": "vertical",
                    "expected_states": ["red", "yellow", "green"]
                },
                ...
            ]
        }
        """
        import json
        with open(hdmap_file) as f:
            return json.load(f)

    def project_lights(self, vehicle_pose: dict) -> List[dict]:
        """
        Proyecta semáforos 3D a 2D según pose del vehículo

        Args:
            vehicle_pose: {
                'position': [x, y, z],
                'orientation': [roll, pitch, yaw],
                'timestamp': float
            }

        Returns:
            Lista de projections 2D:
            [
                {
                    'semantic_id': 'TL001',
                    'bbox_2d': [x1, y1, x2, y2],
                    'distance': float,
                    'orientation': str
                },
                ...
            ]
        """
        projections = []

        for tl in self.hdmap['traffic_lights']:
            # Calcular transformación 3D→2D
            pos_3d = np.array(tl['position_3d'])

            # Transform: World → Vehicle → Camera
            pos_vehicle = self.world_to_vehicle(pos_3d, vehicle_pose)
            pos_camera = self.vehicle_to_camera(pos_vehicle)

            # Proyección perspectiva
            u, v = self.camera_to_pixel(pos_camera, self.camera_calib)

            # Estimar tamaño en imagen (basado en distancia)
            distance = np.linalg.norm(pos_camera)
            estimated_size = self.estimate_size(distance, tl['orientation'])

            # Crear bbox 2D
            x1 = int(u - estimated_size[0] / 2)
            y1 = int(v - estimated_size[1] / 2)
            x2 = int(u + estimated_size[0] / 2)
            y2 = int(v + estimated_size[1] / 2)

            projections.append({
                'semantic_id': tl['id'],
                'bbox_2d': [x1, y1, x2, y2],
                'distance': distance,
                'orientation': tl['orientation']
            })

        return projections

```

**Integración en Pipeline**:

```python
# src/tlr/pipeline.py (con HD-Map)

class Pipeline(nn.Module):
    def __init__(self, ..., hdmap_projector=None):
        # ...
        self.hdmap_projector = hdmap_projector

    def forward(self, img, vehicle_pose, frame_ts=None):
        # PASO 1: Proyectar semáforos del HD-Map
        if self.hdmap_projector:
            hdmap_projections = self.hdmap_projector.project_lights(vehicle_pose)
            boxes = [[p['bbox_2d'][0], p['bbox_2d'][1],
                     p['bbox_2d'][2], p['bbox_2d'][3],
                     p['semantic_id']] for p in hdmap_projections]
        else:
            # Fallback a boxes estáticas
            boxes = load_static_boxes()

        # PASO 2-N: Pipeline normal
        detections = self.detect(img, boxes)
        # ...

```

**Ventajas**:

- ✅ Solución completa Apollo-style
- ✅ Projection boxes dinámicas (siguen semáforos físicos)
- ✅ IDs semánticos persistentes
- ✅ Elimina cross-history transfer
- ✅ Escalable a cualquier escenario

**Desventajas**:

- ❌ Requiere HD-Map con coordenadas 3D
- ❌ Requiere pose del vehículo (GPS + IMU)
- ❌ Requiere calibración de cámara precisa
- ❌ Complejidad alta de implementación

---

### 3.6 Recomendación de Implementación

### **Roadmap Sugerido**

**Fase 1 (Corto Plazo - 1 semana)**: Opción A - Projection Boxes Específicas

- Revisar annotations actuales
- Split ROIs grandes en múltiples projections específicas
- Re-generar `projection_bboxes_master.txt`
- **Resultado**: Funciona con código actual, 0 cambios necesarios

**Fase 2 (Mediano Plazo - 2-3 semanas)**: Opción B - Detección Iterativa

- Implementar `detect_multi()` con mapeo det→proj
- Integrar con Selection Algorithm (Gap #1)
- Testing exhaustivo con múltiples semáforos
- **Resultado**: Más robusto, maneja casos inesperados

**Fase 3 (Largo Plazo - 1-2 meses)**: Opción C - HD-Map Integration

- Crear módulo `HDMapProjector`
- Obtener/crear HD-Map del escenario de prueba
- Implementar transformaciones 3D→2D
- Integrar con vehicle pose tracking
- **Resultado**: Sistema completo Apollo-equivalent

---

### 3.7 Testing Multi-Detection

```python
# test_multi_detection.py

def test_two_lights_same_roi():
    """
    Test: 2 semáforos distintos en 1 ROI grande
    Esperado: Ambos detectados y asignados
    """
    # 1 ROI grande que contiene 2 semáforos
    boxes = [[300, 100, 600, 400, 0]]

    # Imagen sintética con 2 semáforos
    img = create_synthetic_image_with_two_lights(
        light1_pos=(350, 200),  # Izquierda
        light2_pos=(550, 200)   # Derecha
    )

    pipeline = load_pipeline(device, use_apollo_selector=True)
    valid_dets, recs, assigns, _, _ = pipeline(img, boxes, frame_ts=0.0)

    # Verificar: Deberían haber 2 detections válidas
    assert len(valid_dets) >= 2, f"Solo detectó {len(valid_dets)}, esperaba 2"

    # Con Selection Algorithm, ambas deberían asignarse
    # (requiere 2 projections específicas o HD-Map con 2 semantic IDs)
    assert len(assigns) == 2, f"Solo {len(assigns)} assignments, esperaba 2"

def test_detection_mapping():
    """
    Test: Verificar que detection→projection mapping es correcto
    """
    boxes = [
        [300, 100, 400, 200, 0],  # ROI izquierda
        [500, 100, 600, 200, 1]   # ROI derecha
    ]

    img = create_synthetic_image_with_two_lights(
        light1_pos=(350, 150),
        light2_pos=(550, 150)
    )

    pipeline = load_pipeline(device)
    dets, det_to_proj = pipeline.detect_multi(img, boxes)

    # Verificar mapeo
    for det_idx, proj_id in enumerate(det_to_proj):
        det = dets[det_idx]
        det_center_x = (det[1] + det[3]) / 2

        if det_center_x < 450:  # Izquierda
            assert proj_id == 0
        else:  # Derecha
            assert proj_id == 1
```

---

## 4. 🗺️ Gap #3: Projection Boxes Dinámicas

### 4.1 Estado Actual vs Apollo

### **Sistema Actual: Projection Boxes Estáticas**

```python
# projection_bboxes_master.txt (archivo manual)
frame_000001.jpg 421,165,460,223,0 466,165,511,256,1
frame_000002.jpg 421,165,460,223,0 466,165,511,256,1  # ← Mismas coordenadas
frame_000003.jpg 421,165,460,223,0 466,165,511,256,1  # ← Mismas coordenadas
# ...

# Carga en pipeline
def load_boxes_from_file(frame_name):
    with open('projection_bboxes_master.txt') as f:
        for line in f:
            if frame_name in line:
                boxes = parse_boxes(line)
                return boxes
    return []

# Resultado: Boxes NO se actualizan, son estáticas por video completo

```

**Características**:

- ❌ Coordenadas fijas (no siguen movimiento de cámara)
- ❌ IDs son índices de array (row index), no semánticos
- ❌ Requiere annotación manual para cada video
- ❌ No escala a nuevos escenarios
- ✅ Simple de implementar y debuggear

---

### **Apollo Original: Projection Boxes Dinámicas**

```cpp
// Apollo's dynamic projection flow (cada frame)

// 1. Obtener pose del vehículo
CarPose current_pose = GetVehiclePose();  // GPS + IMU + Odometry
// current_pose = {position: [x, y, z], orientation: [roll, pitch, yaw]}

// 2. Query HD-Map por semáforos cercanos
vector<TrafficLight> nearby_lights = hdmap_->GetTrafficLightsInRange(
    current_pose.position,
    search_radius = 200.0  // metros
);

// 3. Proyectar cada semáforo 3D → 2D
for (auto &light : nearby_lights) {
    // Transformar coordenadas: World → Vehicle → Camera → Image
    Eigen::Vector3d pos_world = light.position_3d;
    Eigen::Vector3d pos_camera = WorldToCamera(pos_world, current_pose);

    // Proyección perspectiva
    Eigen::Vector2d pixel_coords = CameraToPixel(pos_camera, camera_calib_);

    // Estimar tamaño en imagen (función de distancia)
    float distance = pos_camera.norm();
    Eigen::Vector2i size = EstimateSizeFromDistance(distance, light.type);

    // Crear projection box 2D
    base::RectI projection_box;
    projection_box.x = pixel_coords.x() - size.x() / 2;
    projection_box.y = pixel_coords.y() - size.y() / 2;
    projection_box.width = size.x();
    projection_box.height = size.y();

    // Asignar semantic ID (del HD-Map, persistente)
    light.id = light.semantic_id;  // e.g., "TL_001"
    light.projection = projection_box;
}

// Resultado: Projection boxes actualizadas cada frame, siguen semáforos físicos

```

**Características**:

- ✅ Coordenadas dinámicas (siguen movimiento de cámara/vehículo)
- ✅ IDs semánticos persistentes (del HD-Map)
- ✅ Automático (no requiere annotación manual)
- ✅ Escala a cualquier escenario con HD-Map
- ❌ Requiere infraestructura compleja (HD-Map + localization)

---

### 4.2 Por Qué Apollo Usa Projection Boxes Dinámicas

### **Razón 1: Compensar Movimiento del Vehículo**

```
Frame N (vehículo en posición A):
    Semáforo físico en (X=100, Y=50, Z=5) coords mundo
    └→ Proyección 2D: bbox (432, 176, 452, 212) en imagen

Frame N+1 (vehículo avanzó 5 metros):
    Mismo semáforo en (X=100, Y=50, Z=5) coords mundo
    └→ Proyección 2D: bbox (440, 180, 460, 216) en imagen ← CAMBIÓ

Sin actualización dinámica:
    - Projection box en frame N+1 sigue en (432, 176, 452, 212)
    - Semáforo físico ahora está en (440, 180, 460, 216)
    - ❌ Projection box ya NO cubre el semáforo → detección falla

```

---

### **Razón 2: Mantener IDs Semánticos Persistentes**

```cpp
// Con projection boxes dinámicas:

// Frame N:
TL_001 (semáforo izquierdo) → bbox (432, 176, 452, 212)
TL_002 (semáforo derecho)   → bbox (476, 175, 501, 247)

// Frame N+100 (vehículo giró, semáforos cambiaron posiciones en imagen):
TL_001 (ahora en derecha)   → bbox (520, 190, 540, 230)  // ← Actualizado
TL_002 (ahora en izquierda) → bbox (380, 185, 400, 225)  // ← Actualizado

// Historial:
history["TL_001"] = estado_semaforo_izquierdo  // ← ID semántico persistente
history["TL_002"] = estado_semaforo_derecho    // ← ID semántico persistente

// ✅ Historial sigue al semáforo físico, NO a la posición espacial

```

**Contraste con sistema actual**:

```python
# Frame N:
history[0] = estado_semaforo_en_posicion_izquierda  # row_index=0
history[1] = estado_semaforo_en_posicion_derecha    # row_index=1

# Frame N+100 (semáforos intercambiaron posiciones):
history[0] = ??? # Ahora tiene historial del semáforo que ESTÉ en posición row=0
history[1] = ??? # (puede ser diferente semáforo físico)

# ❌ Cross-history transfer ocurre

```

---

### **Razón 3: Adaptación Automática a Nuevos Escenarios**

```
Apollo con HD-Map:
    - Nuevo escenario → Cargar HD-Map del área
    - Projection boxes generadas automáticamente
    - No requiere annotación manual

Sistema actual:
    - Nuevo escenario → Annotación manual de projection_bboxes_master.txt
    - Frame por frame (o propagación manual)
    - Propenso a errores humanos

```

---

### 4.3 Componentes Necesarios para Projection Boxes Dinámicas

### **Componente 1: HD-Map (High Definition Map)**

```json
// Ejemplo: hdmap_intersection_001.json
{
    "map_version": "1.0",
    "coordinate_system": "WGS84",  // GPS coords
    "traffic_lights": [
        {
            "semantic_id": "TL_INT001_001",
            "position_3d": {
                "latitude": -34.603722,
                "longitude": -58.381592,
                "altitude": 25.5
            },
            "orientation": "vertical",
            "lanes_controlled": ["lane_001", "lane_002"],
            "expected_states": ["red", "yellow", "green"],
            "metadata": {
                "installation_date": "2023-01-15",
                "type": "vehicle_signal"
            }
        },
        {
            "semantic_id": "TL_INT001_002",
            "position_3d": {
                "latitude": -34.603735,
                "longitude": -58.381605,
                "altitude": 25.5
            },
            "orientation": "horizontal",
            "lanes_controlled": ["lane_003"],
            "expected_states": ["red", "yellow_arrow", "green_arrow"]
        }
    ],
    "lanes": [
        {
            "id": "lane_001",
            "type": "driving",
            "direction": "north",
            "waypoints": [...]
        }
    ]
}

```

**Herramientas para crear HD-Map**:

- **Manual**: Google Earth + mediciones GPS
- **Semi-automático**: LiDAR scan + annotation tool (e.g., Apollo Studio)
- **Automático**: SLAM + semantic segmentation

---

### **Componente 2: Vehicle Localization**

```python
# src/tlr/localization.py (NUEVO MÓDULO)

class VehicleLocalizer:
    """
    Provee pose del vehículo en tiempo real
    Fusiona múltiples sensores para localization robusta
    """

    def __init__(self):
        self.gps = GPSSensor()
        self.imu = IMUSensor()
        self.odometry = WheelOdometry()

        # Kalman filter para fusión de sensores
        self.ekf = ExtendedKalmanFilter(
            state_dim=6,  # [x, y, z, roll, pitch, yaw]
            measurement_dim=9  # GPS(3) + IMU(3) + Odom(3)
        )

    def get_current_pose(self) -> dict:
        """
        Retorna pose actual del vehículo

        Returns:
            {
                'position': [x, y, z],      # Coordenadas mundo (metros)
                'orientation': [r, p, y],   # Roll, pitch, yaw (radianes)
                'velocity': [vx, vy, vz],   # Velocidad (m/s)
                'timestamp': float,         # Unix timestamp
                'confidence': float         # 0-1
            }
        """
        # Leer sensores
        gps_data = self.gps.read()      # [lat, lon, alt]
        imu_data = self.imu.read()      # [ax, ay, az, gx, gy, gz]
        odom_data = self.odometry.read()  # [dx, dy, dtheta]

        # Fusión con Kalman filter
        measurement = np.concatenate([gps_data, imu_data[:3], odom_data])
        self.ekf.update(measurement)

        state = self.ekf.get_state()

        return {
            'position': state[:3].tolist(),
            'orientation': state[3:6].tolist(),
            'velocity': self.compute_velocity(state),
            'timestamp': time.time(),
            'confidence': self.ekf.get_confidence()
        }

```

**Alternativa simplificada (para testing sin hardware)**:

```python
class SimulatedLocalizer:
    """
    Localizer simulado para testing sin sensores reales
    Usa odometría visual o asume vehículo estático
    """

    def __init__(self, initial_pose=None):
        self.pose = initial_pose or {
            'position': [0, 0, 0],
            'orientation': [0, 0, 0],
            'velocity': [0, 0, 0],
            'timestamp': time.time(),
            'confidence': 1.0
        }

    def get_current_pose(self):
        return self.pose

    def update_from_visual_odometry(self, prev_frame, curr_frame):
        """
        Estima movimiento usando feature matching entre frames
        """
        # Detectar features (ORB, SIFT, etc.)
        kp1, desc1 = self.feature_detector.detect(prev_frame)
        kp2, desc2 = self.feature_detector.detect(curr_frame)

        # Match features
        matches = self.matcher.match(desc1, desc2)

        # Estimar transformación (Essential matrix → R, t)
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.camera_matrix
        )
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)

        # Actualizar pose
        self.pose['position'] += t.flatten().tolist()
        # ... (actualizar orientation con R)

```

---

### **Componente 3: Calibración de Cámara**

```python
# camera_calibration.json
{
    "camera_name": "front_6mm",
    "image_width": 1920,
    "image_height": 1080,
    "intrinsics": {
        "fx": 1000.0,      # Focal length X (pixels)
        "fy": 1000.0,      # Focal length Y (pixels)
        "cx": 960.0,       # Principal point X
        "cy": 540.0,       # Principal point Y
        "skew": 0.0        # Axis skew (usualmente 0)
    },
    "distortion": {
        "model": "radial-tangential",
        "k1": -0.15,       # Radial distortion coef 1
        "k2": 0.08,        # Radial distortion coef 2
        "p1": 0.001,       # Tangential distortion 1
        "p2": -0.002,      # Tangential distortion 2
        "k3": -0.01        # Radial distortion coef 3
    },
    "extrinsics": {
        "position": [1.5, 0.0, 1.2],    # Cámara relativa a vehículo (m)
        "rotation": [0, 0.1, 0]         # Roll, pitch, yaw (rad)
    }
}

```

**Herramientas para calibración**:

- OpenCV calibration tool (checkerboard pattern)
- Kalibr (multi-camera calibration)
- MATLAB Camera Calibrator app

---

### **Componente 4: Proyección 3D → 2D**

```python
# src/tlr/projection_3d_to_2d.py (NUEVO MÓDULO)

import numpy as np

class Projector3Dto2D:
    """
    Proyecta coordenadas 3D del mundo a píxeles 2D de imagen
    """

    def __init__(self, camera_calib: dict):
        self.calib = camera_calib

        # Matriz intrínseca de cámara (K)
        fx = camera_calib['intrinsics']['fx']
        fy = camera_calib['intrinsics']['fy']
        cx = camera_calib['intrinsics']['cx']
        cy = camera_calib['intrinsics']['cy']

        self.K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])

        # Distorsión
        self.dist_coeffs = np.array([
            camera_calib['distortion']['k1'],
            camera_calib['distortion']['k2'],
            camera_calib['distortion']['p1'],
            camera_calib['distortion']['p2'],
            camera_calib['distortion']['k3']
        ])

        # Extrínsecos (cámara relativa a vehículo)
        self.T_cam_vehicle = self.build_transform_matrix(
            camera_calib['extrinsics']['position'],
            camera_calib['extrinsics']['rotation']
        )

    def world_to_camera(self, point_world: np.ndarray, vehicle_pose: dict) -> np.ndarray:
        """
        Transforma punto del mundo a coordenadas de cámara

        point_world: [x, y, z] en coordenadas mundo
        vehicle_pose: pose del vehículo

        Returns: [x, y, z] en coordenadas de cámara
        """
        # Transformación: World → Vehicle
        T_vehicle_world = self.build_transform_matrix(
            vehicle_pose['position'],
            vehicle_pose['orientation']
        )

        # Transformación completa: World → Vehicle → Camera
        T_cam_world = self.T_cam_vehicle @ np.linalg.inv(T_vehicle_world)

        # Aplicar transformación
        point_world_h = np.append(point_world, 1)  # Homogeneous coords
        point_camera_h = T_cam_world @ point_world_h

        return point_camera_h[:3]

    def camera_to_pixel(self, point_camera: np.ndarray) -> tuple:
        """
        Proyecta punto 3D de cámara a píxel 2D

        point_camera: [x, y, z] en coordenadas de cámara

        Returns: (u, v) coordenadas de píxel
        """
        # Proyección perspectiva
        x, y, z = point_camera

        if z <= 0:
            return None  # Punto detrás de la cámara

        # Proyección (sin distorsión)
        u_norm = x / z
        v_norm = y / z

        # Aplicar distorsión radial-tangencial
        r2 = u_norm**2 + v_norm**2
        k1, k2, p1, p2, k3 = self.dist_coeffs

        radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        u_dist = u_norm * radial + 2*p1*u_norm*v_norm + p2*(r2 + 2*u_norm**2)
        v_dist = v_norm * radial + p1*(r2 + 2*v_norm**2) + 2*p2*u_norm*v_norm

        # Aplicar intrínsecos
        u = self.K[0,0] * u_dist + self.K[0,2]
        v = self.K[1,1] * v_dist + self.K[1,2]

        return (int(u), int(v))

    def project_traffic_light(self, tl_world_pos: np.ndarray,
                             vehicle_pose: dict,
                             tl_type: str = "vertical") -> dict:
        """
        Proyecta semáforo 3D a bbox 2D

        Args:
            tl_world_pos: Posición 3D del semáforo en mundo
            vehicle_pose: Pose actual del vehículo
            tl_type: "vertical", "horizontal", "quad"

        Returns:
            {
                'bbox': [x1, y1, x2, y2],
                'center': (u, v),
                'distance': float,
                'visible': bool
            }
        """
        # Transformar a coordenadas de cámara
        point_cam = self.world_to_camera(tl_world_pos, vehicle_pose)

        # Proyectar a píxel
        pixel_coords = self.camera_to_pixel(point_cam)

        if pixel_coords is None:
            return {'visible': False}

        u, v = pixel_coords

        # Calcular distancia
        distance = np.linalg.norm(point_cam)

        # Estimar tamaño en imagen (función de distancia)
        # Tamaño típico: 30cm ancho × 90cm alto para semáforo vertical
        size_world = {'vertical': (0.3, 0.9),
                     'horizontal': (0.9, 0.3),
                     'quad': (0.6, 0.6)}[tl_type]

        # Proyección de tamaño (aproximación simple)
        focal_length = self.K[0,0]
        width_pixels = int((size_world[0] * focal_length) / distance)
        height_pixels = int((size_world[1] * focal_length) / distance)

        # Crear bbox
        x1 = u - width_pixels // 2
        y1 = v - height_pixels // 2
        x2 = u + width_pixels // 2
        y2 = v + height_pixels // 2

        return {
            'bbox': [x1, y1, x2, y2],
            'center': (u, v),
            'distance': distance,
            'visible': True
        }

    @staticmethod
    def build_transform_matrix(position, rotation):
        """
        Construye matriz de transformación 4×4

        position: [x, y, z]
        rotation: [roll, pitch, yaw]
        """
        from scipy.spatial.transform import Rotation

        R = Rotation.from_euler('xyz', rotation).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position

        return T

```

---

### 4.4 Implementación Completa: Dynamic Projector

```python
# src/tlr/dynamic_projector.py (NUEVO MÓDULO)

class DynamicProjector:
    """
    Sistema completo de projection boxes dinámicas
    Integra HD-Map + Localization + Projection 3D→2D
    """

    def __init__(self, hdmap_file: str, camera_calib_file: str):
        # Cargar HD-Map
        with open(hdmap_file) as f:
            self.hdmap = json.load(f)

        # Cargar calibración
        with open(camera_calib_file) as f:
            camera_calib = json.load(f)

        # Inicializar projector
        self.projector = Projector3Dto2D(camera_calib)

        # Cache de semáforos cercanos
        self.nearby_lights_cache = []
        self.cache_position = None
        self.cache_radius = 50.0  # metros

    def get_projection_boxes(self, vehicle_pose: dict) -> List[dict]:
        """
        Genera projection boxes dinámicas para frame actual

        Args:
            vehicle_pose: Pose del vehículo (de Localizer)

        Returns:
            Lista de projections:
            [
                {
                    'semantic_id': 'TL_001',
                    'bbox': [x1, y1, x2, y2],
                    'distance': float,
                    'orientation': str,
                    'visible': bool
                },
                ...
            ]
        """
        # Actualizar cache si vehículo se movió significativamente
        if self._should_update_cache(vehicle_pose):
            self._update_nearby_lights(vehicle_pose)

        projections = []

        for tl in self.nearby_lights_cache:
            # Proyectar semáforo 3D → 2D
            proj_result = self.projector.project_traffic_light(
                np.array(tl['position_3d']),
                vehicle_pose,
                tl['orientation']
            )

            if not proj_result['visible']:
                continue

            # Verificar que bbox está dentro de imagen
            bbox = proj_result['bbox']
            if not self._is_bbox_valid(bbox):
                continue

            projections.append({
                'semantic_id': tl['semantic_id'],
                'bbox': bbox,
                'distance': proj_result['distance'],
                'orientation': tl['orientation'],
                'visible': True
            })

        return projections

    def _should_update_cache(self, vehicle_pose: dict) -> bool:
        """Verifica si necesita actualizar cache de semáforos cercanos"""
        if self.cache_position is None:
            return True

        displacement = np.linalg.norm(
            np.array(vehicle_pose['position']) -
            np.array(self.cache_position)
        )

        return displacement > self.cache_radius * 0.5

    def _update_nearby_lights(self, vehicle_pose: dict):
        """Actualiza cache de semáforos cercanos"""
        vehicle_pos = np.array(vehicle_pose['position'])

        self.nearby_lights_cache = []

        for tl in self.hdmap['traffic_lights']:
            tl_pos = np.array(tl['position_3d'])
            distance = np.linalg.norm(tl_pos - vehicle_pos)

            if distance < self.cache_radius:
                self.nearby_lights_cache.append(tl)

        self.cache_position = vehicle_pose['position']

    def _is_bbox_valid(self, bbox: List[int]) -> bool:
        """Verifica que bbox está dentro de límites de imagen"""
        x1, y1, x2, y2 = bbox

        # Obtener dimensiones de imagen de calibración
        img_w = self.projector.calib['image_width']
        img_h = self.projector.calib['image_height']

        # Verificar bounds
        if x2 < 0 or y2 < 0 or x1 > img_w or y1 > img_h:
            return False

        # Verificar tamaño mínimo
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return False

        return True

```

---

### 4.5 Integración en Pipeline

```python
# src/tlr/pipeline.py (MODIFICADO)

class Pipeline(nn.Module):
    def __init__(self, detector, classifiers, ho, means_det, means_rec,
                 device=None, tracker=None,
                 dynamic_projector=None):  # NUEVO
        super().__init__()
        # ... (inicialización existente) ...

        self.dynamic_projector = dynamic_projector

    def forward(self, img, boxes_or_pose, frame_ts=None):
        """
        Args:
            img: Imagen
            boxes_or_pose:
                - Si dynamic_projector=None: boxes estáticas
                - Si dynamic_projector!=None: vehicle_pose dict
            frame_ts: Timestamp
        """
        # NUEVO: Generar projection boxes dinámicas
        if self.dynamic_projector:
            vehicle_pose = boxes_or_pose  # Es un dict con pose
            projections_data = self.dynamic_projector.get_projection_boxes(vehicle_pose)

            # Convertir a formato boxes
            boxes = [[p['bbox'][0], p['bbox'][1], p['bbox'][2], p['bbox'][3],
                     p['semantic_id']] for p in projections_data]

            # Guardar semantic IDs para tracking
            self.semantic_ids = {i: p['semantic_id'] for i, p in enumerate(projections_data)}
        else:
            boxes = boxes_or_pose  # Es una lista de boxes estáticas
            self.semantic_ids = {i: i for i in range(len(boxes))}

        # Early exit
        if len(boxes) == 0:
            # ...

        # RESTO DEL PIPELINE IGUAL
        detections = self.detect(img, boxes)
        # ...

        # TRACKING con semantic IDs
        if self.tracker:
            # Usar semantic IDs en vez de row indices
            assigns_with_semantic_ids = [
                (self.semantic_ids[proj_idx], det_idx)
                for proj_idx, det_idx in assignments
            ]

            revised = self.tracker.track(
                frame_ts,
                assigns_with_semantic_ids,
                recognitions.cpu().tolist()
            )

        return valid_detections, recognitions, assignments, invalid_detections, revised

# Uso:
# Con projection boxes estáticas (actual)
pipeline = load_pipeline(device)
result = pipeline(img, static_boxes, frame_ts)

# Con projection boxes dinámicas (nuevo)
dynamic_proj = DynamicProjector('hdmap.json', 'camera_calib.json')
localizer = VehicleLocalizer()
pipeline = load_pipeline(device, dynamic_projector=dynamic_proj)

vehicle_pose = localizer.get_current_pose()
result = pipeline(img, vehicle_pose, frame_ts)

```

---

### 4.6 Testing Dynamic Projections

```python
# test_dynamic_projections.py

def test_projection_follows_vehicle_movement():
    """
    Test: Projection boxes actualizan con movimiento del vehículo
    """
    # HD-Map con 1 semáforo fijo
    hdmap = {
        'traffic_lights': [{
            'semantic_id': 'TL_001',
            'position_3d': [100.0, 50.0, 5.0],  # Coordenadas mundo
            'orientation': 'vertical'
        }]
    }

    projector = DynamicProjector(hdmap, camera_calib)

    # Pose 1: Vehículo en origen
    pose1 = {'position': [0, 0, 0], 'orientation': [0, 0, 0]}
    boxes1 = projector.get_projection_boxes(pose1)

    # Pose 2: Vehículo avanzó 10 metros
    pose2 = {'position': [10, 0, 0], 'orientation': [0, 0, 0]}
    boxes2 = projector.get_projection_boxes(pose2)

    # Verificar que bbox cambió
    assert boxes1[0]['bbox'] != boxes2[0]['bbox'], "Bbox debería actualizarse"

    # Verificar que semantic ID se mantiene
    assert boxes1[0]['semantic_id'] == boxes2[0]['semantic_id'] == 'TL_001'

def test_semantic_id_persistence():
    """
    Test: Semantic IDs persisten a pesar de cambios espaciales
    """
    # 2 semáforos que intercambian posiciones visuales
    hdmap = {
        'traffic_lights': [
            {'semantic_id': 'TL_LEFT', 'position_3d': [100, -5, 5]},
            {'semantic_id': 'TL_RIGHT', 'position_3d': [100, 5, 5]}
        ]
    }

    projector = DynamicProjector(hdmap, camera_calib)
    tracker = TrafficLightTracker()

    # Pose inicial: TL_LEFT aparece a la izquierda en imagen
    pose1 = {'position': [0, 0, 0], 'orientation': [0, 0, 0]}
    boxes1 = projector.get_projection_boxes(pose1)

    # Inicializar tracking
    assignments1 = [(0, 0), (1, 1)]  # TL_LEFT→det0, TL_RIGHT→det1
    recognitions1 = [[0,0,0,1], [0,0,1,0]]  # GREEN, YELLOW
    revised1 = tracker.track(0.0, assignments1, recognitions1)

    # Pose nueva: Vehículo giró, ahora TL_LEFT aparece a la derecha
    pose2 = {'position': [0, 0, 0], 'orientation': [0, 0, np.pi]}  # Giró 180°
    boxes2 = projector.get_projection_boxes(pose2)

    # Verificar: TL_LEFT sigue siendo TL_LEFT (aunque cambió posición visual)
    tl_left_box = [b for b in boxes2 if b['semantic_id'] == 'TL_LEFT'][0]

    # History debería seguir al semantic_id
    assert tracker.semantic.history['TL_LEFT'].color == 'green'
    # ✅ No hay cross-history transfer porque ID es semántico, no espacial

```

---

### 4.7 Roadmap de Implementación

### **Fase 1: Preparación (1-2 semanas)**

**Tarea 1.1: Crear HD-Map Simplificado**

```python
# Usar Google Earth + GPS coordinates
# Para escenario de test actual (video doble_chico):

hdmap_test = {
    'traffic_lights': [
        {
            'semantic_id': 'TL_LEFT',
            'position_3d': [-34.603722, -58.381592, 25.5],  # GPS
            'orientation': 'quad',
            'lanes_controlled': ['lane_straight']
        },
        {
            'semantic_id': 'TL_RIGHT',
            'position_3d': [-34.603735, -58.381605, 25.5],
            'orientation': 'quad',
            'lanes_controlled': ['lane_straight']
        }
    ]
}

```

**Tarea 1.2: Calibrar Cámara**

```bash
# Usar OpenCV calibration tool
python calibrate_camera.py --checkerboard_images ./calib_images/*.jpg
# Output: camera_calibration.json

```

**Tarea 1.3: Implementar Localizer Simulado**

```python
# Para testing sin GPS/IMU real
# Asumir vehículo estático o usar visual odometry
localizer = SimulatedLocalizer(initial_pose={'position': [0,0,0], ...})

```

---

### **Fase 2: Implementación Core (2-3 semanas)**

**Tarea 2.1: Implementar Projector3Dto2D**

- Transformaciones world→camera→pixel
- Manejo de distorsión de lente
- Estimación de tamaño en imagen

**Tarea 2.2: Implementar DynamicProjector**

- Carga de HD-Map
- Cache de semáforos cercanos
- Generación de projection boxes

**Tarea 2.3: Modificar Pipeline**

- Aceptar vehicle_pose en vez de boxes estáticas
- Usar semantic IDs para tracking
- Backward compatibility con boxes estáticas

---

### **Fase 3: Testing y Validación (1-2 semanas)**

**Test 1: Projection Accuracy**

```python
# Comparar projection boxes generadas vs ground truth manual
ground_truth_boxes = load_manual_boxes('frame_000001.jpg')
dynamic_boxes = projector.get_projection_boxes(vehicle_pose)

iou = compute_iou(ground_truth_boxes, dynamic_boxes)
assert iou > 0.8, "Projection accuracy insuficiente"

```

**Test 2: Semantic ID Persistence**

```python
# Verificar que semantic IDs no cambian con movimiento
# (test mostrado anteriormente)

```

**Test 3: Cross-History Transfer Fix**

```python
# Verificar que NO ocurre cross-history con semantic IDs
# (test mostrado anteriormente)

```

---

### **Fase 4: Deployment (1 semana)**

**Configuración final**:

```python
# config.yaml
dynamic_projection:
  enabled: true
  hdmap_file: "maps/intersection_001.json"
  camera_calib: "calibration/front_6mm.json"
  localizer_type: "simulated"  # or "gps_imu" para producción

tracking:
  use_semantic_ids: true
  revise_time_s: 1.5
  blink_threshold_s: 0.55

```

**Pipeline final**:

```python
config = load_config('config.yaml')

if config['dynamic_projection']['enabled']:
    projector = DynamicProjector(
        config['dynamic_projection']['hdmap_file'],
        config['dynamic_projection']['camera_calib']
    )
    localizer = create_localizer(config['dynamic_projection']['localizer_type'])
else:
    projector = None
    localizer = None

pipeline = load_pipeline(device, dynamic_projector=projector)

# Loop de procesamiento
for frame in video:
    if projector:
        vehicle_pose = localizer.get_current_pose()
        result = pipeline(frame, vehicle_pose, frame_ts)
    else:
        static_boxes = load_boxes_from_file(frame_name)
        result = pipeline(frame, static_boxes, frame_ts)
```

---

## 5. 🏷️ Gap #4: ID Management (Row Index → Semantic ID)

### 5.1 Problema Fundamental

### **Sistema Actual: Row Index como ID**

```python
# src/tlr/selector.py - Hungarian algorithm
def select_tls(ho, detections, projections, item_shape):
    costs = torch.zeros([len(projections), len(detections)])

    for row, projection in enumerate(projections):  # ← row = 0, 1, 2, ...
        for col, detection in enumerate(detections):
            costs[row, col] = calculate_score(...)

    assignments = ho.maximize(costs)
    # Resultado: [[row_idx, det_idx], [row_idx, det_idx], ...]
    #              ↑
    #         Este row_idx se usa como proj_id en tracking

```

**En tracking**:

```python
# src/tlr/tracking.py
def update(self, frame_ts, assignments, recognitions):
    for proj_id, det_idx in assignments:  # proj_id = row_idx
        if proj_id not in self.history:
            self.history[proj_id] = SemanticTable(proj_id, ...)

        st = self.history[proj_id]  # ← Historial indexado por row_idx

```

**Consecuencia**:

```python
# projection_bboxes_master.txt
# Frame 1:
421,165,460,223,0  # row_idx=0, semáforo físico A
466,165,511,256,1  # row_idx=1, semáforo físico B

# Si en Frame 100 intercambiamos orden en archivo:
466,165,511,256,1  # row_idx=0 ← AHORA semáforo físico B
421,165,460,223,0  # row_idx=1 ← AHORA semáforo físico A

# Tracking:
history[0] = historial de lo que esté en row_idx=0
# ❌ El historial se "transfiere" entre semáforos físicos

```

---

### **Apollo Original: Semantic ID Persistente**

```cpp
// Apollo's HD-Map based IDs
struct TrafficLight {
    string semantic_id;  // e.g., "TL_INTERSECTION_001_NORTH_LEFT"
    // Este ID:
    // - Viene del HD-Map
    // - Es único globalmente
    // - Persiste independientemente de orden o posición
};

// En tracking:
map<string, SemanticTable> history_;
// history_["TL_INTERSECTION_001_NORTH_LEFT"] = estado del semáforo físico específico

// ✅ El historial SIEMPRE sigue al mismo semáforo físico

```

**Ventaja crítica**:

```cpp
// Frame 1: Vehículo ve semáforo desde lejos (aparece a la izquierda)
TL_NORTH_LEFT → projection bbox (100, 50, 140, 120)
history_["TL_NORTH_LEFT"] = {color: "red", ...}

// Frame 100: Vehículo giró (mismo semáforo ahora a la derecha)
TL_NORTH_LEFT → projection bbox (800, 50, 840, 120)  // ← Cambió posición
history_["TL_NORTH_LEFT"] = {color: "red", ...}      // ← MISMO historial

// ✅ No hay cross-history transfer porque ID es semántico, no espacial

```

---

### 5.2 Análisis del Gap

| Aspecto | Row Index (Actual) | Semantic ID (Apollo) |
| --- | --- | --- |
| **Definición** | Índice en array de projections | ID único del HD-Map |
| **Persistencia** | ❌ Depende del orden en archivo | ✅ Persistente entre frames |
| **Scope** | Local (por frame/video) | Global (toda la ciudad) |
| **Tracking** | Sigue posición espacial (region) | Sigue semáforo físico |
| **Cross-history** | ✅ Puede ocurrir | ❌ No ocurre |
| **Requiere** | Nada (solo array) | HD-Map con IDs |
| **Debugging** | Difícil (números sin significado) | Fácil (nombres descriptivos) |

---

### 5.3 Por Qué Semantic IDs Son Críticos

### **Razón 1: Tracking Robusto a Cambios de Vista**

```python
# Escenario: Vehículo girando en intersección

# Vista 1 (vehículo mirando norte):
projections = [
    {'bbox': [100, 50, 140, 120], 'id': 'TL_NORTH_LEFT'},
    {'bbox': [800, 50, 840, 120], 'id': 'TL_NORTH_RIGHT'}
]

# Vista 2 (vehículo giró 90°, ahora mira este):
projections = [
    {'bbox': [100, 50, 140, 120], 'id': 'TL_EAST_LEFT'},   # ← Nuevo semáforo visible
    {'bbox': [800, 50, 840, 120], 'id': 'TL_NORTH_RIGHT'}  # ← Mismo de antes
]

# Con row_index:
# row=0 en vista 1 = TL_NORTH_LEFT
# row=0 en vista 2 = TL_EAST_LEFT
# ❌ history[0] se "reasigna" a semáforo diferente

# Con semantic_id:
# history['TL_NORTH_LEFT'] persiste (aunque ya no visible)
# history['TL_EAST_LEFT'] se crea nuevo
# ✅ Cada semáforo mantiene su propio historial

```

---

### **Razón 2: Fusión Multi-Cámara**

```python
# Apollo con múltiples cámaras

# Cámara frontal (6mm wide-angle):
projections_front = [
    {'id': 'TL_001', 'bbox': [100, 50, 140, 120], 'camera': 'front'}
]

# Cámara telephoto (25mm):
projections_tele = [
    {'id': 'TL_001', 'bbox': [500, 300, 580, 420], 'camera': 'tele'}  # ← MISMO semáforo
]

# Fusión:
detections_front = detector(front_camera_image, projections_front)
detections_tele = detector(tele_camera_image, projections_tele)

# Ambas detecciones del MISMO semáforo (TL_001) se fusionan
# porque comparten semantic_id

# Con row_index:
# front: row=0
# tele: row=0
# ❌ Son dos "0" diferentes, no se puede fusionar

```

---

### **Razón 3: Debugging y Análisis**

```python
# Log con row_index (actual):
# Frame 100: proj_id=0 changed red→green
# Frame 101: proj_id=1 blink detected
# ❓ ¿Cuál semáforo es el 0? ¿El 1? ¿Izquierdo o derecho?

# Log con semantic_id (Apollo):
# Frame 100: TL_INTERSECTION_001_NORTH_LEFT changed red→green
# Frame 101: TL_INTERSECTION_001_SOUTH_YELLOW blink detected
# ✅ Inmediatamente se sabe qué semáforo es

```

---

### 5.4 Solución: Migración a Semantic IDs

### **Paso 1: Extender Formato de Projection Boxes**

```python
# projection_bboxes_master.txt (formato extendido)
# Antes:
# frame_000001.jpg 421,165,460,223,0 466,165,511,256,1

# Después (con semantic IDs):
# frame_000001.jpg 421,165,460,223,TL_LEFT 466,165,511,256,TL_RIGHT

# O en JSON para mayor flexibilidad:
# projection_boxes.json
{
    "frames": {
        "frame_000001.jpg": [
            {
                "semantic_id": "TL_LEFT",
                "bbox": [421, 165, 460, 223],
                "orientation": "quad",
                "expected_states": ["red", "yellow", "green"]
            },
            {
                "semantic_id": "TL_RIGHT",
                "bbox": [466, 165, 511, 256],
                "orientation": "quad",
                "expected_states": ["red", "yellow_blink"]
            }
        ]
    }
}

```

---

### **Paso 2: Modificar Selector para Usar Semantic IDs**

```python
# src/tlr/selector.py (MODIFICADO)

def select_tls_with_semantic_ids(ho, detections, projections_with_ids, item_shape):
    """
    Version con semantic IDs

    Args:
        projections_with_ids: Lista de dicts con 'semantic_id' y ProjectionROI
    """
    # Construir mapping
    semantic_id_to_idx = {p['semantic_id']: i for i, p in enumerate(projections_with_ids)}

    # Matriz de costos (igual que antes)
    costs = torch.zeros([len(projections_with_ids), len(detections)])

    for row, proj_data in enumerate(projections_with_ids):
        projection = proj_data['projection']
        for col, detection in enumerate(detections):
            # ... (cálculo de score igual) ...
            costs[row, col] = score

    # Hungarian assignment
    row_indices, col_indices = ho.maximize(costs.numpy())

    # Convertir row_indices a semantic_ids
    assignments_with_semantic_ids = []
    for row_idx, det_idx in zip(row_indices, col_indices):
        semantic_id = projections_with_ids[row_idx]['semantic_id']
        assignments_with_semantic_ids.append((semantic_id, det_idx))

    return assignments_with_semantic_ids
    # Retorna: [("TL_LEFT", 0), ("TL_RIGHT", 1), ...]

```

---

### **Paso 3: Modificar Tracking para Usar Semantic IDs**

```python
# src/tlr/tracking.py (MODIFICADO)

class SemanticTable:
    def __init__(self, semantic_id: str, time_stamp: float, color: str):
        self.semantic_id = semantic_id  # ← String ID en vez de int
        # ... (resto igual) ...

class SemanticDecision:
    def __init__(self, ...):
        # history indexado por semantic_id (string)
        self.history: Dict[str, SemanticTable] = {}  # ← Cambio de int a str

    def update(self, frame_ts, assignments, recognitions):
        results = {}

        for semantic_id, det_idx in assignments:  # ← semantic_id es string
            cls = int(max(range(len(recognitions[det_idx])),
                         key=lambda i: recognitions[det_idx][i]))
            color = ["black","red","yellow","green"][cls]

            # Obtener o crear historial
            if semantic_id not in self.history:
                self.history[semantic_id] = SemanticTable(semantic_id, frame_ts, color)

            st = self.history[semantic_id]

            # ... (lógica de tracking igual) ...

            results[semantic_id] = (st.color, st.blink)

        return results
        # Retorna: {"TL_LEFT": ("green", False), "TL_RIGHT": ("red", True)}

```

---

### **Paso 4: Modificar Pipeline**

```python
# src/tlr/pipeline.py (MODIFICADO)

class Pipeline(nn.Module):
    def forward(self, img, boxes_with_ids, frame_ts=None):
        """
        Args:
            boxes_with_ids: Lista de dicts con 'semantic_id' y 'bbox'
                [
                    {'semantic_id': 'TL_LEFT', 'bbox': [x1,y1,x2,y2]},
                    {'semantic_id': 'TL_RIGHT', 'bbox': [x1,y1,x2,y2]}
                ]
        """
        # Convertir a formato interno
        projections_with_ids = []
        for box_data in boxes_with_ids:
            bbox = box_data['bbox']
            projection = ProjectionROI(bbox[0], bbox[1],
                                      bbox[2]-bbox[0], bbox[3]-bbox[1])
            projections_with_ids.append({
                'semantic_id': box_data['semantic_id'],
                'projection': projection
            })

        # Early exit
        if len(projections_with_ids) == 0:
            # ...

        # Detección (igual)
        detections = self.detect(img, [p['projection'] for p in projections_with_ids])

        # Filtrado (igual)
        tl_types = torch.argmax(detections[:, 5:], dim=1)
        valid_mask = tl_types != 0
        valid_detections = detections[valid_mask]

        # Assignment CON semantic IDs
        assignments = select_tls_with_semantic_ids(
            self.ho, valid_detections, projections_with_ids, img.shape
        )
        # assignments = [("TL_LEFT", det_idx), ("TL_RIGHT", det_idx)]

        # Reconocimiento (igual)
        recognitions = self.recognize(img, valid_detections, tl_types[valid_mask])

        # Tracking CON semantic IDs
        if self.tracker:
            revised = self.tracker.track(
                frame_ts,
                assignments,  # Ya tienen semantic IDs
                recognitions.cpu().tolist()
            )
            # revised = {"TL_LEFT": ("green", False), "TL_RIGHT": ("red", True)}

        return valid_detections, recognitions, assignments, invalid_detections, revised

```

---

### 5.5 Migración Gradual: Backward Compatibility

### **Dual Support: Row Index + Semantic ID**

```python
# src/tlr/utils.py (NUEVO)

def normalize_boxes_input(boxes_input):
    """
    Acepta boxes en múltiples formatos y normaliza a formato con semantic_ids

    Formatos aceptados:
    1. Lista antigua: [[x1,y1,x2,y2,id_num], ...]
    2. Lista con IDs string: [[x1,y1,x2,y2,"TL_001"], ...]
    3. Lista de dicts: [{'semantic_id': "TL_001", 'bbox': [x1,y1,x2,y2]}, ...]

    Returns:
        Lista de dicts con 'semantic_id' (string) y 'bbox'
    """
    if not boxes_input:
        return []

    # Detectar formato
    first_box = boxes_input[0]

    # Formato 3: Ya está normalizado
    if isinstance(first_box, dict) and 'semantic_id' in first_box:
        return boxes_input

    # Formato 1 o 2: Lista
    if isinstance(first_box, (list, tuple)):
        normalized = []
        for i, box in enumerate(boxes_input):
            x1, y1, x2, y2, box_id = box

            # Convertir ID a string
            if isinstance(box_id, (int, float)):
                semantic_id = f"proj_{int(box_id)}"  # Fallback: "proj_0", "proj_1"
            else:
                semantic_id = str(box_id)

            normalized.append({
                'semantic_id': semantic_id,
                'bbox': [x1, y1, x2, y2],
                '_original_index': i  # Para debugging
            })

        return normalized

    raise ValueError(f"Formato de boxes no reconocido: {type(first_box)}")

# Uso en pipeline:
def forward(self, img, boxes_input, frame_ts=None):
    # Normalizar input
    boxes_with_ids = normalize_boxes_input(boxes_input)

    # ... (resto del pipeline con semantic IDs) ...

```

---

### **Helper para Migración de Archivos**

```python
# tools/migrate_to_semantic_ids.py (NUEVO SCRIPT)

import json

def migrate_boxes_file(old_file: str, output_file: str, id_mapping: dict = None):
    """
    Migra projection_bboxes_master.txt a formato con semantic IDs

    Args:
        old_file: projection_bboxes_master.txt (formato antiguo)
        output_file: projection_boxes.json (formato nuevo)
        id_mapping: Dict opcional {numeric_id: "semantic_id"}
                   Si None, genera IDs automáticos
    """
    frames_data = {}

    with open(old_file) as f:
        for line in f:
            parts = line.strip().split()
            frame_name = parts[0]
            boxes_str = parts[1:]

            boxes = []
            for box_str in boxes_str:
                coords = list(map(int, box_str.split(',')))
                x1, y1, x2, y2, numeric_id = coords

                # Generar semantic_id
                if id_mapping and numeric_id in id_mapping:
                    semantic_id = id_mapping[numeric_id]
                else:
                    semantic_id = f"TL_{frame_name.split('.')[0]}_{numeric_id}"

                boxes.append({
                    'semantic_id': semantic_id,
                    'bbox': [x1, y1, x2, y2],
                    'orientation': 'unknown',  # Llenar manualmente después
                    '_migrated_from_id': numeric_id
                })

            frames_data[frame_name] = boxes

    # Guardar en JSON
    output_data = {'frames': frames_data}
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Migración completada: {len(frames_data)} frames procesados")
    print(f"Output guardado en: {output_file}")

# Uso:
if __name__ == '__main__':
    # Opción 1: IDs automáticos
    migrate_boxes_file(
        'projection_bboxes_master.txt',
        'projection_boxes.json'
    )

    # Opción 2: Con mapeo manual
    id_mapping = {
        0: "TL_INTERSECTION_001_LEFT",
        1: "TL_INTERSECTION_001_RIGHT"
    }
    migrate_boxes_file(
        'projection_bboxes_master.txt',
        'projection_boxes.json',
        id_mapping
    )

```

---

### 5.6 Testing Semantic IDs

```python
# test_semantic_ids.py

def test_semantic_id_persistence():
    """
    Test: Semantic IDs persisten entre frames
    """
    # Frame 1
    boxes1 = [
        {'semantic_id': 'TL_LEFT', 'bbox': [100, 50, 140, 120]},
        {'semantic_id': 'TL_RIGHT', 'bbox': [800, 50, 840, 120]}
    ]

    pipeline = load_pipeline(device)
    tracker = pipeline.tracker

    # Procesar frame 1
    img1 = load_image('frame_001.jpg')
    result1 = pipeline(img1, boxes1, frame_ts=0.0)

    # Frame 2: Boxes intercambiadas espacialmente
    boxes2 = [
        {'semantic_id': 'TL_RIGHT', 'bbox': [100, 50, 140, 120]},  # ← Cambió posición
        {'semantic_id': 'TL_LEFT', 'bbox': [800, 50, 840, 120]}    # ← Cambió posición
    ]

    img2 = load_image('frame_002.jpg')
    result2 = pipeline(img2, boxes2, frame_ts=0.033)

    # Verificar: Historiales siguen a semantic_id, NO a posición
    assert 'TL_LEFT' in tracker.semantic.history
    assert 'TL_RIGHT' in tracker.semantic.history

    # TL_LEFT debería tener su propio historial (independiente de posición)
    assert tracker.semantic.history['TL_LEFT'].semantic_id == 'TL_LEFT'
    assert tracker.semantic.history['TL_RIGHT'].semantic_id == 'TL_RIGHT'

def test_backward_compatibility():
    """
    Test: Sistema acepta formato antiguo y nuevo
    """
    # Formato antiguo (lista con numeric IDs)
    old_format = [
        [100, 50, 140, 120, 0],
        [800, 50, 840, 120, 1]
    ]

    # Formato nuevo (dicts con semantic IDs)
    new_format = [
        {'semantic_id': 'TL_LEFT', 'bbox': [100, 50, 140, 120]},
        {'semantic_id': 'TL_RIGHT', 'bbox': [800, 50, 840, 120]}
    ]

    pipeline = load_pipeline(device)
    img = load_image('frame_001.jpg')

    # Ambos formatos deberían funcionar
    result_old = pipeline(img, old_format, frame_ts=0.0)
    result_new = pipeline(img, new_format, frame_ts=0.0)

    # Verificar que producen resultados equivalentes
    # (excepto por IDs: "proj_0" vs "TL_LEFT")
    assert len(result_old[0]) == len(result_new[0])  # Same detections

def test_cross_history_fix():
    """
    Test: Semantic IDs eliminan cross-history transfer
    """
    boxes = [
        {'semantic_id': 'TL_LEFT', 'bbox': [100, 50, 140, 120]},
        {'semantic_id': 'TL_RIGHT', 'bbox': [800, 50, 840, 120]}
    ]

    pipeline = load_pipeline(device)

    # Frame 1-100: TL_LEFT=green, TL_RIGHT=yellow_blink
    for i in range(100):
        img = create_frame_with_lights(left_color='green', right_color='yellow')
        _ = pipeline(img, boxes, frame_ts=i*0.033)

    # Verificar historiales
    assert pipeline.tracker.semantic.history['TL_LEFT'].color == 'green'
    assert pipeline.tracker.semantic.history['TL_RIGHT'].blink == True

    # Frame 101: Intercambio físico de semáforos
    # (simular con modificación de detecciones)
    img_swapped = create_frame_with_lights(left_color='yellow', right_color='green')
    result = pipeline(img_swapped, boxes, frame_ts=101*0.033)

    # Con semantic IDs, cada semáforo mantiene su historial
    # TL_LEFT ahora ve yellow → transición green→yellow (válida)
    # TL_RIGHT ahora ve green → blink se detiene (correcto)

    assert pipeline.tracker.semantic.history['TL_LEFT'].color == 'yellow'
    assert pipeline.tracker.semantic.history['TL_RIGHT'].blink == False
    # ✅ No hay cross-history transfer

```

---

### 5.7 Roadmap de Implementación

### **Fase 1: Preparación (3-5 días)**

**Día 1-2: Extender Formato de Datos**

```bash
# Crear nuevos archivos con semantic IDs
python tools/migrate_to_semantic_ids.py \
    --input projection_bboxes_master.txt \
    --output projection_boxes.json \
    --id-mapping id_mapping.yaml

```

**Día 3: Implementar Normalización**

```python
# Implementar normalize_boxes_input() en utils.py
# Testing con ambos formatos

```

**Día 4-5: Documentación**

```markdown
# Actualizar README.md con nuevo formato
# Crear guía de migración para usuarios

```

---

### **Fase 2: Modificación de Código (1 semana)**

**Día 1-2: Selector**

- Implementar `select_tls_with_semantic_ids()`
- Testing unitario

**Día 3-4: Tracking**

- Modificar `SemanticTable` y `SemanticDecision`
- Cambiar `Dict[int, ...]` a `Dict[str, ...]`
- Testing de persistencia

**Día 5: Pipeline**

- Integrar semantic IDs en `forward()`
- Backward compatibility con `normalize_boxes_input()`

---

### **Fase 3: Testing y Validación (3-5 días)**

**Test Suite**:

```bash
# Test 1: Persistence
pytest test_semantic_ids.py::test_semantic_id_persistence

# Test 2: Backward compatibility
pytest test_semantic_ids.py::test_backward_compatibility

# Test 3: Cross-history fix
pytest test_semantic_ids.py::test_cross_history_fix

# Test 4: Integration
pytest test_semantic_ids.py::test_full_pipeline_with_semantic_ids

```

---

### **Fase 4: Deployment (2-3 días)**

**Configuración**:

```yaml
# config.yaml
projection_boxes:
  format: "semantic_ids"  # or "numeric_ids" for backward compat
  file: "projection_boxes.json"

tracking:
  use_semantic_ids: true
  history_backend: "dict"  # Future: "redis" for distributed

```

**Rollout**:

1. Deploy con backward compatibility activada
2. Migrar datos existentes
3. Monitorear logs para warnings de formato antiguo
4. Deprecar formato antiguo después de período de transición

---

## 6. 🧠 Gap #6: Dependencia Espacial del Recognizer

### 6.1 Problema Descubierto

### **Comportamiento Observado**

```python
# Test de Swapping (experimento crítico)

# Configuración normal (posiciones esperadas):
Det0 en posición (432,176,452,212):  # Izquierda
  → Input: ROI de semáforo verde
  → Output: [0.0, 0.0, 0.0, 1.0]  # GREEN ✅

Det1 en posición (476,175,501,247):  # Derecha
  → Input: ROI de semáforo amarillo
  → Output: [0.0, 0.0, 1.0, 0.0]  # YELLOW ✅

# Swap físico (intercambio de detecciones):
Det0 en posición (476,175,501,247):  # Derecha (intercambiado)
  → Input: MISMOS PÍXELES de semáforo verde
  → Output: [1.0, 0.0, 0.0, 0.0]  # BLACK ❌ (¡cambió!)

Det1 en posición (432,176,452,212):  # Izquierda (intercambiado)
  → Input: MISMOS PÍXELES de semáforo amarillo
  → Output: [1.0, 0.0, 0.0, 0.0]  # BLACK ❌ (¡cambió!)

```

**Hallazgo crítico**: El modelo NO clasificó según píxeles, sino según **posición espacial**

---

### **Root Cause: Sobreajuste Espacial en Entrenamiento**

```python
# Dataset de entrenamiento (hipótesis basada en comportamiento):
# Todos los ejemplos tienen estructura espacial fija:

training_data = [
    # Semáforo verde SIEMPRE en posición ~(432, 176)
    {'image': img1, 'bbox': [430, 175, 450, 210], 'label': 'GREEN'},
    {'image': img2, 'bbox': [432, 176, 452, 212], 'label': 'GREEN'},
    # ...

    # Semáforo amarillo SIEMPRE en posición ~(476, 175)
    {'image': img10, 'bbox': [475, 174, 500, 246], 'label': 'YELLOW'},
    {'image': img11, 'bbox': [476, 175, 501, 247], 'label': 'YELLOW'},
    # ...
]

# El modelo aprendió correlación espuria:
# "Si bbox está en ~(432, 176) Y píxeles muestran luz → GREEN"
# "Si bbox está en ~(476, 175) Y píxeles muestran luz → YELLOW"
# "Si bbox NO está en posición esperada → BLACK (desconocido)"

```

---

### 6.2 ¿Cómo es Posible Esta Dependencia?

### **Análisis de la Arquitectura del Recognizer**

```python
# src/tlr/recognizer.py
class Recognizer(nn.Module):
    def forward(self, x):
        # x shape: [1, 3, H, W] donde H×W depende del tipo
        # Para quad: [1, 3, 64, 64]

        conv1 = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2, padding=1)
        # Shape: [1, 32, 32, 32]

        conv2 = F.max_pool2d(F.relu(self.conv2(conv1)), kernel_size=3, stride=2, padding=1)
        # Shape: [1, 64, 16, 16]

        conv3 = F.max_pool2d(F.relu(self.conv3(conv2)), kernel_size=3, stride=2, padding=1)
        # Shape: [1, 128, 8, 8]

        conv4 = F.max_pool2d(F.relu(self.conv4(conv3)), kernel_size=3, stride=2, padding=1)
        # Shape: [1, 128, 4, 4]

        conv5 = self.pool5(F.relu(self.conv5(conv4)))
        # pool5 = AvgPool2d(kernel_size=(4,4), stride=(4,4))
        # Shape: [1, 128, 1, 1]

        ft = F.relu(self.ft(conv5))
        # Shape: [1, 128, 1, 1]

        logits = self.logits(ft.reshape(-1, 128))
        # Shape: [1, 4]

        prob = F.softmax(logits, dim=1)
        return prob

```

**¿Dónde está la información espacial?**

Teoría 1: **Pooling con información de posición**

```python
# pool5 parámetros específicos por tipo:
# quad:  kernel=(4,4), stride=(4,4) → de 4×4 a 1×1
# hori:  kernel=(2,6), stride=(2,6) → de 2×6 a 1×1
# vert:  kernel=(6,2), stride=(6,2) → de 6×2 a 1×1

# Si el crop NO está perfectamente centrado en el semáforo,
# la posición del semáforo dentro del crop puede variar
# → Pooling "recoge" diferentes activaciones según dónde esté la luz

```

Teoría 2: **Preprocesamiento variable por posición**

```python
# src/tlr/tools/utils.py:241-252
def preprocess4rec(img, det_box, shape, means_rec):
    xl, xr, yt, yb = det_box[0], det_box[2], det_box[1], det_box[3]
    src = img[yt:yb, xl:xr]  # ← Crop usa coordenadas absolutas

    # Resize a tamaño fijo
    dst = torch.zeros(shape, device=src.device)
    resized = ResizeGPU(src, dst, means_rec)
    return resized

# Problema potencial:
# Si resize introduce artifacts que correlacionan con posición original...
# O si means_rec fueron calculados con bias espacial...

```

Teoría 3: **Feature leakage en entrenamiento**

```python
# Durante entrenamiento, si el modelo tuvo acceso a metadata:
# - Coordenadas absolutas del bbox
# - ID de la imagen
# - Cualquier info correlacionada con posición

# Ejemplo de leakage accidental:
training_input = {
    'image_crop': cropped_image,
    'bbox_coords': [x1, y1, x2, y2],  # ← Si esto se pasó al modelo
    'frame_id': 'frame_000123'
}

# El modelo podría usar bbox_coords para mejorar predicción
# → Aprende "si x~430 → verde, si x~476 → amarillo"

```

---

### 6.3 Impacto del Problema

### **Escenarios Afectados**

**Escenario 1: Cambio de Ángulo de Cámara**

```python
# Video 1: Cámara frontal
# Semáforo izq en (432, 176) → GREEN ✅
# Semáforo der en (476, 175) → YELLOW ✅

# Video 2: Cámara con ángulo diferente (5° rotada)
# MISMO semáforo izq ahora en (450, 185) → BLACK ❌
# MISMO semáforo der ahora en (495, 190) → BLACK ❌

# Accuracy: 100% → 0% solo por cambio de ángulo

```

**Escenario 2: Diferentes Tipos de Vehículos**

```python
# Vehículo bajo (sedan): Cámara a 1.2m altura
# Semáforos aparecen en posiciones (432, 176) y (476, 175) → OK ✅

# Vehículo alto (SUV): Cámara a 1.6m altura
# MISMOS semáforos aparecen en (432, 150) y (476, 145) → FAIL ❌

```

**Escenario 3: Nuevas Intersecciones**

```python
# Intersección entrenamiento: Semáforos separados por 44 píxeles
# Modelo aprendió: "verde a la izquierda, amarillo a la derecha"

# Nueva intersección: Semáforos separados por 200 píxeles
# Posiciones: (200, 150) y (600, 150)
# Modelo: BLACK para ambos ❌

```

---

### 6.4 ¿Por Qué Apollo No Tiene Este Problema?

### **Diseño de Apollo: Position-Agnostic Training**

```cpp
// Apollo's training data preparation
void PrepareTrainingData() {
    for (auto &sample : dataset) {
        // 1. Detectar semáforo en imagen completa
        BBox traffic_light = DetectTrafficLight(sample.image);

        // 2. Crop con MARGIN VARIABLE (data augmentation espacial)
        int margin_x = random_uniform(-20, 20);  // ← Variación espacial
        int margin_y = random_uniform(-20, 20);

        BBox crop_box = {
            traffic_light.x1 + margin_x,
            traffic_light.y1 + margin_y,
            traffic_light.x2 + margin_x,
            traffic_light.y2 + margin_y
        };

        // 3. Crop y resize
        cv::Mat cropped = CropAndResize(sample.image, crop_box, target_size);

        // 4. Agregar a dataset de entrenamiento
        training_samples.push_back({
            'image': cropped,
            'label': sample.ground_truth_color
        });
    }
}

```

**Características clave**:

- ✅ Crops con offsets aleatorios (semáforo no siempre centrado)
- ✅ Múltiples escalas (zoom in/out)
- ✅ Rotaciones ligeras
- ✅ Solo píxeles como input (sin coordenadas absolutas)

---

### 6.5 Solución: Re-entrenamiento con Data Augmentation

### **Estrategia 1: Spatial Augmentation**

```python
# tools/retrain_recognizer.py

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class TrafficLightDataset(Dataset):
    """
    Dataset con augmentation espacial para eliminar dependencia de posición
    """

    def __init__(self, images, labels, augment=True):
        self.images = images
        self.labels = labels
        self.augment = augment

        # Augmentation pipeline
        self.spatial_aug = T.Compose([
            # 1. Random crop (simula diferentes posiciones)
            T.RandomCrop(size=(64, 64), padding=8),

            # 2. Random affine (rotación + traslación + escala)
            T.RandomAffine(
                degrees=10,           # ±10° rotación
                translate=(0.2, 0.2), # ±20% traslación
                scale=(0.8, 1.2),     # 80%-120% escala
                shear=5               # ±5° shear
            ),

            # 3. Random horizontal flip (solo si semáforo simétrico)
            T.RandomHorizontalFlip(p=0.3),
        ])

        self.color_aug = T.Compose([
            # 4. Color jitter (robustez a iluminación)
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),

            # 5. Random noise
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
        ])

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            # Aplicar augmentation espacial
            img = self.spatial_aug(img)
            img = self.color_aug(img)

        return img, label

    def __len__(self):
        return len(self.images)

# Training loop
def train_position_agnostic_recognizer(model, train_dataset, val_dataset, epochs=50):
    """
    Entrena recognizer SIN dependencia espacial
    """
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, labels in train_loader:
            optimizer.zero_grad()

            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_acc = evaluate_position_robustness(model, val_loader)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    return model

def evaluate_position_robustness(model, val_loader):
    """
    Evalúa robustez a cambios de posición
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # Test 1: Posición original
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # Test 2: Shifted positions (augmentation)
            for shift_x in [-10, 0, 10]:
                for shift_y in [-10, 0, 10]:
                    shifted = T.functional.affine(
                        images,
                        angle=0,
                        translate=[shift_x, shift_y],
                        scale=1.0,
                        shear=0
                    )
                    outputs_shifted = model(shifted)
                    pred_shifted = torch.argmax(outputs_shifted, dim=1)

                    # Verificar consistencia
                    if not torch.equal(pred, pred_shifted):
                        print(f"WARNING: Inconsistency at shift ({shift_x}, {shift_y})")

    return correct / total

```

---

### **Estrategia 2: Position Encoding Removal**

```python
# Verificar que el modelo NO recibe coordenadas como input

def audit_model_inputs(model, sample_batch):
    """
    Audita qué información recibe realmente el modelo
    """
    # Hook para capturar inputs
    inputs_captured = []

    def hook_fn(module, input, output):
        inputs_captured.append(input[0].shape)

    # Registrar hook en primera capa
    hook = model.conv1.register_forward_hook(hook_fn)

    # Forward pass
    _ = model(sample_batch)

    # Verificar shape
    input_shape = inputs_captured[0]
    print(f"Model input shape: {input_shape}")

    # Debería ser: [batch, 3, H, W] (solo píxeles)
    # NO debería ser: [batch, 5, H, W] (con coords) o similar
    assert input_shape[1] == 3, f"Model receives {input_shape[1]} channels, expected 3 (RGB only)"

    hook.remove()
    print("✅ Model only receives pixel data (no position info)")

# Ejecutar audit
audit_model_inputs(quad_recognizer, torch.randn(1, 3, 64, 64))

```

---

### **Estrategia 3: Curriculum Learning con Posiciones Variadas**

```python
# Progressive training: fácil → difícil

def curriculum_training(model, dataset, curriculum_stages=3):
    """
    Entrena con dificultad creciente en variación espacial
    """

    # Stage 1: Sin variación espacial (baseline)
    print("Stage 1: Original positions only")
    stage1_dataset = TrafficLightDataset(
        dataset.images,
        dataset.labels,
        augment=False  # Sin augmentation
    )
    train_position_agnostic_recognizer(model, stage1_dataset, epochs=20)

    # Stage 2: Variación leve (±10 píxeles)
    print("Stage 2: Light spatial variation")
    stage2_aug = T.RandomAffine(degrees=5, translate=(0.1, 0.1))
    stage2_dataset = TrafficLightDataset(
        dataset.images,
        dataset.labels,
        augment=True,
        custom_aug=stage2_aug
    )
    train_position_agnostic_recognizer(model, stage2_dataset, epochs=15)

    # Stage 3: Variación fuerte (±30 píxeles, rotación)
    print("Stage 3: Strong spatial variation")
    stage3_aug = T.RandomAffine(degrees=15, translate=(0.3, 0.3), scale=(0.7, 1.3))
    stage3_dataset = TrafficLightDataset(
        dataset.images,
        dataset.labels,
        augment=True,
        custom_aug=stage3_aug
    )
    train_position_agnostic_recognizer(model, stage3_dataset, epochs=15)

    return model

```

---

### 6.6 Solución Alternativa: Feature Normalization

Si re-entrenar no es posible, aplicar normalización espacial:

```python
# src/tlr/recognizer_wrapper.py (NUEVO)

class PositionNormalizedRecognizer(nn.Module):
    """
    Wrapper que normaliza features espaciales antes de clasificación
    """

    def __init__(self, base_recognizer):
        super().__init__()
        self.base_recognizer = base_recognizer

        # Spatial Transformer Network para normalización
        self.stn = SpatialTransformerNetwork()

    def forward(self, x):
        # 1. Normalizar posición con STN
        x_normalized = self.stn(x)

        # 2. Clasificar con recognizer original
        output = self.base_recognizer(x_normalized)

        return output

class SpatialTransformerNetwork(nn.Module):
    """
    Red que aprende a normalizar posición del semáforo en el crop
    """

    def __init__(self):
        super().__init__()

        # Localization network (aprende transformación)
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor para parámetros de transformación
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 12 * 12, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)  # Affine transform: 2×3 matrix
        )

        # Inicializar con identidad
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # 1. Localizar semáforo
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 12 * 12)

        # 2. Predecir transformación
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # 3. Aplicar transformación (centra semáforo)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x_normalized = F.grid_sample(x, grid, align_corners=False)

        return x_normalized

# Uso:
quad_recognizer_original = Recognizer(quad_pool_params)
quad_recognizer_original.load_state_dict(torch.load('quad.torch'))

# Wrap con normalización
quad_recognizer_normalized = PositionNormalizedRecognizer(quad_recognizer_original)

# Entrenar SOLO el STN (freeze recognizer)
for param in quad_recognizer_normalized.base_recognizer.parameters():
    param.requires_grad = False

# Train STN para aprender a centrar semáforos
optimizer = torch.optim.Adam(quad_recognizer_normalized.stn.parameters(), lr=1e-3)
# ...

```

---

### 6.7 Testing y Validación

```python
# test_position_robustness.py

def test_spatial_invariance():
    """
    Test: Recognizer clasifica igual independiente de posición
    """
    model = load_recognizer('quad_retrained.torch')

    # Imagen base con semáforo verde
    base_image = create_traffic_light_image(color='green', position='center')

    # Test en múltiples posiciones
    positions = [
        ('top-left', -20, -20),
        ('top-right', 20, -20),
        ('center', 0, 0),
        ('bottom-left', -20, 20),
        ('bottom-right', 20, 20)
    ]

    results = []
    for pos_name, shift_x, shift_y in positions:
        # Shift image
        shifted = shift_image(base_image, shift_x, shift_y)

        # Classify
        output = model(shifted)
        pred_class = torch.argmax(output)

        results.append((pos_name, pred_class))
        print(f"{pos_name}: {['BLACK','RED','YELLOW','GREEN'][pred_class]}")

    # Verificar consistencia
    predictions = [r[1] for r in results]
    assert all(p == predictions[0] for p in predictions), \
        f"Inconsistent predictions across positions: {results}"

    print("✅ Model is spatially invariant")

def test_swapping_robustness():
    """
    Test: Swapping ya NO causa clasificación errónea
    """
    model = load_recognizer('quad_retrained.torch')

    # Semáforo verde en posición izquierda
    green_left = create_light_at_position(color='green', x=432, y=176)
    output1 = model(green_left)

    # MISMO semáforo verde en posición derecha
    green_right = create_light_at_position(color='green', x=476, y=175)
    output2 = model(green_right)

    # Deberían clasificar igual
    pred1 = torch.argmax(output1)
    pred2 = torch.argmax(output2)

    assert pred1 == pred2 == 3, f"GREEN classification failed: {pred1}, {pred2}"
    print("✅ Swapping test passed")

def benchmark_position_robustness(model, test_dataset):
    """
    Benchmark: Accuracy en múltiples posiciones
    """
    shifts = [
        (0, 0),      # Original
        (-20, 0),    # Left
        (20, 0),     # Right
        (0, -20),    # Up
        (0, 20),     # Down
        (-10, -10),  # Diagonal
        (10, 10)
    ]

    results = {}

    for shift_x, shift_y in shifts:
        correct = 0
        total = 0

        for img, label in test_dataset:
            shifted = shift_image(img, shift_x, shift_y)
            output = model(shifted)
            pred = torch.argmax(output)

            if pred == label:
                correct += 1
            total += 1

        accuracy = correct / total
        results[(shift_x, shift_y)] = accuracy
        print(f"Shift ({shift_x:3}, {shift_y:3}): {accuracy:.2%}")

    # Calcular varianza (debería ser baja)
    accuracies = list(results.values())
    variance = np.var(accuracies)

    print(f"\nAccuracy variance: {variance:.4f}")
    print(f"Min accuracy: {min(accuracies):.2%}")
    print(f"Max accuracy: {max(accuracies):.2%}")

    # Criterio: Varianza < 0.01, Min accuracy > 90%
    assert variance < 0.01, f"High variance: {variance}"
    assert min(accuracies) > 0.90, f"Low min accuracy: {min(accuracies)}"

    print("✅ Position robustness benchmark passed")

```

---

### 6.8 Roadmap de Implementación

### **Opción A: Re-entrenamiento Completo (Recomendado)**

**Fase 1: Data Collection (2-3 semanas)**

```python
# 1. Expandir dataset con variación espacial
# - Capturar mismo semáforo desde múltiples ángulos
# - Diferentes alturas de cámara
# - Diferentes distancias

# 2. Synthetic data generation
# - Generar crops con offsets aleatorios
# - Rotaciones, escalas, traslaciones

```

**Fase 2: Training (1-2 semanas)**

```bash
# Entrenar con spatial augmentation
python tools/retrain_recognizer.py \
    --dataset augmented_traffic_lights/ \
    --augment spatial \
    --epochs 100 \
    --model quad

# Validar robustez
python tools/validate_position_robustness.py \
    --model quad_retrained.torch

```

**Fase 3: Deployment (1 semana)**

```python
# Reemplazar modelos
quad_recognizer.load_state_dict(torch.load('quad_retrained.torch'))
hori_recognizer.load_state_dict(torch.load('hori_retrained.torch'))
vert_recognizer.load_state_dict(torch.load('vert_retrained.torch'))

```

---

### **Opción B: STN Wrapper (Quick Fix - 1 semana)**

**Día 1-3: Implementar STN**

```python
# Implementar PositionNormalizedRecognizer
# Test unitario con synthetic data

```

**Día 4-5: Train STN**

```python
# Entrenar SOLO el STN (freeze recognizer)
# Dataset: pares (desalineado, alineado)

```

**Día 6-7: Integration y Testing**

```python
# Integrar en pipeline
# Testing de robustez
```

---

## 7. 🗺️ Roadmap de Implementación Completo

### 7.1 Estrategia de Implementación

### **Enfoque: Incremental & Validated**

```
Principio: Cada gap se cierra de forma independiente y se valida antes de continuar

Gap 1 (Selection) → Test → ✅
    ↓
Gap 2 (Multi-detection) → Test → ✅
    ↓
Gap 3 (Dynamic Projections) → Test → ✅
    ↓
Gap 4 (Semantic IDs) → Test → ✅
    ↓
Gap 5 (Spatial Dependency) → Test → ✅
    ↓
Integration Testing → ✅
    ↓
Production Deployment

```

---

### 7.2 Priorización por Impacto

| Gap | Impacto Funcional | Complejidad | Prioridad | Duración Estimada |
| --- | --- | --- | --- | --- |
| **Gap #1: Selection Algorithm** | 🔴 Crítico | 🟡 Media | **P0** | 1-2 semanas |
| **Gap #4: Semantic IDs** | 🟠 Alto | 🟢 Baja | **P0** | 3-5 días |
| **Gap #2: Multi-Detection** | 🔴 Crítico | 🟡 Media | **P1** | 1-2 semanas |
| **Gap #6: Spatial Dependency** | 🔴 Crítico | 🔴 Alta | **P1** | 2-4 semanas |
| **Gap #3: Dynamic Projections** | 🔴 Crítico | 🔴 Muy Alta | **P2** | 1-3 meses |
| **Gap #5: Multi-Camera** | 🟡 Medio | 🔴 Muy Alta | **P3** | 2-3 meses |

---

### 7.3 Roadmap Detallado por Fase

---

## 📅 FASE 1: Quick Wins (Semanas 1-3)

**Objetivo**: Implementar mejoras de alto impacto y baja complejidad

---

### Semana 1: Gap #4 - Semantic IDs

**Día 1-2: Preparación**

```bash
# Crear herramienta de migración
git checkout -b feature/semantic-ids

# Implementar migración de datos
python tools/migrate_to_semantic_ids.py \
    --input projection_bboxes_master.txt \
    --output projection_boxes.json \
    --id-mapping configs/id_mapping.yaml

```

**Día 3-4: Código Core**

```python
# src/tlr/utils.py
- Implementar normalize_boxes_input()
- Testing con formatos antiguos y nuevos

# src/tlr/tracking.py
- Cambiar Dict[int, ...] → Dict[str, ...]
- Actualizar SemanticTable

# src/tlr/selector.py
- Implementar select_tls_with_semantic_ids()

```

**Día 5: Testing**

```bash
pytest tests/test_semantic_ids.py -v
pytest tests/test_backward_compatibility.py -v
pytest tests/test_cross_history_fix.py -v

```

**Entregables**:

- ✅ Sistema acepta ambos formatos (backward compatible)
- ✅ Semantic IDs funcionando en tracking
- ✅ Cross-history transfer eliminado
- ✅ Tests pasando

---

### Semana 2-3: Gap #1 - Selection Algorithm

**Semana 2, Día 1-3: Implementación Apollo Selector**

```python
# src/tlr/apollo_selector.py (NUEVO)
- Implementar SelectionCriteria class
- Implementar ApolloSelector class
- 4 métricas: detection, spatial, shape, temporal

# Tests unitarios
- test_selection_criteria()
- test_score_calculation()

```

**Semana 2, Día 4-5: Integración Pipeline**

```python
# src/tlr/pipeline.py
- Agregar flag use_apollo_selector
- Integrar ApolloSelector en forward()
- Mantener Hungarian para backward compat

```

**Semana 3, Día 1-2: Testing Comparativo**

```python
# tests/test_apollo_vs_hungarian.py
def test_multiple_detections_same_light():
    # Hungarian: Solo asigna 1
    # Apollo: Selecciona mejor
    assert len(apollo_assignments) == 1
    assert apollo_assignments[0] == best_detection

def test_temporal_consistency():
    # Apollo usa temporal_score
    # Hungarian no
    assert apollo_selected_consistent_detection

```

**Semana 3, Día 3-5: Validation & Deployment**

```bash
# Correr pipeline completo con ambos
python run_pipeline_comparison.py \
    --selector hungarian \
    --selector apollo \
    --compare-outputs

# Análisis de resultados
python analyze_selector_performance.py

```

**Entregables**:

- ✅ Apollo Selection Algorithm implementado
- ✅ Tests mostrando mejora vs Hungarian
- ✅ Backward compatible (flag configurable)
- ✅ Documentación de diferencias

---

## 📅 FASE 2: Core Improvements (Semanas 4-7)

**Objetivo**: Cerrar gaps funcionales críticos

---

### Semana 4-5: Gap #2 - Múltiples Detections por ROI

**Semana 4, Día 1-2: Diseño**

```python
# Decidir estrategia:
# Opción A: Split projection boxes (Quick)
# Opción B: Detección iterativa (Medium)
# Opción C: HD-Map integration (completo, pero requiere Fase 3)

# Decisión: Implementar B (detección iterativa)

```

**Semana 4, Día 3-5: Implementación**

```python
# src/tlr/pipeline.py
def detect_multi(self, image, boxes):
    """
    Permite múltiples detections por projection
    Mantiene mapeo det_idx → proj_id
    """
    all_detections = []
    detection_to_projection = []

    for proj_id, box in enumerate(boxes):
        # Detectar en esta ROI
        detections = self.detect_in_roi(image, box)

        for det in detections:
            all_detections.append(det)
            detection_to_projection.append(proj_id)

    # NMS global
    # ...

    return all_detections, detection_to_projection

```

**Semana 5, Día 1-3: Integración con Selection**

```python
# Modificar ApolloSelector para considerar mapeo
def select(self, detections, projections, det_to_proj_map, ...):
    # Solo considerar detections que vienen de ROI correcta
    # ...

```

**Semana 5, Día 4-5: Testing**

```bash
pytest tests/test_multi_detection.py::test_two_lights_same_roi
pytest tests/test_multi_detection.py::test_detection_mapping
pytest tests/test_integration_selection_multi.py

```

**Entregables**:

- ✅ Sistema maneja N detections por ROI
- ✅ Selection algorithm usa mapeo correcto
- ✅ Tests con casos de múltiples semáforos
- ✅ No regresiones en casos simples

---

### Semana 6-7: Gap #6 - Dependencia Espacial (Opción STN)

**Nota**: Re-entrenamiento completo requiere Fase 3. STN es quick fix.

**Semana 6, Día 1-3: Implementar STN**

```python
# src/tlr/recognizer_wrapper.py (NUEVO)
class PositionNormalizedRecognizer(nn.Module):
    def __init__(self, base_recognizer):
        self.base_recognizer = base_recognizer
        self.stn = SpatialTransformerNetwork()

    def forward(self, x):
        x_normalized = self.stn(x)
        return self.base_recognizer(x_normalized)

# Implementar SpatialTransformerNetwork

```

**Semana 6, Día 4-5: Training STN**

```python
# Crear dataset de pares (desalineado, ground_truth)
# Entrenar SOLO STN (freeze recognizer)

python tools/train_stn.py \
    --base-model quad.torch \
    --dataset stn_training_data/ \
    --epochs 50

```

**Semana 7, Día 1-2: Integration**

```python
# src/tlr/pipeline.py
# Reemplazar recognizers con wrapped versions
self.quad_recognizer = PositionNormalizedRecognizer(
    quad_recognizer_base
)

```

**Semana 7, Día 3-5: Validation**

```bash
# Test de robustez espacial
pytest tests/test_position_robustness.py::test_spatial_invariance
pytest tests/test_position_robustness.py::test_swapping_robustness

# Benchmark
python tools/benchmark_position_robustness.py \
    --model quad_stn.torch \
    --test-shifts -30 -20 -10 0 10 20 30

```

**Entregables**:

- ✅ STN funcionando y normalizando posición
- ✅ Swapping test pasa (sin BLACK falso)
- ✅ Accuracy estable en múltiples posiciones
- ✅ Performance no degradada (<10% overhead)

---

## 📅 FASE 3: Apollo-Level Features (Semanas 8-20)

**Objetivo**: Implementar features completas de Apollo

---

### Semana 8-11: Gap #3 - Projection Boxes Dinámicas

**Semana 8: Preparación Infraestructura**

**Día 1-2: HD-Map Creation**

```bash
# Opción 1: Manual (para escenario test)
# - Usar Google Earth para obtener coords GPS
# - Crear hdmap_test.json manualmente

# Opción 2: Semi-automático
# - Capturar video con GPS logger
# - Marcar semáforos manualmente
# - Script genera HD-Map

python tools/create_hdmap.py \
    --video test_video.mp4 \
    --gps-log gps_data.csv \
    --output hdmap_test.json

```

**Día 3-4: Camera Calibration**

```bash
# Calibrar cámara con checkerboard
python tools/calibrate_camera.py \
    --images calibration_images/*.jpg \
    --pattern-size 9x6 \
    --square-size 0.025 \
    --output camera_calib.json

# Validar calibración
python tools/validate_calibration.py \
    --calib camera_calib.json \
    --test-images test_calib/*.jpg

```

**Día 5: Localization Setup**

```python
# Implementar SimulatedLocalizer para testing
# (GPS real requiere hardware adicional)

# src/tlr/localization.py
class SimulatedLocalizer:
    def __init__(self, trajectory_file):
        # Cargar trayectoria pre-grabada
        self.trajectory = load_trajectory(trajectory_file)

    def get_current_pose(self, timestamp):
        # Interpolar pose en timestamp
        return interpolate_pose(self.trajectory, timestamp)

```

**Semana 9-10: Implementación Core**

**Día 1-3: Projector 3D→2D**

```python
# src/tlr/projection_3d_to_2d.py
- Implementar Projector3Dto2D
- Transformaciones world→camera→pixel
- Manejo de distorsión de lente
- Tests unitarios con casos conocidos

```

**Día 4-7: Dynamic Projector**

```python
# src/tlr/dynamic_projector.py
- Implementar DynamicProjector
- Carga de HD-Map
- Cache de semáforos cercanos
- Generación de projection boxes

# Tests
- test_projection_accuracy()
- test_cache_updates()
- test_out_of_view_filtering()

```

**Día 8-10: Pipeline Integration**

```python
# src/tlr/pipeline.py
- Modificar forward() para aceptar vehicle_pose
- Usar semantic_ids del HD-Map
- Backward compatibility con boxes estáticas

```

**Semana 11: Testing & Validation**

```bash
# Test 1: Projection accuracy
python tests/test_dynamic_projections.py::test_projection_vs_ground_truth

# Test 2: Semantic ID persistence
python tests/test_dynamic_projections.py::test_semantic_id_persistence

# Test 3: Cross-history fix
python tests/test_dynamic_projections.py::test_no_cross_history_with_dynamic

# Integration test
python run_pipeline_with_dynamic_projections.py \
    --video test_video.mp4 \
    --hdmap hdmap_test.json \
    --calib camera_calib.json \
    --localization simulated

```

**Entregables**:

- ✅ HD-Map del escenario de prueba
- ✅ Calibración de cámara validada
- ✅ Dynamic projector funcionando
- ✅ Semantic IDs persistentes (no cross-history)
- ✅ Pipeline integrado con pose tracking

---

### Semana 12-16: Gap #6 - Dependencia Espacial (Re-entrenamiento)

**Semana 12-13: Data Collection & Preparation**

**Tarea 1: Expandir Dataset**

```python
# Capturar nuevo data con variación espacial
# - Mismos semáforos desde múltiples ángulos
# - Diferentes alturas de cámara (sedan, SUV, truck)
# - Diferentes distancias (5m, 20m, 50m, 100m)

# Meta: 10,000+ samples con diversidad espacial

```

**Tarea 2: Synthetic Augmentation**

```python
# tools/generate_augmented_dataset.py

def augment_dataset(original_dataset, output_dir):
    for img, label in original_dataset:
        # Generar 20 variaciones por imagen
        for i in range(20):
            # Random spatial transform
            augmented = apply_random_transform(
                img,
                shift_range=(-30, 30),
                rotation_range=(-15, 15),
                scale_range=(0.7, 1.3)
            )

            save(augmented, label, f"{output_dir}/{img_id}_{i}.jpg")

# Resultado: 200,000+ samples augmented

```

**Semana 14-15: Training**

```bash
# Train recognizers con spatial augmentation
for model in quad hori vert; do
    python tools/train_recognizer.py \
        --model $model \
        --dataset augmented_traffic_lights/ \
        --augment spatial \
        --epochs 100 \
        --batch-size 64 \
        --lr 1e-4 \
        --output ${model}_spatially_robust.torch
done

# Validación continua
python tools/validate_during_training.py \
    --watch-dir checkpoints/ \
    --test-suite position_robustness

```

**Semana 16: Validation & Deployment**

```bash
# Benchmark completo
python tools/benchmark_recognizers.py \
    --old-models quad.torch hori.torch vert.torch \
    --new-models quad_robust.torch hori_robust.torch vert_robust.torch \
    --test-suite comprehensive

# Comparison report
# - Accuracy en posiciones originales
# - Accuracy en posiciones shifted
# - Varianza entre posiciones
# - Casos edge (rotaciones extremas)

# Si mejora > 20% en robustez Y mantiene accuracy:
# → Deploy nuevos modelos

```

**Entregables**:

- ✅ Dataset augmented (200K+ samples)
- ✅ Recognizers re-entrenados
- ✅ Benchmark mostrando mejora en robustez
- ✅ Sin degradación en accuracy base
- ✅ Swapping test pasa consistentemente

---

### Semana 17-20: Integration Testing & Refinement

**Semana 17: System Integration**

```bash
# Integrar TODOS los gaps implementados
# Gap #1: Apollo Selection ✅
# Gap #2: Multi-detection ✅
# Gap #3: Dynamic Projections ✅
# Gap #4: Semantic IDs ✅
# Gap #6: Spatial robustness ✅

# Pipeline completo Apollo-equivalent
python run_full_apollo_pipeline.py \
    --video test_suite/*.mp4 \
    --hdmap maps/*.json \
    --config config_apollo_mode.yaml

```

**Semana 18: Performance Optimization**

```python
# Profiling
python -m cProfile run_full_apollo_pipeline.py > profile.txt

# Identificar bottlenecks
# - Selection algorithm: O(N³) → implementar optimizaciones
# - Dynamic projections: cache agresivo
# - STN overhead: considerar TensorRT

# Optimizaciones
- Batch processing donde sea posible
- GPU acceleration para Hungarian (si disponible)
- Compiled models (TorchScript)

```

**Semana 19: Stress Testing**

```bash
# Test 1: Intersecciones complejas (10+ semáforos)
python tests/test_complex_intersection.py

# Test 2: Long videos (1000+ frames)
python tests/test_long_video_memory.py

# Test 3: Edge cases
python tests/test_edge_cases.py
# - Semáforos muy cercanos
# - Oclusiones temporales
# - Cambios de iluminación extremos
# - Movimiento rápido de vehículo

```

**Semana 20: Documentation & Handoff**

```markdown
# Crear documentación completa
docs/
  ├── architecture_apollo_mode.md
  ├── api_reference.md
  ├── configuration_guide.md
  ├── troubleshooting.md
  └── migration_guide.md

# Training materials
training/
  ├── setup_guide.md
  ├── hdmap_creation_tutorial.md
  ├── camera_calibration_guide.md
  └── video_walkthrough.mp4

```

**Entregables Fase 3**:

- ✅ Sistema Apollo-equivalent completo
- ✅ Todos los gaps cerrados
- ✅ Performance optimizado
- ✅ Testing exhaustivo
- ✅ Documentación completa

---

## 📅 FASE 4: Production Readiness (Opcional - Semanas 21-24)

### Semana 21-22: Deployment Infrastructure

**Containerization**

```docker
# Dockerfile
FROM pytorch/pytorch:2.0-cuda11.8

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/

CMD ["python", "run_apollo_pipeline.py"]

```

**Orchestration**

```yaml
# docker-compose.yml
services:
  apollo-tlr:
    build: .
    volumes:
      - ./data:/data
      - ./models:/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - CONFIG_FILE=/configs/production.yaml

```

---

### Semana 23: Monitoring & Logging

```python
# src/tlr/monitoring.py (NUEVO)

import logging
from prometheus_client import Counter, Histogram

# Metrics
frames_processed = Counter('frames_processed_total', 'Total frames')
detection_latency = Histogram('detection_latency_seconds', 'Detection time')
recognition_accuracy = Gauge('recognition_accuracy', 'Accuracy')

class PipelineMonitor:
    def log_frame(self, frame_id, results):
        frames_processed.inc()

        # Log detections
        logging.info(f"Frame {frame_id}: {len(results['detections'])} lights detected")

        # Log anomalies
        if len(results['detections']) == 0:
            logging.warning(f"Frame {frame_id}: No detections")

        if any(r['color'] == 'BLACK' for r in results['recognitions']):
            logging.warning(f"Frame {frame_id}: Unknown color detected")

```

---

### Semana 24: CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: Apollo TLR CI/CD

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src/

      - name: Benchmark performance
        run: |
          python tools/benchmark.py --report ci_report.json

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: ci_report.json

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t apollo-tlr:latest .

      - name: Push to registry
        run: docker push apollo-tlr:latest

```

---

## 8. 📊 Plan de Testing y Validación

### 8.1 Test Pyramid

```
                    ▲
                   / \
                  /   \
                 /  E2E \          10% - End-to-End (1-2 tests)
                /       \
               /_________\
              /           \
             / Integration \      30% - Integration (10-15 tests)
            /               \
           /_________________\
          /                   \
         /    Unit Tests       \   60% - Unit (50+ tests)
        /_______________________\

```

---

### 8.2 Test Suite por Gap

**Gap #1: Selection Algorithm**

```python
tests/test_apollo_selector.py
├── test_selection_criteria_scoring()
├── test_multiple_detections_fusion()
├── test_temporal_consistency_scoring()
├── test_shape_validation()
├── test_vs_hungarian_comparison()
└── test_performance_benchmark()

```

**Gap #2: Multi-Detection**

```python
tests/test_multi_detection.py
├── test_two_lights_same_roi()
├── test_detection_to_projection_mapping()
├── test_nms_across_rois()
└── test_complex_intersection()

```

**Gap #3: Dynamic Projections**

```python
tests/test_dynamic_projections.py
├── test_3d_to_2d_projection()
├── test_projection_accuracy_vs_ground_truth()
├── test_semantic_id_persistence()
├── test_cache_updates()
└── test_vehicle_movement_tracking()

```

**Gap #4: Semantic IDs**

```python
tests/test_semantic_ids.py
├── test_semantic_id_persistence()
├── test_backward_compatibility()
├── test_cross_history_fix()
└── test_tracking_with_semantic_ids()

```

**Gap #6: Spatial Robustness**

```python
tests/test_position_robustness.py
├── test_spatial_invariance()
├── test_swapping_robustness()
├── test_rotation_robustness()
└── benchmark_position_robustness()

```

---

### 8.3 Acceptance Criteria

### **Funcionalidad**

- ✅ Selection algorithm selecciona mejor detection (vs Hungarian)
- ✅ Múltiples detections por ROI manejadas correctamente
- ✅ Projection boxes actualizan dinámicamente con pose
- ✅ Semantic IDs persisten entre frames
- ✅ No cross-history transfer
- ✅ Recognizer robusto a cambios de posición (variance < 1%)

### **Performance**

- ✅ Pipeline completo: <100ms por frame (GPU)
- ✅ Selection algorithm: <10ms
- ✅ Dynamic projection: <5ms
- ✅ Memory usage: <1GB

### **Robustez**

- ✅ Accuracy > 95% en test set original
- ✅ Accuracy > 90% con spatial shifts (±30px)
- ✅ No degradación con cambios de ángulo (±10°)
- ✅ Maneja intersecciones con 10+ semáforos

---

## 9. 🎯 Resumen Ejecutivo para Implementación

### 9.1 Quick Start (Semanas 1-3)

**Si solo tienes 3 semanas, implementa**:

1. **Semantic IDs** (Gap #4) - 5 días
    - Elimina cross-history transfer
    - Bajo riesgo, alto impacto
2. **Apollo Selection** (Gap #1) - 10 días
    - Mejora inmediata en assignment
    - Fusiona múltiples detections

**Resultado**: Sistema 40% más robusto con 3 semanas de trabajo

---

### 9.2 Medium Term (Semanas 1-7)

**Si tienes 2 meses, agrega**: 3. **Multi-Detection** (Gap #2) - 10 días

- Maneja casos complejos
1. **STN Wrapper** (Gap #6 quick fix) - 10 días
    - Mejora robustez espacial

**Resultado**: Sistema 70% Apollo-equivalent

---

### 9.3 Long Term (Semanas 1-20)

**Para sistema completo Apollo-level**: 5. **Dynamic Projections** (Gap #3) - 4 semanas

- Requiere infraestructura (HD-Map, localization)
1. **Re-entrenamiento Recognizer** (Gap #6 completo) - 5 semanas
    - Elimina dependencia espacial completamente

**Resultado**: Sistema 95%+ Apollo-equivalent

---

## 10. 📝 Checklist de Implementación

### Para Ti (Cuando Vuelvas a Trabajar en Esto)

**Antes de empezar:**

```bash
# 1. Revisar este documento completo
# 2. Entender estado actual del código
git log --oneline --graph --all

# 3. Verificar tests baseline
pytest tests/ -v

# 4. Crear branch de trabajo
git checkout -b feature/apollo-gaps-implementation

```

**Durante implementación:**

- [ ]  Implementar un gap a la vez (no mezclar)
- [ ]  Escribir tests ANTES de código (TDD)
- [ ]  Validar cada gap antes de continuar
- [ ]  Documentar decisiones en commits
- [ ]  Mantener backward compatibility

**Después de cada gap:**

- [ ]  Tests unitarios pasan
- [ ]  Tests de integración pasan
- [ ]  Performance no degradado
- [ ]  Documentación actualizada
- [ ]  Code review (si aplica)

---

## 11. 📚 Referencias y Recursos

### Papers y Documentación

- **Apollo Platform**: https://github.com/ApolloAuto/apollo
- **Traffic Light Detection Paper**: "Apollo: An Open Autonomous Driving Platform" (Baidu Research)
- **Hungarian Algorithm**: Kuhn-Munkres algorithm explanation
- **Spatial Transformer Networks**: Jaderberg et al., 2015

### Herramientas Recomendadas

- **HD-Map Creation**: Apollo Studio, JOSM (OpenStreetMap editor)
- **Camera Calibration**: Kalibr, OpenCV calibration tool
- **Profiling**: PyTorch Profiler, cProfile
- **Visualization**: TensorBoard, Weights & Biases

### Datasets Útiles

- **LISA Traffic Light Dataset**: Labeled images
- **Bosch Small Traffic Lights Dataset**: Multiple scenarios
- **Synthetic Data**: CARLA simulator