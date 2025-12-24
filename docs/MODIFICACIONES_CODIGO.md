# Modificaciones de Código Necesarias

**Fecha:** 2024-12-23
**Objetivo:** Implementar mejoras de robustez identificadas en análisis comparativo con Apollo

---

## 📋 Resumen Ejecutivo

Este documento describe las **3 modificaciones críticas** necesarias para:
1. Soportar casos de test de robustez
2. Reducir falsos positivos
3. Alinear comportamiento con Apollo

**Tiempo estimado total:** 2-3 horas

---

## 🔴 MODIFICACIÓN 1: Agregar `signal_id` estable

### **Problema actual:**

Projection boxes usan ID temporal que cambia entre frames:

```
# frames_auto_labeled/projection_bboxes_master.txt (ACTUAL)
frame_0000.jpg,466,181,504,256,0  ← ID = 0 (temporal)
frame_0001.jpg,468,186,500,256,0  ← ID = 0 (puede ser OTRO semáforo)
```

Esto impide:
- Rastrear el mismo semáforo físico entre frames
- Detectar cross-history transfer
- Tracking temporal robusto

### **Solución:**

Agregar `signal_id` estable (como HD-Map de Apollo):

```
# Formato NUEVO:
frame_0000.jpg,466,181,504,256,proj_0,signal_001
frame_0001.jpg,468,186,500,256,proj_0,signal_001  ← Mismo signal_id
```

**Campos:**
- `proj_0`: ID de projection box en este frame (temporal, puede cambiar)
- `signal_001`: ID del semáforo físico (permanente, NO cambia)

### **Archivos a modificar:**

#### **1. Formato de projection boxes**

**Script de conversión (crear nuevo):**

```python
# convert_projection_boxes.py
import os

def convert_file(input_file, output_file):
    """
    Convierte formato viejo a nuevo agregando signal_id.

    Viejo: frame,x1,y1,x2,y2,id
    Nuevo: frame,x1,y1,x2,y2,proj_id,signal_id
    """
    signal_counter = {}  # Para generar signal_ids únicos

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(',')
            frame, x1, y1, x2, y2, old_id = parts

            # Generar signal_id basado en posición aproximada
            # (semáforos reales se mueven poco entre frames)
            key = f"{int(x1)//50}_{int(y1)//50}"  # Grid 50px

            if key not in signal_counter:
                signal_counter[key] = len(signal_counter) + 1

            signal_id = f"signal_{signal_counter[key]:03d}"
            proj_id = f"proj_{old_id}"

            # Escribir nuevo formato
            f_out.write(f"{frame},{x1},{y1},{x2},{y2},{proj_id},{signal_id}\n")

# Convertir todos los archivos
convert_file(
    'frames_auto_labeled/projection_bboxes_master.txt',
    'frames_auto_labeled/projection_bboxes_master_new.txt'
)
```

#### **2. Parser de projection boxes (utils.py o donde se lea)**

**Ubicación:** Donde se carguen las projection boxes

**Modificar de:**
```python
# ACTUAL (asumiendo que existe algo así)
def load_projection_boxes(file_path):
    boxes = []
    with open(file_path) as f:
        for line in f:
            frame, x1, y1, x2, y2, box_id = line.strip().split(',')
            boxes.append({
                'frame': frame,
                'coords': [int(x1), int(y1), int(x2), int(y2)],
                'id': int(box_id)
            })
    return boxes
```

**A:**
```python
def load_projection_boxes(file_path):
    boxes = []
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split(',')

            # Soportar ambos formatos (viejo y nuevo)
            if len(parts) == 6:
                # Formato viejo: frame,x1,y1,x2,y2,id
                frame, x1, y1, x2, y2, box_id = parts
                signal_id = f"signal_{box_id}"  # Fallback
                proj_id = box_id
            elif len(parts) == 7:
                # Formato nuevo: frame,x1,y1,x2,y2,proj_id,signal_id
                frame, x1, y1, x2, y2, proj_id, signal_id = parts
            else:
                raise ValueError(f"Invalid format: {line}")

            boxes.append({
                'frame': frame,
                'coords': [int(x1), int(y1), int(x2), int(y2)],
                'proj_id': proj_id,
                'signal_id': signal_id  # NUEVO
            })
    return boxes
```

#### **3. Tracking (tracking.py)**

**Línea 73-74:**

**De:**
```python
if proj_id not in self.history:
    self.history[proj_id] = SemanticTable(proj_id, frame_ts, color)
st = self.history[proj_id]
```

**A:**
```python
# Usar signal_id en lugar de proj_id para tracking
if signal_id not in self.history:
    self.history[signal_id] = SemanticTable(signal_id, frame_ts, color)
st = self.history[signal_id]
```

**Modificar firma del método `update()`:**

**De:**
```python
def update(self,
           frame_ts: float,
           assignments: List[Tuple[int,int]],
           recognitions: List[List[float]]
           ) -> Dict[int, Tuple[str,bool]]:
```

**A:**
```python
def update(self,
           frame_ts: float,
           assignments: List[Tuple[int,int]],
           recognitions: List[List[float]],
           signal_ids: Dict[int, str]  # NUEVO: proj_id → signal_id
           ) -> Dict[str, Tuple[str,bool]]:  # NUEVO: retorna por signal_id
    """
    :param signal_ids: mapeo de proj_id → signal_id
    :returns: dict {signal_id: (revised_color, blink_flag)}
    """
```

**Actualizar loop (línea 66):**

```python
for proj_id, det_idx in assignments:
    # Obtener signal_id de este proj_id
    signal_id = signal_ids.get(proj_id, f"unknown_{proj_id}")

    # decidir color actual
    cls = int(max(range(len(recognitions[det_idx])),
                  key=lambda i: recognitions[det_idx][i]))
    color = ["black","red","yellow","green"][cls]

    # obtener o crear estado histórico POR SIGNAL_ID
    if signal_id not in self.history:
        self.history[signal_id] = SemanticTable(signal_id, frame_ts, color)
    st = self.history[signal_id]

    # ... resto igual ...

    results[signal_id] = (st.color, st.blink)  # Usar signal_id
```

#### **4. Pipeline (pipeline.py)**

**Línea 144: Pasar signal_ids al tracker**

**De:**
```python
assigns_list = assignments.cpu().tolist()
recs_list    = recognitions.cpu().tolist()
revised = self.tracker.track(frame_ts, assigns_list, recs_list)
```

**A:**
```python
assigns_list = assignments.cpu().tolist()
recs_list    = recognitions.cpu().tolist()

# NUEVO: Crear mapeo proj_id → signal_id
# (Asumiendo que boxes tiene signal_id, ajustar según tu código)
signal_ids = {i: box.signal_id for i, box in enumerate(boxes)}

revised = self.tracker.track(frame_ts, assigns_list, recs_list, signal_ids)
```

### **Impacto:**
- ✅ Permite rastrear semáforos físicos entre frames
- ✅ Detecta cross-history transfer
- ✅ Soporta Caso 3 (Projection Box Staleness)

### **Tiempo estimado:** 1-2 horas

---

## 🟡 MODIFICACIÓN 2: Confidence Threshold

### **Problema actual:**

Detecciones con confianza muy baja (score < 0.3) pasan al Hungarian algorithm:

```python
# pipeline.py línea 122-126
tl_types = torch.argmax(detections[:, 5:], dim=1)
valid_mask = tl_types != 0  # Solo filtra "background"
valid_detections = detections[valid_mask]
# ❌ Detecciones con score=0.05 PASAN
```

**Resultado:** Falsos positivos (luces traseras, reflejos) con score bajo se asignan.

### **Solución:**

Filtrar detecciones con `score < 0.3` (como Apollo):

```python
# pipeline.py - DESPUÉS de línea 119
detections = self.detect(img, boxes)

# NUEVO: Filtrar por confidence ANTES de filtrar por tipo
MIN_CONFIDENCE = 0.3
confidence_mask = detections[:, 0] >= MIN_CONFIDENCE
detections = detections[confidence_mask]

# Ahora sí filtrar por tipo
tl_types = torch.argmax(detections[:, 5:], dim=1)
valid_mask = tl_types != 0
valid_detections = detections[valid_mask]
invalid_detections = detections[~valid_mask]
```

### **Código completo modificado:**

```python
def forward(self, img, boxes, frame_ts=None):
    # ... código existente hasta línea 119 ...

    # 2) Detección
    detections = self.detect(img, boxes)

    # NUEVO: Filtro de confidence (Apollo-style)
    MIN_CONFIDENCE = 0.3
    if len(detections) > 0:
        confidence_mask = detections[:, 0] >= MIN_CONFIDENCE
        detections = detections[confidence_mask]

    # 3) Filtrado por tipo y asignación
    if len(detections) > 0:
        tl_types = torch.argmax(detections[:, 5:], dim=1)
        valid_mask = tl_types != 0
        valid_detections = detections[valid_mask]
        invalid_detections = detections[~valid_mask]
    else:
        # Sin detecciones después de filtro
        tl_types = torch.empty(0, dtype=torch.long, device=self.device)
        valid_detections = torch.empty((0, 9), device=self.device)
        invalid_detections = torch.empty((0, 9), device=self.device)

    # ... resto igual ...
```

### **Impacto:**
- ✅ Reduce falsos positivos ~30-40%
- ✅ Soporta Caso 2 (High-Confidence False Positive)
- ✅ Alineado con Apollo

### **Tiempo estimado:** 5-10 minutos

---

## 🟢 MODIFICACIÓN 3: Validación de tamaño de detecciones (OPCIONAL)

### **Problema actual:**

Detecciones absurdamente grandes (>300px) o pequeñas (<5px) pasan:

```python
# pipeline.py línea 47
detections = detections_sorted[idxs]
return detections
# ❌ Detecciones de 500×400px o 3×2px PASAN
```

### **Solución:**

Validar tamaño después de NMS:

```python
# pipeline.py - Modificar método detect(), DESPUÉS de línea 47

idxs = nms(detections_sorted[:, 1:5], 0.6)
detections = detections_sorted[idxs]

# NUEVO: Validar tamaño de detecciones (Apollo-style)
MIN_SIZE = 5
MAX_SIZE = 300
MIN_ASPECT = 0.5
MAX_ASPECT = 8.0

valid_mask = torch.ones(len(detections), dtype=torch.bool, device=detections.device)

for i, det in enumerate(detections):
    w = det[3] - det[1]  # xmax - xmin
    h = det[4] - det[2]  # ymax - ymin

    # Tamaño válido
    if w < MIN_SIZE or h < MIN_SIZE or w > MAX_SIZE or h > MAX_SIZE:
        valid_mask[i] = False
        continue

    # Aspect ratio válido
    aspect = h / w if w > 0 else 0
    if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
        valid_mask[i] = False

detections = detections[valid_mask]

return detections
```

### **Impacto:**
- ✅ Rechaza detecciones absurdas (edificios, ruido)
- ✅ Mejora robustez del sistema
- ⚠️ Opcional (no crítico para casos de test)

### **Tiempo estimado:** 15-20 minutos

---

## 📊 Tabla Resumen

| Modificación | Archivo | Líneas | Tiempo | Impacto | Prioridad |
|--------------|---------|--------|--------|---------|-----------|
| **1. signal_id** | tracking.py, pipeline.py, utils | ~30 | 1-2h | MUY ALTO | 🔴 CRÍTICO |
| **2. Confidence threshold** | pipeline.py | ~10 | 5-10min | ALTO | 🔴 CRÍTICO |
| **3. Validación tamaño** | pipeline.py | ~20 | 15-20min | MEDIO | 🟢 OPCIONAL |

---

## 🎯 Plan de Implementación Recomendado

### **Fase 1: Crítico (1-2h total)**
1. ✅ Modificación 2 (Confidence threshold) - 5 min
2. ✅ Modificación 1 (signal_id) - 1-2h

### **Fase 2: Opcional (15-20 min)**
3. ⚠️ Modificación 3 (Validación tamaño) - Implementar si sobra tiempo

---

## 🔍 Testing

Después de cada modificación, ejecutar:

```bash
# Test básico
python example.py

# Test con tracking
python example_with_tracking.py

# Verificar que signal_id se propaga correctamente
# (agregar prints temporales en tracking.py)
```

---

## 📚 Referencias

- **Apollo código:** `detection.cc:368-375` (confidence), `select.cc:76-83` (validación)
- **Documento FALTANTES:** `docs/FALTANTES_EN_NUESTRO_SISTEMA.md`
- **Papers relevantes:** Caso C (GPS Degradation), arXiv 2024 (projection errors)

---

**Última actualización:** 2024-12-23
