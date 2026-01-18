# Infraestructura del Sistema Apollo TLR

## 1. Visión General

El sistema Apollo Traffic Light Recognition (TLR) es un pipeline de detección y reconocimiento de semáforos desarrollado por Baidu para vehículos autónomos. Procesa cada frame de video en **5 etapas secuenciales**:

```
Frame de Cámara + Pose del Vehículo + HD-Map
                    │
                    ▼
        ┌───────────────────────┐
        │  1. PREPROCESAMIENTO  │  Query HD-Map → Proyección 3D→2D
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │     2. DETECCIÓN      │  CNN (SSD) encuentra semáforos en ROIs
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │     3. ASIGNACIÓN     │  Hungarian Algorithm: detections → HD-Map
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │   4. RECONOCIMIENTO   │  CNN clasifica color (R/Y/G/Black)
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │      5. TRACKING      │  Consistencia temporal + reglas de seguridad
        └───────────┬───────────┘
                    ▼
            Resultado Final
```

El diseño clave de Apollo es que **los semáforos tienen identidad persistente** gracias al HD-Map: cada semáforo físico del mundo real tiene un ID único que se mantiene frame a frame.

---

## 2. Conceptos Fundamentales

### 2.1 HD-Map y Signals

El **HD-Map** (High-Definition Map) es un mapa de alta precisión (±5cm) que contiene información de todos los elementos de la vía. Para semáforos, almacena objetos llamados **Signals**:

```cpp
Signal {
  id: "signal_12345"              // ID único del semáforo
  boundary: {                     // Contorno 3D del semáforo (4 puntos)
    point[0]: {x: 500.23, y: 1200.45, z: 5.12}   // Esquina inferior izquierda
    point[1]: {x: 500.28, y: 1200.50, z: 5.12}   // Esquina inferior derecha
    point[2]: {x: 500.28, y: 1200.50, z: 5.92}   // Esquina superior derecha
    point[3]: {x: 500.23, y: 1200.45, z: 5.92}   // Esquina superior izquierda
  }
  type: MIX_3_VERTICAL            // Tipo de semáforo
  stop_line: [...]                // Línea de parada asociada
  overlap_id: ["lane_123"]        // Conexión con carriles
}
```

Las coordenadas `boundary` están en el **sistema de coordenadas mundial** (metros absolutos). El sistema necesita proyectarlas a coordenadas 2D de la imagen usando la pose del vehículo y la calibración de la cámara.

**Flujo de consulta al HD-Map:**
1. El vehículo conoce su posición (GPS + IMU + odometría)
2. Consulta: "Dame todos los semáforos en un radio de 150 metros"
3. El HD-Map retorna una lista de Signals con sus coordenadas 3D

---

### 2.2 ROI y Projection Boxes

Una **ROI** (Region of Interest) es una región rectangular de la imagen donde el sistema busca semáforos. Apollo genera ROIs a partir de las coordenadas 3D del HD-Map:

**Proceso de generación:**

```
Coordenadas 3D del Signal (HD-Map)
            │
            ▼
    Transformación mundo → cámara (usando pose del vehículo)
            │
            ▼
    Proyección 3D → 2D (usando calibración de cámara)
            │
            ▼
    Bounding box 2D = projection_roi
            │
            ▼
    Expansión 2.5× = crop_roi (para compensar imprecisiones)
```

**¿Por qué la expansión 2.5×?**

Las proyecciones del HD-Map pueden tener errores por:
- Imprecisión del GPS (~10cm)
- Errores de calibración de cámara
- Movimiento del vehículo entre frames

Apollo expande la ROI 2.5 veces para asegurar que el semáforo real esté contenido:

```python
# Ejemplo de expansión
projection_roi = [850, 300, 40, 80]   # [x, y, width, height]

center_x = 850 + 40/2 = 870
center_y = 300 + 80/2 = 340
max_dim = max(40, 80) = 80
resize = max_dim × 2.5 = 200
resize = max(resize, 270) = 270       # Mínimo 270px (input del detector)

crop_roi = [870-135, 340-135, 270, 270] = [735, 205, 270, 270]
```

---

### 2.3 Modelos de Machine Learning

Apollo utiliza **4 modelos** de redes neuronales pre-entrenados:

| Modelo | Archivo | Propósito | Input | Output |
|--------|---------|-----------|-------|--------|
| **Detector** | `tl.torch` | Detectar semáforos en ROIs | 270×270×3 | Bboxes + tipo |
| **Recognizer Vertical** | `vert.torch` | Clasificar color (semáforos verticales) | 96×32×3 | [Black, R, Y, G] |
| **Recognizer Horizontal** | `hori.torch` | Clasificar color (semáforos horizontales) | 32×96×3 | [Black, R, Y, G] |
| **Recognizer Quad** | `quad.torch` | Clasificar color (semáforos 2×2) | 64×64×3 | [Black, R, Y, G] |

**Detector (SSD-style):**
- Arquitectura: Single Shot Multibox Detector adaptado
- Entrada: Imagen 270×270 (crop de la ROI)
- Salida: Lista de detecciones, cada una con:
  - Bounding box `[x1, y1, x2, y2]`
  - Scores por tipo: `[background, vertical, quad, horizontal]`

**Recognizers (CNN):**
- Arquitectura: CNN con 5 capas convolucionales + pooling + FC
- Entrada: Crop del semáforo detectado, resizeado según tipo
- Salida: Probabilidades `[Black, Red, Yellow, Green]`

---

## 3. Las 5 Etapas del Pipeline

### 3.1 Etapa 1: Preprocesamiento (Region Proposal)

**Archivo fuente:** `traffic_light_region_proposal_component.cc`, `tl_preprocessor.cc`

**Entradas:**
- Frame de la cámara (1920×1080)
- Pose del vehículo (posición + orientación)
- HD-Map de la zona

**Proceso:**

1. **Query al HD-Map:** Obtener todos los Signals en radio de 150m
2. **Crear objetos TrafficLight:** Para cada Signal, crear un objeto con su ID y coordenadas 3D
3. **Proyección 3D→2D:** Transformar coordenadas mundo a píxeles de imagen
4. **Selección de cámara:** Si hay múltiples cámaras (telephoto 25mm, wide 6mm), elegir la óptima
5. **Validación:** Descartar proyecciones fuera de la imagen

**Salida:** Lista de objetos `TrafficLight` con `projection_roi` calculado

```cpp
TrafficLight {
  id: "signal_12345"                    // Del HD-Map
  region.points: [(x,y,z), ...]         // Coordenadas 3D originales
  region.projection_roi: [850, 300, 40, 80]  // Proyección 2D calculada
  region.outside_image: false

  // Campos vacíos (se llenan en etapas siguientes)
  region.detection_roi: [0, 0, 0, 0]
  region.is_detected: false
  status.color: UNKNOWN
}
```

---

### 3.2 Etapa 2: Detección

**Archivo fuente:** `detection.cc`

**Entradas:**
- Lista de TrafficLight con projection_roi
- Imagen de la cámara

**Proceso:**

Para cada TrafficLight (procesados en serie, NO en batch):

1. **Expandir ROI:** `crop_roi = projection_roi × 2.5` (mínimo 270×270)
2. **Recortar imagen:** Extraer la región `crop_roi`
3. **Resize:** Escalar a 270×270 si es necesario
4. **Inferencia CNN:** Ejecutar detector SSD
5. **Procesar detecciones:**
   - Filtrar background (score[0] > otros)
   - Determinar tipo (vertical/quad/horizontal)
   - Transformar coordenadas al sistema de imagen original
6. **Acumular:** Agregar detecciones válidas a buffer global

7. **NMS Global:** Aplicar Non-Maximum Suppression (IoU threshold = 0.6)

**Salida:** Buffer `detected_bboxes` con N detecciones post-NMS

```cpp
// Cada detection es un TrafficLight SIN identidad del HD-Map
TrafficLight {
  region.detection_roi: [845, 280, 35, 65]    // Bbox en imagen original
  region.detect_class_id: TL_VERTICAL_CLASS   // Tipo detectado
  region.detect_score: 0.92                   // Confianza
  region.is_detected: true

  // NO tienen ID del HD-Map (son outputs puros de la CNN)
  id: ""
}
```

**Relación clave:** De M projection boxes → N detections (N puede ser >, =, o < que M)

---

### 3.3 Etapa 3: Asignación (Hungarian Algorithm)

**Archivo fuente:** `select.cc`, `hungarian_optimizer.h`

**Entradas:**
- `hdmap_bboxes`: M semáforos del HD-Map (con identidad)
- `detected_bboxes`: N detecciones de la CNN (sin identidad)

**Problema a resolver:** ¿Cómo asociar de forma óptima las detecciones a los semáforos del HD-Map?

**Proceso:**

1. **Construir matriz de costos (M×N):**

Para cada par (hdmap[i], detection[j]):

```python
# Score de distancia (Gaussian 2D)
dx = center_detection.x - center_projection.x
dy = center_detection.y - center_projection.y
distance_score = exp(-0.5 × ((dx/100)² + (dy/100)²))

# Score de confianza (clipped a 0.9)
detection_score = min(detect_score, 0.9)

# Score combinado (70% distancia, 30% confianza)
cost[i,j] = 0.7 × distance_score + 0.3 × detection_score
```

2. **Validación espacial:**
   - Si detection está fuera del crop_roi → `cost[i,j] = 0`

3. **Ejecutar Hungarian Algorithm:**
   - Encuentra asignación óptima 1-to-1 que maximiza suma total de scores

**Salida:** Lista de pares `(hdmap_idx, detection_idx)`

```python
assignments = [(0, 1), (1, 3), (2, 4), (3, 6), ...]
# hdmap[0] ← detection[1]
# hdmap[1] ← detection[3]
# etc.
```

**Caso especial:** Si hay más detecciones que semáforos HD-Map, las detecciones extra quedan sin asignar.

---

### 3.4 Etapa 4: Reconocimiento

**Archivo fuente:** `classify.cc`

**Entradas:**
- TrafficLights con detecciones asignadas
- Imagen de la cámara

**Proceso:**

Para cada TrafficLight con `is_detected = true`:

1. **Seleccionar recognizer:** Según `detect_class_id`
   - VERTICAL → `vert.torch` (input 96×32)
   - HORIZONTAL → `hori.torch` (input 32×96)
   - QUAD → `quad.torch` (input 64×64)

2. **Preprocesar:**
   - Recortar `detection_roi` de la imagen
   - Resize a dimensiones del recognizer
   - Restar means y escalar (`× 0.01`)

3. **Inferencia CNN:**
   - Output: probabilidades `[Black, Red, Yellow, Green]`

4. **Clasificar (Prob2Color):**
```cpp
max_prob, max_idx = argmax(probabilities)
if (max_prob > 0.5) {
    color = [BLACK, RED, YELLOW, GREEN][max_idx]
} else {
    color = BLACK  // Desconocido
}
```

**Salida:** TrafficLights con `status.color` y `status.confidence` asignados

---

### 3.5 Etapa 5: Tracking Temporal

**Archivo fuente:** `semantic_decision.cc`

**Entradas:**
- TrafficLights con color reconocido
- Historial de frames anteriores

**Proceso:**

El tracking aplica **consistencia temporal** y **reglas de seguridad**:

1. **Histéresis:**
   - Para cambiar de estado, necesita N frames consecutivos con el nuevo color
   - Evita parpadeos por errores puntuales del recognizer

2. **Detección de Blink:**
   - Si el color cambia en menos de 0.55 segundos → es parpadeo
   - Parpadeo de amarillo → forzar a ROJO (seguridad)

3. **Reglas de transición:**
   - YELLOW después de RED → mantener RED (transición inválida)
   - Secuencia válida: RED → GREEN → YELLOW → RED

4. **Ventana temporal:**
   - Considera últimos 1.5 segundos de historia
   - Reset de histéresis si pasa mucho tiempo sin ver el semáforo

**Salida:** TrafficLights con `status.color` revisado y `status.blink` flag

---

## 4. Flujo de Datos y Estados del Objeto TrafficLight

El objeto `TrafficLight` evoluciona a través de las etapas:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESPUÉS DE ETAPA 1 (Preprocesamiento)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  id: "signal_12345"          ← Del HD-Map                                    │
│  region.points: [(x,y,z)...] ← Coordenadas 3D del boundary                   │
│  region.projection_roi: [850, 300, 40, 80]  ← Proyección 2D calculada        │
│  region.outside_image: false                                                 │
│                                                                              │
│  region.detection_roi: [0,0,0,0]   ← Vacío                                   │
│  region.is_detected: false         ← Vacío                                   │
│  status.color: UNKNOWN             ← Vacío                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESPUÉS DE ETAPA 2 (Detección)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  Buffer detected_bboxes contiene N detecciones SIN id:                       │
│                                                                              │
│  TrafficLight {                                                              │
│    id: ""                          ← Sin identidad                           │
│    region.detection_roi: [845, 280, 35, 65]  ← Bbox detectada                │
│    region.detect_class_id: VERTICAL          ← Tipo                          │
│    region.detect_score: 0.92                 ← Confianza                     │
│    region.is_detected: true                                                  │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESPUÉS DE ETAPA 3 (Asignación)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Los campos de detección se COPIAN al objeto hdmap:                          │
│                                                                              │
│  id: "signal_12345"                ← Mantiene identidad HD-Map               │
│  region.projection_roi: [850, 300, 40, 80]                                   │
│  region.detection_roi: [845, 280, 35, 65]   ← Copiado de la detection        │
│  region.detect_class_id: VERTICAL           ← Copiado                        │
│  region.detect_score: 0.92                  ← Copiado                        │
│  region.is_detected: true                   ← Ahora true                     │
│                                                                              │
│  status.color: UNKNOWN             ← Aún vacío                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESPUÉS DE ETAPA 4 (Reconocimiento)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  id: "signal_12345"                                                          │
│  region.detection_roi: [845, 280, 35, 65]                                    │
│  region.detect_class_id: VERTICAL                                            │
│  region.is_detected: true                                                    │
│                                                                              │
│  status.color: GREEN               ← Clasificado por CNN                     │
│  status.confidence: 0.94           ← Probabilidad del color                  │
│  status.blink: false               ← Aún no evaluado                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESPUÉS DE ETAPA 5 (Tracking)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  id: "signal_12345"                                                          │
│  region.detection_roi: [845, 280, 35, 65]                                    │
│  region.is_detected: true                                                    │
│                                                                              │
│  status.color: GREEN               ← Puede ser corregido por tracking        │
│  status.confidence: 0.94                                                     │
│  status.blink: false               ← Evaluado con historial temporal         │
│                                                                              │
│  ✅ RESULTADO FINAL: Semáforo signal_12345 está en GREEN                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Notas Adicionales

### 5.1 Semantic ID: Diseñado pero No Implementado

El código de Apollo contiene referencias a un campo `semantic_id` que **conceptualmente** debería agrupar semáforos del mismo cruce:

```cpp
// En traffic_light_region_proposal_component.cc:335
int cur_semantic = 0;  // ← SIEMPRE 0, hardcodeado
light->semantic = cur_semantic;
```

**Diseño teórico:**
- Semáforos del mismo cruce compartirían `semantic_id`
- Permitiría voting (si 2 de 3 semáforos del cruce son GREEN, el tercero también)
- El tracking podría aplicar consistencia por grupo

**Realidad:**
- El campo `semantic_id` NO existe en el proto `Signal` del HD-Map
- El código SIEMPRE asigna `semantic = 0` a todos los semáforos
- La lógica de voting por grupos existe pero nunca se ejecuta

**Impacto:** Cada semáforo se procesa de forma completamente independiente, sin considerar que múltiples semáforos del mismo cruce deberían mostrar el mismo color.

---

### 5.2 Sistema Multi-Cámara

Apollo soporta múltiples cámaras con diferentes características:

| Cámara | Focal | FOV | Uso |
|--------|-------|-----|-----|
| Telephoto | 25mm | ~30° | Semáforos lejanos (>50m), mayor resolución |
| Wide-angle | 6mm | ~120° | Semáforos cercanos, campo amplio |

**Lógica de selección:**
1. Intenta usar telephoto (mejor resolución)
2. Si alguna proyección queda fuera o muy cerca del borde → usa wide-angle
3. Siempre prioriza la cámara de mayor focal que contenga todas las proyecciones

---

### 5.3 Parámetros Clave del Sistema

| Parámetro | Valor | Ubicación |
|-----------|-------|-----------|
| Radio de query HD-Map | 150m | `region_proposal_component.cc` |
| Factor de expansión ROI | 2.5× | `cropbox.cc` |
| Tamaño mínimo de crop | 270px | `cropbox.cc` |
| Threshold NMS | 0.6 IoU | `detection.cc` |
| Threshold reconocimiento | 0.5 | `classify.cc` |
| Peso distancia (asignación) | 0.7 | `select.cc` |
| Peso confianza (asignación) | 0.3 | `select.cc` |
| Ventana temporal tracking | 1.5s | `semantic_decision.cc` |
| Threshold blink | 0.55s | `semantic_decision.cc` |
