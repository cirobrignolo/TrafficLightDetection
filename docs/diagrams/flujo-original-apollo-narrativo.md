# Flujo Original de Apollo Traffic Light Detection - Explicación Narrativa Detallada

## Visión General

El sistema Apollo procesa cada frame de video en 5 etapas secuenciales. Imaginemos que el vehículo está circulando y llega un nuevo frame de la cámara. Veamos qué sucede paso a paso.

---

## 🗺️ CONCEPTOS CLAVE: HD-MAP Y SEMANTIC IDS

### ¿Qué es el HD-Map?

El HD-Map (High-Definition Map) es un mapa de alta precisión que contiene información detallada de cada elemento de la vía:
- **Precisión**: Centimétrica (±5cm)
- **Contenido**: Carriles, señales, semáforos, líneas de stop, cruces, etc.
- **Formato 3D**: Cada elemento tiene coordenadas exactas en el mundo real (x, y, z en metros)

Para semáforos, el HD-Map almacena:
```cpp
Signal {
  id: "signal_12345"               // ID único del semáforo individual
  semantic_id: 100                  // ID de grupo (varios semáforos comparten)
  boundary: {                       // Contorno 3D del semáforo
    point[0]: {x: 500.23, y: 1200.45, z: 5.12}
    point[1]: {x: 500.28, y: 1200.50, z: 5.12}
    point[2]: {x: 500.28, y: 1200.50, z: 5.92}
    point[3]: {x: 500.23, y: 1200.45, z: 5.92}
  }
  stop_line: ...                    // Línea de parada asociada
}
```

### ⚠️ ¿Qué es el Semantic ID? (NOTA: NO IMPLEMENTADO EN APOLLO)

**IMPORTANTE**: El `semantic_id` está **DISEÑADO en el código pero NO IMPLEMENTADO** en Apollo. Conceptualmente debería agrupar semáforos del mismo cruce, pero en la práctica siempre vale `0`.

**Cómo DEBERÍA funcionar (diseño teórico)**:

El `semantic_id` agrupa semáforos que pertenecen al **mismo cruce o intersección** y que están **funcionalmente relacionados**.

**Ejemplo teórico de un cruce:**

```
Intersección Main St. y 5th Ave (EJEMPLO TEÓRICO):

┌─────────────────────────────────────┐
│  Semáforo A (vehicular Norte)      │
│    - id: "signal_12345"             │
│    - semantic_id: 100    ┌────┐     │  ← En teoría, debería ser 100
│    - (x,y,z): (500, 1200, 5) │ 🚦│    │  ← En práctica, SIEMPRE es 0
│                          └────┘     │
├─────────────────────────────────────┤
│  Semáforo B (vehicular Sur)        │
│    - id: "signal_12346"             │
│    - semantic_id: 100    ┌────┐     │  ← En teoría, debería ser 100
│    - (x,y,z): (502, 1198, 5) │ 🚦│    │  ← En práctica, SIEMPRE es 0
│                          └────┘     │
├─────────────────────────────────────┤
│  Semáforo C (vehicular Este)       │
│    - id: "signal_12347"             │
│    - semantic_id: 100    ┌────┐     │  ← En teoría, debería ser 100
│    - (x,y,z): (498, 1202, 5) │ 🚦│    │  ← En práctica, SIEMPRE es 0
│                          └────┘     │
└─────────────────────────────────────┘
```

**Realidad en Apollo:**
- ❌ El HD-Map proto `Signal` **NO tiene** campo `semantic_id`
- ❌ El código SIEMPRE asigna `semantic = 0` (línea 335 de `traffic_light_region_proposal_component.cc`)
- ❌ La documentación oficial NO menciona semantic grouping
- ✅ El HD-Map SÍ tiene `overlap_id` (para conectar signals con lanes), pero es diferente
- ✅ El código de tracking SÍ tiene lógica para voting por grupos, pero nunca se usa

**¿Por qué SERÍA útil si estuviera implementado?**

1. **Voting**: Si detecto A=GREEN, B=GREEN, C=BLACK → por mayoría, corrijo C a GREEN
2. **Consistencia temporal**: El grupo compartiría un historial, no cada semáforo individual
3. **Reglas de tránsito**: Todos los semáforos vehiculares del mismo cruce cambiarían de forma coordinada

**Ver ETAPA 5: TRACKING** para más detalles sobre el impacto de esta NO-implementación.

---

## 🔷 ETAPA 1: PREPROCESAMIENTO (Region Proposal)

### ¿Qué recibe?

El sistema recibe cuatro entradas fundamentales:

**1. Imagen de la cámara**
- Frame capturado: 1920×1080 píxeles
- Formato: RGB
- Timestamp sincronizado con GPS

**2. Pose del vehículo**
- Posición GPS: (latitud, longitud, altitud)
- Orientación: (roll, pitch, yaw) - 6 grados de libertad
- Obtenida de: GPS + IMU + odometría
- Precisión: ~10cm (con GPS RTK)

**3. HD-Map**
- Mapa de alta definición pre-cargado de la zona
- Contiene coordenadas 3D exactas de todos los semáforos
- Cada semáforo tiene: `id`, `semantic_id`, `boundary` (puntos 3D)

**4. Calibración de cámara**
- Matriz intrínseca K (focal length, centro óptico)
- Matriz extrínseca T (transformación cámara → vehículo)
- Parámetros de distorsión

### ¿Qué hace?

#### **Paso 1: Query al HD-Map**

**Archivo**: `traffic_light_region_proposal_component.cc:343-377`

**⚠️ Dependencia Crítica: Car Pose** (`GetCarPose` línea 487-516)

Antes de consultar el HD-Map, Apollo necesita saber dónde está el vehículo:

```cpp
// Obtener pose del vehículo desde sistema de localización
GetCarPose(timestamp, &pose);

// pose contiene:
// - pose.pose_: Transformación car→world (posición GPS+IMU+odometry)
// - pose.c2w_poses_[camera_name]: Transformación de CADA cámara al mundo
//   Ejemplo: c2w_poses_["front_6mm"], c2w_poses_["front_25mm"]
```

**Impacto en testing**:
- ❌ **Pose incorrecta** → Busca signals en ubicación equivocada del mapa
- ❌ **c2w_pose con error de rotación** (ej: 5°) → Proyección 3D→2D desalineada
- ❌ **Delay en pose** (ej: 0.5s) → Obtiene signals donde el auto YA NO está

---

Apollo consulta el HD-Map con la posición actual del vehículo:

```cpp
// Línea 357-359
Eigen::Vector3d car_position = pose->getCarPosition();
if (!hd_map_->GetSignals(car_position, forward_distance_to_query_signals, &signals)) {
  // forward_distance_to_query_signals = 150.0 metros
}
```

**Pregunta**: *"Dame todos los semáforos que están dentro de un radio de 150 metros desde mi posición"*

**Implementación interna de `GetSignals`** (`hdmap_impl.cc:357-373`):

```cpp
// Línea 357: Implementación de GetSignals
int HDMapImpl::GetSignals(const Vec2d& point, double distance,
                          std::vector<SignalInfoConstPtr>* signals) const {
  signals->clear();
  std::vector<std::string> ids;

  // Buscar en KDTree todos los signals cercanos
  SearchObjects(point, distance, *signal_segment_kdtree_, &ids);

  // Para cada ID encontrado, obtener el SignalInfo completo
  for (const auto& id : ids) {
    signals->emplace_back(GetSignalById(CreateHDMapId(id)));
  }
  return 0;
}
```

**Respuesta del HD-Map** (ejemplo):

```cpp
signals = [
  SignalInfo {
    signal_.id: "signal_12345",
    signal_.boundary: {
      point[0]: {x: 500.23, y: 1200.45, z: 5.12},
      point[1]: {x: 500.28, y: 1200.50, z: 5.12},
      point[2]: {x: 500.28, y: 1200.50, z: 5.92},
      point[3]: {x: 500.23, y: 1200.45, z: 5.92}
    },
    signal_.type: MIX_3_VERTICAL,
    signal_.stop_line: [...],
    signal_.overlap_id: ["lane_123", "lane_124"],  // ← Conecta con lanes
    // ⚠️ NO tiene semantic_id
  },
  SignalInfo {
    signal_.id: "signal_12346",
    signal_.boundary: [(502.10, 1198.30, 5.15), ...],
    ...
  },
  ... (total: 8 semáforos)
]
```

**Observar**:
- Cada semáforo viene del HD-Map con su `id` único
- ✅ SÍ tiene `overlap_id[]` para conectar con lanes
- ❌ **NO** tiene `semantic_id` en el proto del HD-Map
- Las coordenadas `boundary` son en el sistema de coordenadas mundial (metros)
- `SignalInfo` es un wrapper que contiene el proto `Signal` del HD-Map

#### **Paso 2: Generación de TrafficLight objects**

**Archivo**: `traffic_light_region_proposal_component.cc:319-341`

Para cada signal del HD-Map, crea un objeto `TrafficLight`:

```cpp
// Línea 323-340
for (auto signal : signals) {
  base::TrafficLightPtr light;
  light.reset(new base::TrafficLight);
  light->id = signal.id().id();                    // Copia el ID del HD-Map

  // Copia los puntos del contorno 3D (boundary del Signal proto)
  for (int i = 0; i < signal.boundary().point_size(); ++i) {
    base::PointXYZID point;
    point.x = signal.boundary().point(i).x();
    point.y = signal.boundary().point(i).y();
    point.z = signal.boundary().point(i).z();
    light->region.points.push_back(point);
  }

  // ⚠️ AQUÍ ES DONDE SE HARDCODEA semantic_id = 0
  int cur_semantic = 0;  // ← Línea 335: SIEMPRE 0, nunca lee del mapa

  light->semantic = cur_semantic;  // ← Línea 337: Asigna 0 a todos
  traffic_lights->push_back(light);
  stoplines_ = signal.stop_line();
}
```

**Estado de TrafficLight después de este paso**:

```cpp
TrafficLight {
  // ✅ Campos llenos
  id: "signal_12345"
  semantic: 0  // ← ⚠️ SIEMPRE 0, no importa el cruce o intersección
  region.points: [(500.23, 1200.45, 5.12), ...]  // Puntos 3D del boundary

  // ❌ Campos vacíos (aún no calculados)
  region.projection_roi: [0, 0, 0, 0]
  region.detection_roi: [0, 0, 0, 0]
  region.crop_roi: [0, 0, 0, 0]
  region.outside_image: false
  region.is_detected: false
  region.detect_class_id: -1
  region.detect_score: 0.0
  status.color: UNKNOWN
  status.confidence: 0.0
  status.blink: false
}
```

#### **Paso 3: Proyección 3D → 2D**

**Archivo**: `tl_preprocessor.cc:236-272` y `multi_camera_projection.cc:86-189`

Para cada `TrafficLight`, proyecta sus puntos 3D (`region.points`) a coordenadas 2D en la imagen:

```cpp
// tl_preprocessor.cc:258-269
for (size_t i = 0; i < lights->size(); ++i) {
  auto light = lights->at(i);

  // projection_.Project() hace la transformación geométrica completa
  if (!projection_.Project(pose, ProjectOption(camera_name), light.get())) {
    // No se puede proyectar (está detrás de la cámara, muy lejos, etc.)
    light->region.outside_image = true;
    lights_outside_image->push_back(light);
  } else {
    // Proyección exitosa → llena light->region.projection_roi
    light->region.outside_image = false;
    lights_on_image->push_back(light);
  }
}
```

**Implementación detallada de `Project()`** (`multi_camera_projection.cc:86-112`):

```cpp
// Línea 86-112
bool MultiCamerasProjection::Project(const camera::CarPose& pose,
                                     const ProjectOption& option,
                                     base::TrafficLight* light) const {
  // Obtener la matriz de transformación cámara→mundo (c2w)
  Eigen::Matrix4d c2w_pose = pose.c2w_poses_.at(option.camera_name);

  // Delegar a BoundaryBasedProject que hace la proyección real
  return BoundaryBasedProject(camera_models_.at(option.camera_name),
                              c2w_pose,
                              light->region.points,  // ← Puntos 3D del boundary
                              light);
}
```

**Implementación de `BoundaryBasedProject()`** (`multi_camera_projection.cc:139-189`):

```cpp
// Línea 139-189
bool MultiCamerasProjection::BoundaryBasedProject(
    const base::BrownCameraDistortionModelPtr camera_model,
    const Eigen::Matrix4d& c2w_pose,
    const std::vector<base::PointXYZID>& points,  // ← Boundary del Signal
    base::TrafficLight* light) const {

  int width = camera_model->get_width();   // 1920
  int height = camera_model->get_height(); // 1080
  int bound_size = points.size();          // 4 puntos (cuadrilátero)

  EigenVector<Eigen::Vector2i> pts2d(bound_size);
  auto c2w_pose_inverse = c2w_pose.inverse();  // Invertir para obtener w2c

  // Para cada punto del boundary del Signal
  for (int i = 0; i < bound_size; ++i) {
    const auto& pt3d_world = points.at(i);  // Punto 3D en coordenadas mundo

    // 1. Transformar de mundo a cámara
    Eigen::Vector3d pt3d_cam =
        (c2w_pose_inverse *
         Eigen::Vector4d(pt3d_world.x, pt3d_world.y, pt3d_world.z, 1.0))
            .head(3);

    // Verificar que no esté detrás de la cámara
    if (pt3d_cam[2] <= 0.0) {
      return false;  // Punto detrás de la cámara
    }

    // 2. Proyectar de 3D cámara a 2D imagen (usando K y distorsión Brown)
    pts2d[i] = camera_model->Project(pt3d_cam.cast<float>()).cast<int>();
    // ⚠️ Usa calibración de cámara: focal length, centro óptico, distorsión
    // Impacto en testing: Calibración incorrecta → projection_roi desalineado
  }

  // 3. Calcular bounding box que envuelve todos los puntos proyectados
  int min_x = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int min_y = std::numeric_limits<int>::max();
  int max_y = std::numeric_limits<int>::min();

  for (const auto& pt : pts2d) {
    min_x = std::min(pt[0], min_x);
    max_x = std::max(pt[0], max_x);
    min_y = std::min(pt[1], min_y);
    max_y = std::max(pt[1], max_y);
  }

  // 4. Crear ROI y verificar que esté dentro de la imagen
  base::BBox2DI roi(min_x, min_y, max_x, max_y);
  if (camera::OutOfValidRegion(roi, width, height) || roi.Area() == 0) {
    return false;  // Proyección fuera de la imagen
  }

  // ✅ AQUÍ SE ASIGNA EL PROJECTION_ROI
  light->region.projection_roi = base::RectI(roi);
  return true;
}
```

**Ejemplo numérico**:
- **Input**: Semáforo #1 con boundary 3D:
  ```
  points = [
    (500.23, 1200.45, 5.12),  // Esquina inferior izquierda
    (500.28, 1200.50, 5.12),  // Esquina inferior derecha
    (500.28, 1200.50, 5.92),  // Esquina superior derecha
    (500.23, 1200.45, 5.92)   // Esquina superior izquierda
  ]
  ```
- **Transformación mundo→cámara**: Aplica `c2w_pose_inverse`
- **Proyección a 2D**: Puntos proyectados: `[(850, 380), (890, 378), (890, 298), (850, 300)]`
- **Bounding box**: `min_x=850, max_x=890, min_y=298, max_y=380`
- **Output**: `projection_roi = [850, 298, 40, 82]` píxeles (x, y, width, height)

#### **Paso 4: Selección de cámara (multi-cámara)**

**Archivo**: `tl_preprocessor.cc:180-234`

Apollo tiene dos cámaras con diferentes características:

| Cámara | Focal Length | FOV | Uso |
|--------|-------------|-----|-----|
| **Telephoto** | 25mm | Estrecho (~30°) | Semáforos lejanos, más resolución |
| **Wide-angle** | 6mm | Amplio (~120°) | Vista general, captura más área |

```cpp
// Línea 189-232
for (size_t cam_id = 0; cam_id < num_cameras_; ++cam_id) {
  const auto &camera_name = camera_names[cam_id];  // Orden: telephoto, wide

  bool ok = true;

  // Si NO es la cámara de menor focal (no es wide-angle):
  if (camera_name != min_focal_len_working_camera) {
    // Verificar que TODAS las proyecciones estén dentro
    if (lights_outside_image_array->at(cam_id).size() > 0) {
      ok = false;  // Alguna proyección quedó fuera
    }

    // Verificar que estén lejos de los bordes
    for (const auto light : lights_on_image_array->at(cam_id)) {
      if (OutOfValidRegion(light->region.projection_roi, width, height, border)) {
        ok = false;  // Muy cerca del borde
      }
    }
  }

  // Primera cámara que cumple condiciones → se selecciona
  if (ok) {
    *selected_camera_name = camera_name;
    break;
  }
}
```

**Lógica de selección**:
1. Intenta primero telephoto (mayor focal length)
2. Si todas las proyecciones caben y están lejos de bordes → usa telephoto
3. Si alguna proyección queda fuera o muy cerca del borde → usa wide-angle
4. Siempre selecciona la de mayor focal que cumpla condiciones (mejor resolución)

#### **Paso 5: Validación**

**Archivo**: `detection.cc:245-255`

Para cada proyección, verifica:

```cpp
// Línea 245-255
for (auto &light : lights_ref) {
  if (light->region.outside_image ||
      OutOfValidRegion(light->region.projection_roi, img_width, img_height) ||
      light->region.projection_roi.Area() <= 0) {
    // Invalidar la proyección
    light->region.projection_roi = base::RectI(0, 0, 0, 0);
  }
}
```

### ¿Qué entrega?

Una lista de objetos `TrafficLight`, donde cada uno representa un semáforo del HD-Map:

**Estado completo de TrafficLight después de Preprocesamiento**:

```cpp
TrafficLight {
  // ✅ Campos del HD-Map (persistentes entre frames)
  id: "signal_12345"                    // ID único del semáforo
  semantic: 0                           // ⚠️ SIEMPRE 0 (no agrupa por cruce)
  region.points: [(x,y,z), ...]         // Puntos 3D del boundary (Signal proto)

  // ✅ Campos calculados en proyección
  region.projection_roi: [850, 300, 40, 80]  // Dónde DEBERÍA aparecer en imagen
  region.outside_image: false           // Flag de visibilidad

  // ❌ Campos aún vacíos (se llenan en etapas siguientes)
  region.detection_roi: [0, 0, 0, 0]    // Dónde se DETECTÓ realmente
  region.crop_roi: [0, 0, 0, 0]         // ROI expandida para CNN
  region.is_detected: false             // Si el detector lo encontró
  region.detect_class_id: -1            // Tipo: vertical/quad/horizontal
  region.detect_score: 0.0              // Confianza del detector
  status.color: UNKNOWN                 // Color del semáforo
  status.confidence: 0.0                // Confianza del reconocimiento
  status.blink: false                   // Si está intermitente
}
```

**Relación clave**: 1 semáforo del HD-Map → 1 objeto TrafficLight con projection_roi

**Ejemplo con 3 semáforos del mismo cruce**:

```cpp
lights = [
  TrafficLight {
    id: "signal_12345",
    semantic: 100,  // ← Mismo grupo
    projection_roi: [850, 300, 40, 80]
  },
  TrafficLight {
    id: "signal_12346",
    semantic: 100,  // ← Mismo grupo
    projection_roi: [920, 310, 35, 75]
  },
  TrafficLight {
    id: "signal_12347",
    semantic: 100,  // ← Mismo grupo
    projection_roi: [780, 295, 38, 77]
  }
]
```

**Archivo fuente**: `traffic_light_region_proposal_component.cc`, `tl_preprocessor.cc`

---

## 🔷 ETAPA 2: DETECCIÓN

### ¿Qué recibe?

**Entrada 1: Lista de TrafficLight objects** (M semáforos, M=8 en ejemplo)

Estado actual de cada objeto:
```cpp
TrafficLight {
  // Campos llenos del preprocesamiento
  id: "signal_12345"
  semantic: 100
  region.projection_roi: [850, 300, 40, 80]  // ← Usará este para detectar
  region.outside_image: false

  // Campos vacíos que se llenarán
  region.detection_roi: [0, 0, 0, 0]
  region.crop_roi: [0, 0, 0, 0]
  region.is_detected: false
  region.detect_class_id: -1
  region.detect_score: 0.0
}
```

**Entrada 2**: Imagen de la cámara seleccionada (1920×1080)

**Entrada 3**: Buffer vacío `detected_bboxes_ = []` donde acumulará detecciones

### ¿Qué hace?

#### **Paso 1: Inicialización**

**Archivo**: `detection.cc:236-243`

```cpp
// Línea 236-243
for (auto &light : lights_ref) {
  // Copiar projection_roi a detection_roi (inicialmente)
  light->region.detection_roi = light->region.projection_roi;

  // Inicializar buffers de debug
  light->region.debug_roi.clear();
  light->region.debug_roi_detect_scores.clear();
}
```

#### **Paso 2: Loop serial sobre proyecciones**

**Archivo**: `detection.cc:142-216` (función `Inference`)

Apollo procesa cada `projection_roi` **una por una** (NO en batch):

```cpp
// Línea 149-150
auto batch_num = lights->size();  // M semáforos
for (size_t i = 0; i < batch_num; ++i) {
  base::TrafficLightPtr light = lights->at(i);
  // Procesar este semáforo...
}
```

Para el semáforo #1 (`projection_roi = [850, 300, 40, 80]`):

#### **Paso 3: Expansión del ROI (crop_scale = 2.5)**

**Archivo**: `detection.cc:175`

La proyección del HD-Map puede tener errores por:
- Imprecisión del GPS (~10cm)
- Errores de calibración de cámara
- Movimientos del vehículo

Entonces expande el rectángulo 2.5 veces:

```cpp
// Línea 175
crop_->getCropBox(img_width, img_height, light, &cbox);
// crop_ es un objeto CropBox con crop_scale=2.5
```

**Cálculo interno de CropBox** (`cropbox.cc:26-79`):
```python
projection_roi = [850, 300, 40, 80]  # [x, y, width, height]

# 1. Calcular centro
center_x = 850 + 40/2 = 870
center_y = 300 + 80/2 = 340

# 2. ⚠️ IMPORTANTE: Usa MAX(width, height) para hacer crop CUADRADO
max_dim = max(40, 80) = 80
resize = max_dim × crop_scale_ = 80 × 2.5 = 200

# 3. Aplicar mínimo (típicamente min_crop_size=270)
resize = max(resize, min_crop_size_) = max(200, 270) = 270

# 4. Clipping para no exceder imagen (1920×1080)
resize = min(resize, img_width) = min(270, 1920) = 270
resize = min(resize, img_height) = min(270, 1080) = 270

# 5. Crear cuadrado centrado
crop_roi.x = center_x - resize/2 + 1 = 870 - 135 + 1 = 736
crop_roi.y = center_y - resize/2 + 1 = 340 - 135 + 1 = 206
crop_roi.width = 270
crop_roi.height = 270

# 6. Ajustar si excede bordes de imagen (clamp)
if crop_roi.x < 0: crop_roi.x = 0
if crop_roi.y < 0: crop_roi.y = 0
if (crop_roi.x + 270) >= 1920: ajustar hacia izquierda
if (crop_roi.y + 270) >= 1080: ajustar hacia arriba

# Resultado: crop_roi = [736, 206, 270, 270]  ← CUADRADO
```

**⚠️ Diferencia clave**: El crop es SIEMPRE cuadrado, no rectangular. Usa la dimensión más grande del projection_roi.

```cpp
// Línea 181-183
light->region.crop_roi = cbox;  // Guardar para uso posterior
```

#### **Paso 4: Recorte de la imagen**

**Archivo**: `detection.cc:185-188`

```cpp
// Línea 185-188
data_provider_image_option_.do_crop = true;
data_provider_image_option_.crop_roi = cbox;  // [736, 206, 270, 270]
data_provider->GetImage(data_provider_image_option_, image_.get());
```

Extrae región `[736, 206, 270, 270]` de la imagen completa 1920×1080.
Ahora tiene una imagen de 270×270 píxeles (ya cuadrada).

#### **Paso 5: Resize a 270×270**

**Archivo**: `detection.cc:191-197`

```cpp
// Línea 191-197
float resize_scale = 270.0 / min(cbox.width, cbox.height);
                   = 270.0 / min(270, 270)
                   = 270.0 / 270
                   = 1.0

inference::ResizeGPU(*image_, input_img_blob, ...);
```

**En este caso**: Como el crop ya es 270×270, `resize_scale = 1.0` (no hay cambio de tamaño).

**En casos donde projection_roi es pequeño**: Si `max_dim × 2.5 < 270`, el crop será 270×270 mínimo, así que resize_scale siempre será ≥ 1.0.

#### **Paso 6: Inferencia de la CNN (Detector)**

**Archivo**: `detection.cc:202-206`

```cpp
// Línea 202-206
cudaDeviceSynchronize();
rt_net_->Infer();  // Ejecuta la red neuronal (tl.torch)
cudaDeviceSynchronize();
```

**Red neuronal**: SSD-style detector
**Entrada**: Imagen 270×270
**Salida**: Tensor [N_detections × 9]

Donde cada fila es:
```
[img_id, x1, y1, x2, y2, bg_score, vertical_score, quad_score, horizontal_score]
```

**Ejemplo de output** para este crop:
```
Detection A: [0, 45, 60, 70, 180, 0.10, 0.85, 0.20, 0.15]
  → img_id=0, bbox=[45,60,70,180], scores=[bg:0.10, v:0.85, q:0.20, h:0.15]
  → Clase: VERTICAL (score 0.85)

Detection B: [0, 80, 50, 95, 150, 0.05, 0.92, 0.10, 0.08]
  → img_id=0, bbox=[80,50,95,150], scores=[bg:0.05, v:0.92, q:0.10, h:0.08]
  → Clase: VERTICAL (score 0.92)

Detection C: [0, 120, 100, 150, 200, 0.95, 0.05, 0.10, 0.02]
  → img_id=0, bbox=[120,100,150,200], scores=[bg:0.95, v:0.05, q:0.10, h:0.02]
  → Clase: BACKGROUND (score 0.95) → Se descarta
```

#### **Paso 7: Procesamiento de detecciones (SelectOutputBoxes)**

**Archivo**: `detection.cc:278-371`

Para cada detection en el output de la CNN:

```cpp
// Línea 300-310
for (int candidate_id = 0; candidate_id < result_box_num; candidate_id++) {
  const float *result_data = output_blob->cpu_data() + candidate_id * each_box_length;

  int img_id = static_cast<int>(result_data[0]);
  if (img_id < 0) continue;  // Detection inválida

  base::TrafficLightPtr tmp(new base::TrafficLight);

  // Extraer coordenadas y scores
  float x1 = result_data[1];
  float y1 = result_data[2];
  float x2 = result_data[3];
  float y2 = result_data[4];
  std::vector<float> score{result_data[5], result_data[6],
                           result_data[7], result_data[8]};
```

**7a. Determinar clase (argmax de scores)**

```cpp
// Línea 323-326
std::vector<float>::iterator biggest = std::max_element(score.begin(), score.end());
tmp->region.detect_class_id =
    base::TLDetectionClass(std::distance(score.begin(), biggest) - 1);
```

**Mapeo de índices**:
```
score[0] = background → class_id = -1 (se descarta)
score[1] = vertical   → class_id = 0 (TL_VERTICAL_CLASS)
score[2] = quadrate   → class_id = 1 (TL_QUADRATE_CLASS)
score[3] = horizontal → class_id = 2 (TL_HORIZONTAL_CLASS)
```

**7b. Transformar coordenadas a imagen original**

```cpp
// Línea 311-312
float inflate_col = 1 / resize_scale;  // 1 / 1.0 = 1.0 (en este ejemplo)
float inflate_row = 1 / resize_scale;

// Línea 329-334
// Coordenadas en crop 270×270 → coordenadas en crop original 270×270
tmp->region.detection_roi.x = static_cast<int>(x1 * inflate_col);
tmp->region.detection_roi.y = static_cast<int>(y1 * inflate_row);
tmp->region.detection_roi.width = static_cast<int>((x2 - x1 + 1) * inflate_col);
tmp->region.detection_roi.height = static_cast<int>((y2 - y1 + 1) * inflate_row);
```

**Ejemplo con Detection A**:
```
En crop 270×270: [45, 60, 70, 180]
Inflate: [45×1.0, 60×1.0, 70×1.0, 180×1.0] = [45, 60, 70, 180]
En crop original 270×270: [45, 60, 70, 180] (sin cambio en este caso)
```

**7c. Validar que esté dentro del crop**

```cpp
// Línea 337-350
if (camera::OutOfValidRegion(tmp->region.detection_roi,
                             crop_box_list.at(img_id).width,   // 270
                             crop_box_list.at(img_id).height) || // 270
    tmp->region.detection_roi.Area() <= 0) {
  AINFO << "Invalid width or height...";
  continue;  // ← Descarta esta detección
}
```

**Verifica** que la bbox detectada esté completamente dentro del crop. Si se sale, se descarta.

**7d. Ajustar bounds y traducir a imagen completa**

```cpp
// Línea 352-356
// Ajustar si la bbox excede ligeramente el crop
camera::RefineBox(tmp->region.detection_roi,
                  crop_box_list.at(img_id).width,
                  crop_box_list.at(img_id).height,
                  &(tmp->region.detection_roi));

// Traducir del crop a la imagen completa
tmp->region.detection_roi.x += crop_box_list.at(img_id).x;  // +736
tmp->region.detection_roi.y += crop_box_list.at(img_id).y;  // +206
```

**Resultado final**:
```
En imagen original 1920×1080: [45+736, 60+206, 70, 180] = [781, 266, 70, 180]
```

**7e. Agregar al buffer global**

```cpp
// Línea 357-363
tmp->region.is_detected = true;
tmp->region.detect_score = *biggest;  // 0.85 para Detection A

// ✨ ESTE ES EL FAMOSO PUSH_BACK
lights->push_back(tmp);
```

**Estado de `detected_bboxes_` después de procesar semáforo #1**:
```cpp
detected_bboxes_ = [
  TrafficLight {
    region.detection_roi: [781, 266, 70, 180],
    region.detect_class_id: TL_VERTICAL_CLASS (0),
    region.detect_score: 0.85,
    region.is_detected: true
  },
  TrafficLight {
    region.detection_roi: [816, 256, 65, 170],
    region.detect_class_id: TL_VERTICAL_CLASS (0),
    region.detect_score: 0.92,
    region.is_detected: true
  }
]
// Detection C se descartó (background con score 0.95)
```

#### **Repetir para todos los semáforos**

El loop continúa con semáforo #2, #3, ..., #8. Cada uno puede agregar 0, 1, 2 o más detecciones.

**Resultado después de procesar los 8 semáforos**:
```cpp
detected_bboxes_.size() = 15  // De 8 projection ROIs
```

**Ejemplo de distribución**:
- Semáforo #1 (projection_roi): 2 detections
- Semáforo #2: 1 detection
- Semáforo #3: 0 detections (no detectó nada)
- Semáforo #4: 3 detections (había 3 semáforos juntos)
- Semáforo #5: 2 detections
- Semáforo #6: 2 detections
- Semáforo #7: 3 detections
- Semáforo #8: 2 detections
**Total: 15 detections**

#### **Paso 8: NMS Global (Non-Maximum Suppression)**

**Archivo**: `detection.cc:373-422`

Ahora `detected_bboxes_` tiene 15 detecciones, pero muchas pueden ser duplicadas (dos detecciones del mismo semáforo físico).

```cpp
// Línea 214
ApplyNMS(&detected_bboxes_);
```

**Algoritmo NMS**:

```cpp
// Línea 381-390: Crear pares (score, index) y ordenar
std::vector<std::pair<float, int>> score_index_vec(lights->size());
for (size_t i = 0; i < lights->size(); ++i) {
  score_index_vec[i].first = lights->at(i)->region.detect_score;
  score_index_vec[i].second = static_cast<int>(i);
}

// Ordenar ASCENDING (de menor a mayor score)
std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
    [](const std::pair<float, int> &pr1, const std::pair<float, int> &pr2) {
      return pr1.first < pr2.first;  // Línea 389: ASCENDING
    });
```

**Lista ordenada** (ejemplo):
```
[
  (score=0.65, idx=5),
  (score=0.72, idx=11),
  (score=0.78, idx=2),
  (score=0.82, idx=8),
  (score=0.85, idx=0),
  (score=0.88, idx=14),
  (score=0.90, idx=3),
  (score=0.92, idx=1),
  ...
]
```

```cpp
// Línea 393-413: Greedy NMS
std::vector<int> kept_indices;
while (!score_index_vec.empty()) {
  const int idx = score_index_vec.back().second;  // Toma el de MAYOR score
  bool keep = true;

  // Compara con todos los ya guardados
  for (size_t k = 0; k < kept_indices.size(); ++k) {
    const int kept_idx = kept_indices[k];
    const auto &rect1 = lights->at(idx)->region.detection_roi;
    const auto &rect2 = lights->at(kept_idx)->region.detection_roi;

    // Calcular IoU (Intersection over Union)
    float overlap = (rect1 & rect2).Area() / (rect1 | rect2).Area();

    // Si overlap > threshold → descartar
    keep = std::fabs(overlap) < iou_thresh;  // iou_thresh = 0.6
    if (!keep) break;
  }

  if (keep) {
    kept_indices.push_back(idx);
  }
  score_index_vec.pop_back();  // Eliminar procesado
}
```

**Ejemplo de ejecución**:
```
Iteración 1: idx=1 (score=0.92) → kept = [1]
Iteración 2: idx=3 (score=0.90)
  - IoU con 1: 0.02 < 0.6 → keep
  - kept = [1, 3]
Iteración 3: idx=14 (score=0.88)
  - IoU con 1: 0.75 > 0.6 → discard (es duplicado de 1)
Iteración 4: idx=0 (score=0.85)
  - IoU con 1: 0.12 < 0.6 → keep
  - IoU con 3: 0.08 < 0.6 → keep
  - kept = [1, 3, 0]
...
```

**Resultado final**:
```cpp
kept_indices = [1, 3, 0, 7, 4, 12, 6, 9, 10]  // 9 detections sobrevivieron
```

```cpp
// Línea 415-421: Eliminar las no guardadas
auto parted_itr = std::stable_partition(
    lights->begin(), lights->end(),
    [&](const base::TrafficLightPtr &light) {
      return std::find(kept_indices.begin(), kept_indices.end(), idx++) !=
             kept_indices.end();
    });
lights->erase(parted_itr, lights->end());
```

**Estado de `detected_bboxes_` después de NMS**:
```cpp
detected_bboxes_.size() = 9  // Eliminó 6 duplicadas
```

### ¿Qué entrega?

**Buffer `detected_bboxes_`** con N detecciones (N=9 en ejemplo):

Cada detection es un `TrafficLight` object con:

```cpp
TrafficLight {
  // ✅ Campos llenos por el detector
  region.detection_roi: [845, 280, 35, 65]   // Bbox en imagen original
  region.detect_class_id: TL_VERTICAL_CLASS (0)  // Tipo detectado
  region.detect_score: 0.92                   // Confianza
  region.is_detected: true

  // ❌ Campos vacíos (estas detections NO tienen id ni semantic)
  id: ""                    // No vienen del HD-Map
  semantic: 0               // No tienen semantic_id asignado aún
  region.projection_roi: [0, 0, 0, 0]
  region.outside_image: false
  status.color: UNKNOWN
  status.confidence: 0.0
  status.blink: false
}
```

**Observación importante**: Las detections en `detected_bboxes_` **NO tienen `id` ni `semantic_id`** porque son outputs puros de la CNN, no están asociadas al HD-Map todavía.

**Relación clave**: De M projection boxes (M=8) → N detections (N=9)
- Puede ser N > M (múltiples detections por projection)
- Puede ser N = M (una detection por projection)
- Puede ser N < M (algunas projections no generaron detections)

**Archivo fuente**: `detection.cc`

---

## 🔷 ETAPA 3: ASIGNACIÓN (Hungarian Algorithm)

### ¿Qué recibe?

**Entrada 1: hdmap_bboxes** (M=8 semáforos del HD-Map)

Estado actual de cada objeto:
```cpp
TrafficLight {
  // ✅ Campos del HD-Map (tienen identidad)
  id: "signal_12345"
  semantic: 100
  region.projection_roi: [850, 300, 40, 80]   // Dónde DEBERÍA estar
  region.crop_roi: [820, 240, 100, 200]       // ROI expandida usada
  region.outside_image: false

  // ❌ Campos vacíos (buscarán un match en detected_bboxes)
  region.detection_roi: [850, 300, 40, 80]    // Copiado de projection (temporal)
  region.is_detected: false                   // Aún no asignado
  region.detect_class_id: -1
  region.detect_score: 0.0
  status.color: UNKNOWN
}
```

**Entrada 2: detected_bboxes** (N=9 detections post-NMS)

Estado de cada detection:
```cpp
TrafficLight {
  // ✅ Campos del detector
  region.detection_roi: [845, 280, 35, 65]
  region.detect_class_id: TL_VERTICAL_CLASS (0)
  region.detect_score: 0.92
  region.is_detected: true

  // ❌ No tienen identidad del HD-Map
  id: ""
  semantic: 0
  region.projection_roi: [0, 0, 0, 0]
  region.crop_roi: [0, 0, 0, 0]
}
```

### ¿Qué hace?

El problema es: *"Tengo 8 semáforos del HD-Map (con identidad) y 9 detections de la CNN (sin identidad). ¿Cómo los asocio de forma óptima?"*

#### **Paso 1: Construcción de la matriz de costos (M×N)**

**Archivo**: `select.cc:42-86`

```cpp
// Línea 46
munkres_.costs()->Resize(hdmap_bboxes->size(), refined_bboxes.size());
// Matriz de 8 filas × 9 columnas
```

Para cada celda `[i,j]`, calcula un score que indica qué tan buena es la asociación entre `hdmap[i]` y `detection[j]`.

```cpp
// Línea 48-85
for (size_t row = 0; row < hdmap_bboxes->size(); ++row) {       // M filas
  auto center_hd = (*hdmap_bboxes)[row]->region.detection_roi.Center();

  // Si la proyección está fuera de imagen → cost = 0 para todas las detections
  if ((*hdmap_bboxes)[row]->region.outside_image) {
    for (size_t col = 0; col < refined_bboxes.size(); ++col) {
      (*munkres_.costs())(row, col) = 0.0;
    }
    continue;
  }

  for (size_t col = 0; col < refined_bboxes.size(); ++col) {    // N columnas
    // Calcular score combinado...
  }
}
```

#### **Paso 2: Cálculo del score combinado**

Para cada par `(hdmap[i], detection[j])`:

**2a. Distance score (Gaussian 2D)**

```cpp
// Línea 58-62
float gaussian_score = 100.0f;  // σ (sigma)
auto center_refine = refined_bboxes[col]->region.detection_roi.Center();

double distance_score = Calc2dGaussianScore(
    center_hd, center_refine, gaussian_score, gaussian_score);
```

**Función Gaussian score** (`select.cc:34-40`):
```cpp
double Select::Calc2dGaussianScore(base::Point2DI p1, base::Point2DI p2,
                                   float sigma1, float sigma2) {
  return std::exp(-0.5 * (
      static_cast<float>((p1.x - p2.x) * (p1.x - p2.x)) / (sigma1 * sigma1) +
      static_cast<float>((p1.y - p2.y) * (p1.y - p2.y)) / (sigma2 * sigma2)
  ));
}
```

**Ejemplo numérico**:
```
center_hd = projection_roi.center() = (850 + 40/2, 300 + 80/2) = (870, 340)
center_det = detection_roi.center() = (845 + 35/2, 280 + 65/2) = (862, 312)

dx = 862 - 870 = -8 píxeles
dy = 312 - 340 = -28 píxeles

distance_score = exp(-0.5 × ((-8/100)² + (-28/100)²))
               = exp(-0.5 × (0.0064 + 0.0784))
               = exp(-0.5 × 0.0848)
               = exp(-0.0424)
               ≈ 0.9585
```

Si la detection está **muy cerca** de la proyección → score **alto** (≈1.0)
Si está **lejos** → score **bajo** (≈0.0)

**2b. Detection score (clipped)**

```cpp
// Línea 64-67
double max_score = 0.9;
auto detect_score = refined_bboxes[col]->region.detect_score;
double detection_score = detect_score > max_score ? max_score : detect_score;
```

Ejemplo:
```
detect_score = 0.92 → detection_score = 0.9  (clipped)
detect_score = 0.75 → detection_score = 0.75 (sin cambio)
```

**2c. Score combinado (70% distancia + 30% confianza)**

```cpp
// Línea 69-73
double distance_weight = 0.7;
double detection_weight = 1 - distance_weight;  // 0.3
(*munkres_.costs())(row, col) =
    static_cast<float>(detection_weight * detection_score +
                       distance_weight * distance_score);
```

Ejemplo:
```
combined_score = 0.3 × 0.9 + 0.7 × 0.9585
               = 0.27 + 0.671
               = 0.941
```

**Peso espacial dominante**: Apollo confía más en la posición de la proyección HD-Map (70%) que en el score del detector (30%).

#### **Paso 3: Validación ROI (ANTES del Hungarian)**

```cpp
// Línea 74-83
const auto &crop_roi = (*hdmap_bboxes)[row]->region.crop_roi;
const auto &detection_roi = refined_bboxes[col]->region.detection_roi;

// Verificar si la detection está COMPLETAMENTE dentro del crop_roi
if ((detection_roi & crop_roi) != detection_roi) {
  // Detection fuera del crop → penalizar
  (*munkres_.costs())(row, col) = 0.0;
}
```

**Ejemplo**:
```
crop_roi = [820, 240, 100, 200]  # ROI expandida 2.5×
detection_roi = [845, 280, 35, 65]

# Calcular intersección
intersection = detection_roi & crop_roi
             = [max(845,820), max(280,240),
                min(845+35,820+100), min(280+65,240+200)]
             = [845, 280, 880, 345]
             = [845, 280, 35, 65]  # == detection_roi

# La detection está completamente dentro → OK, mantener cost
```

Si la detection estuviera parcialmente fuera del crop_roi, se marca `cost = 0` (penaliza fuertemente).

**Matriz de costos final** (ejemplo 8×9):

```
         det0   det1   det2   det3   det4   det5   det6   det7   det8
hd0    | 0.65 | 0.92 | 0.31 | 0.15 | 0.08 | 0.12 | 0.00 | 0.19 | 0.22 |
hd1    | 0.11 | 0.08 | 0.74 | 0.88 | 0.21 | 0.09 | 0.14 | 0.00 | 0.18 |
hd2    | 0.09 | 0.13 | 0.19 | 0.22 | 0.91 | 0.76 | 0.11 | 0.25 | 0.00 |
hd3    | 0.00 | 0.17 | 0.00 | 0.11 | 0.14 | 0.82 | 0.94 | 0.31 | 0.12 |
hd4    | 0.21 | 0.10 | 0.08 | 0.12 | 0.15 | 0.13 | 0.20 | 0.85 | 0.79 |
hd5    | 0.88 | 0.15 | 0.12 | 0.09 | 0.18 | 0.11 | 0.17 | 0.13 | 0.93 |
hd6    | 0.12 | 0.89 | 0.14 | 0.16 | 0.10 | 0.09 | 0.21 | 0.11 | 0.15 |
hd7    | 0.17 | 0.11 | 0.92 | 0.13 | 0.09 | 0.14 | 0.10 | 0.12 | 0.08 |
```

#### **Paso 4: Ejecutar Hungarian Algorithm**

```cpp
// Línea 88
munkres_.Maximize(&assignments);
```

El algoritmo húngaro encuentra la asignación óptima que **maximiza la suma total** de scores, respetando la restricción de **1-to-1** (cada fila se asigna a máximo una columna y viceversa).

---

### 🔍 **Cómo funciona internamente el Hungarian Algorithm**

**Archivo**: `hungarian_optimizer.h:244-721`

El algoritmo Húngaro (también llamado algoritmo de Munkres) resuelve el **problema de asignación óptima** en tiempo O(n³).

#### **Conversión Maximize → Minimize**

Apollo necesita **maximizar** scores, pero el algoritmo Húngaro clásico **minimiza** costos. La conversión es simple:

```cpp
// hungarian_optimizer.h:248-259
void HungarianOptimizer<T>::Maximize(
    std::vector<std::pair<size_t, size_t>>* assignments) {
  OptimizationInit();

  // Convertir maximización a minimización
  // Restando cada score del max_score
  for (size_t row = 0; row < height_; ++row) {
    for (size_t col = 0; col < width_; ++col) {
      costs_(row, col) = max_cost_ - costs_(row, col);
    }
  }

  Minimize(assignments);  // Ahora minimizar costos invertidos
}
```

**Ejemplo numérico**:
```
Matriz original (scores a MAXIMIZAR):
         det0   det1   det2
hd0    | 0.65 | 0.92 | 0.31 |
hd1    | 0.11 | 0.08 | 0.74 |
hd2    | 0.09 | 0.13 | 0.19 |

max_cost = 0.94

Matriz invertida (costos a MINIMIZAR):
         det0   det1   det2
hd0    | 0.29 | 0.02 | 0.63 |  (0.94 - scores)
hd1    | 0.83 | 0.86 | 0.20 |
hd2    | 0.85 | 0.81 | 0.75 |
```

Ahora **minimizar** costos invertidos = **maximizar** scores originales.

#### **Los 6 Pasos del Algoritmo Munkres**

El algoritmo ejecuta un estado-máquina con 6 pasos iterativos hasta encontrar la asignación óptima:

```cpp
// hungarian_optimizer.h:484-495
void HungarianOptimizer<T>::DoMunkres() {
  int max_num_iter = 1000;
  int num_iter = 0;
  fn_state_ = std::bind(&HungarianOptimizer::ReduceRows, this);

  while (fn_state_ != nullptr && num_iter < max_num_iter) {
    fn_state_();  // Ejecuta el paso actual
    ++num_iter;
  }
}
```

---

**PASO 1: ReduceRows** - Restar mínimo de cada fila

```cpp
// hungarian_optimizer.h:524-535
void HungarianOptimizer<T>::ReduceRows() {
  for (size_t row = 0; row < matrix_size_; ++row) {
    // Encontrar mínimo de la fila
    T min_cost = costs_(row, 0);
    for (size_t col = 1; col < matrix_size_; ++col) {
      min_cost = std::min(min_cost, costs_(row, col));
    }
    // Restar mínimo de toda la fila
    for (size_t col = 0; col < matrix_size_; ++col) {
      costs_(row, col) -= min_cost;
    }
  }
  fn_state_ = std::bind(&HungarianOptimizer::StarZeroes, this);
}
```

**Ejemplo**:
```
Matriz invertida:                    Después de ReduceRows:
         det0   det1   det2                   det0   det1   det2
hd0    | 0.29 | 0.02 | 0.63 |  min=0.02  →  | 0.27 | 0.00 | 0.61 |
hd1    | 0.83 | 0.86 | 0.20 |  min=0.20  →  | 0.63 | 0.66 | 0.00 |
hd2    | 0.85 | 0.81 | 0.75 |  min=0.75  →  | 0.10 | 0.06 | 0.00 |
```

**Objetivo**: Crear ceros en la matriz (candidatos para asignación).

---

**PASO 2: StarZeroes** - Marcar ceros independientes con ⭐

```cpp
// hungarian_optimizer.h:542-563
void HungarianOptimizer<T>::StarZeroes() {
  for (size_t row = 0; row < matrix_size_; ++row) {
    if (RowCovered(row)) continue;

    for (size_t col = 0; col < matrix_size_; ++col) {
      if (ColCovered(col)) continue;

      if (costs_(row, col) == 0) {
        Star(row, col);      // Marcar con ⭐
        CoverRow(row);       // Cubrir fila (no buscar más ceros aquí)
        CoverCol(col);       // Cubrir columna
        break;
      }
    }
  }
  ClearCovers();
  fn_state_ = std::bind(&HungarianOptimizer::CoverStarredZeroes, this);
}
```

**Ejemplo**:
```
Matriz con ceros:                    Después de StarZeroes:
         det0   det1   det2                   det0   det1   det2
hd0    | 0.27 | 0.00 | 0.61 |          →     | 0.27 | 0.00⭐| 0.61 |
hd1    | 0.63 | 0.66 | 0.00 |          →     | 0.63 | 0.66 | 0.00⭐|
hd2    | 0.10 | 0.06 | 0.00 |          →     | 0.10 | 0.06 | 0.00 |
                                              (hd2 no tiene ⭐ porque det2 ya cubierto)
```

**Objetivo**: Encontrar asignaciones independientes 1-to-1.

---

**PASO 3: CoverStarredZeroes** - Cubrir columnas con ⭐

```cpp
// hungarian_optimizer.h:570-585
void HungarianOptimizer<T>::CoverStarredZeroes() {
  size_t num_covered = 0;

  for (size_t col = 0; col < matrix_size_; ++col) {
    if (ColContainsStar(col)) {
      CoverCol(col);
      num_covered++;
    }
  }

  // Si todas las columnas están cubiertas → ¡SOLUCIÓN ENCONTRADA!
  if (num_covered >= matrix_size_) {
    fn_state_ = nullptr;  // Terminar
    return;
  }
  fn_state_ = std::bind(&HungarianOptimizer::PrimeZeroes, this);
}
```

**Ejemplo**:
```
Matriz:                              Columnas cubiertas:
         det0   det1   det2                   det0   det1✓  det2✓
hd0    | 0.27 | 0.00⭐| 0.61 |          →     (2 de 3 columnas cubiertas)
hd1    | 0.63 | 0.66 | 0.00⭐|          →     ⚠️ Falta cubrir det0 → continuar
hd2    | 0.10 | 0.06 | 0.00 |
```

**Condición de éxito**: Si todas las columnas están cubiertas, cada fila tiene exactamente una ⭐ → asignación óptima completa.

---

**PASO 4: PrimeZeroes** - Buscar ceros no cubiertos y marcar con '

```cpp
// hungarian_optimizer.h:593-623
void HungarianOptimizer<T>::PrimeZeroes() {
  for (;;) {
    size_t zero_row, zero_col;
    if (!FindZero(&zero_row, &zero_col)) {
      // No hay ceros descubiertos → ir a Paso 6
      fn_state_ = std::bind(&HungarianOptimizer::AugmentPath, this);
      return;
    }

    Prime(zero_row, zero_col);  // Marcar con '
    int star_col = FindStarInRow(zero_row);

    if (star_col != kHungarianOptimizerColNotFound) {
      // Hay ⭐ en la misma fila → cubrir fila, descubrir columna
      CoverRow(zero_row);
      UncoverCol(star_col);
    } else {
      // No hay ⭐ en la fila → encontramos un "augmenting path"
      assignments_[0] = std::make_pair(zero_row, zero_col);
      fn_state_ = std::bind(&HungarianOptimizer::MakeAugmentingPath, this);
      return;
    }
  }
}
```

**Ejemplo**:
```
Matriz (columnas det1, det2 cubiertas):
         det0   det1✓  det2✓
hd0    | 0.27 | 0.00⭐| 0.61 |
hd1    | 0.63 | 0.66 | 0.00⭐|
hd2    | 0.10 | 0.06 | 0.00 |

Buscar ceros NO cubiertos:
- (hd0, det0)? 0.27 ≠ 0 → no
- (hd1, det0)? 0.63 ≠ 0 → no
- (hd2, det0)? 0.10 ≠ 0 → no

No hay ceros descubiertos → ir a Paso 6 (AugmentPath)
```

---

**PASO 5: MakeAugmentingPath** - Alternar ' y ⭐ para mejorar asignación

```cpp
// hungarian_optimizer.h:635-696
void HungarianOptimizer<T>::MakeAugmentingPath() {
  bool done = false;
  size_t count = 0;

  while (!done) {
    // Buscar ⭐ en la columna del ' actual
    int row = FindStarInCol(assignments_[count].second);

    if (row != kHungarianOptimizerRowNotFound) {
      count++;
      assignments_[count].first = row;
      assignments_[count].second = assignments_[count - 1].second;
    } else {
      done = true;  // No hay ⭐ → terminar path
    }

    if (!done) {
      // Buscar ' en la fila de la ⭐
      int col = FindPrimeInRow(assignments_[count].first);
      count++;
      assignments_[count].first = assignments_[count - 1].first;
      assignments_[count].second = col;
    }
  }

  // Alternar: ⭐ → sin marca, ' → ⭐
  for (size_t i = 0; i <= count; ++i) {
    size_t row = assignments_[i].first;
    size_t col = assignments_[i].second;

    if (IsStarred(row, col)) {
      Unstar(row, col);
    } else {
      Star(row, col);
    }
  }

  ClearCovers();
  ClearPrimes();
  fn_state_ = std::bind(&HungarianOptimizer::CoverStarredZeroes, this);
}
```

**Objetivo**: Incrementar el número de asignaciones ⭐ intercambiando ' y ⭐ a lo largo de un camino alternante.

---

**PASO 6: AugmentPath** - Crear más ceros ajustando la matriz

```cpp
// hungarian_optimizer.h:703-721
void HungarianOptimizer<T>::AugmentPath() {
  T minval = FindSmallestUncovered();

  // Sumar mínimo a filas CUBIERTAS
  for (size_t row = 0; row < matrix_size_; ++row) {
    if (RowCovered(row)) {
      for (size_t c = 0; c < matrix_size_; ++c) {
        costs_(row, c) += minval;
      }
    }
  }

  // Restar mínimo de columnas DESCUBIERTAS
  for (size_t col = 0; col < matrix_size_; ++col) {
    if (!ColCovered(col)) {
      for (size_t r = 0; r < matrix_size_; ++r) {
        costs_(r, col) -= minval;
      }
    }
  }

  fn_state_ = std::bind(&HungarianOptimizer::PrimeZeroes, this);
}
```

**Ejemplo**:
```
Matriz (antes):                      Matriz (después):
         det0   det1✓  det2✓               det0   det1✓  det2✓
hd0    | 0.27 | 0.00⭐| 0.61 |        →    | 0.17 | 0.00⭐| 0.61 |
hd1    | 0.63 | 0.66 | 0.00⭐|        →    | 0.53 | 0.66 | 0.00⭐|
hd2    | 0.10 | 0.06 | 0.00 |        →    | 0.00 | 0.06 | 0.00 |

minval = 0.10 (mínimo en det0 descubierta)
Restar 0.10 de det0 → crea nuevo cero en (hd2, det0)
```

**Objetivo**: Modificar la matriz para crear nuevos ceros sin cambiar la optimalidad.

---

#### **Extracción de asignaciones finales**

Después de que el algoritmo termina (Paso 3 con todas las columnas cubiertas), extrae las asignaciones:

```cpp
// hungarian_optimizer.h:329-340
void HungarianOptimizer<T>::FindAssignments(
    std::vector<std::pair<size_t, size_t>>* assignments) {
  assignments->clear();

  for (size_t row = 0; row < height_; ++row) {
    for (size_t col = 0; col < width_; ++col) {
      if (IsStarred(row, col)) {
        assignments->push_back(std::make_pair(row, col));
        break;  // Solo una ⭐ por fila
      }
    }
  }
}
```

**Resultado** (ejemplo):
```cpp
assignments = [
  (hd0 → det1),  // score: 0.92
  (hd1 → det3),  // score: 0.88
  (hd2 → det4),  // score: 0.91
  (hd3 → det6),  // score: 0.94
  (hd4 → det7),  // score: 0.85
  (hd5 → det8),  // score: 0.93
  (hd6 → det0),  // score: 0.88
  (hd7 → det2)   // score: 0.92
]
```

**Garantía**: El algoritmo Húngaro garantiza que esta asignación maximiza la suma total de scores (0.92 + 0.88 + 0.91 + 0.94 + 0.85 + 0.93 + 0.88 + 0.92 = 7.23) respetando la restricción 1-to-1.

---

#### **Paso 5: Post-procesamiento con flags is_selected**

**Archivo**: `select.cc:90-120`

```cpp
// Línea 90-93: Inicializar todos como no seleccionados
for (size_t i = 0; i < hdmap_bboxes->size(); ++i) {
  (*hdmap_bboxes)[i]->region.is_selected = false;
  (*hdmap_bboxes)[i]->region.is_detected = false;
}

// Línea 95-119: Procesar cada assignment
for (size_t i = 0; i < assignments.size(); ++i) {
  size_t hd_idx = assignments[i].first;
  size_t det_idx = assignments[i].second;

  // VALIDACIÓN 1: Índices dentro de bounds
  if (hd_idx >= hdmap_bboxes->size() || det_idx >= refined_bboxes.size()) {
    continue;  // Skip
  }

  // VALIDACIÓN 2: Verificar flags is_selected (prevenir duplicados)
  if ((*hdmap_bboxes)[hd_idx]->region.is_selected ||
      refined_bboxes[det_idx]->region.is_selected) {
    continue;  // Ya fueron usados → skip
  }

  auto &refined_bbox_region = refined_bboxes[det_idx]->region;
  auto &hdmap_bbox_region = (*hdmap_bboxes)[hd_idx]->region;

  // MARCAR COMO SELECCIONADOS (1-to-1 enforcement)
  refined_bbox_region.is_selected = true;
  hdmap_bbox_region.is_selected = true;

  // VALIDACIÓN 3: Detection dentro de crop_roi
  const auto &crop_roi = hdmap_bbox_region.crop_roi;
  const auto &detection_roi = refined_bbox_region.detection_roi;
  bool outside_crop_roi = ((crop_roi & detection_roi) != detection_roi);

  // COPIAR o INVALIDAR
  if (hdmap_bbox_region.outside_image || outside_crop_roi) {
    hdmap_bbox_region.is_detected = false;  // No válido
  } else {
    // ✅ COPIAR DATOS DE LA DETECTION AL HD-MAP LIGHT
    hdmap_bbox_region.detection_roi = refined_bbox_region.detection_roi;
    hdmap_bbox_region.detect_class_id = refined_bbox_region.detect_class_id;
    hdmap_bbox_region.detect_score = refined_bbox_region.detect_score;
    hdmap_bbox_region.is_detected = refined_bbox_region.is_detected;
    hdmap_bbox_region.is_selected = refined_bbox_region.is_selected;
  }
}
```

**Ejemplo de ejecución**:

```
Assignment 1: (hd0 → det1)
  - hd0.is_selected? false ✓
  - det1.is_selected? false ✓
  - Marcar ambos como selected
  - Copiar: hd0.detection_roi = det1.detection_roi
  - hd0.is_detected = true ✓

Assignment 2: (hd1 → det3)
  - hd1.is_selected? false ✓
  - det3.is_selected? false ✓
  - Copiar datos ✓

...

Assignment 6: (hd6 → det1)
  - hd6.is_selected? false ✓
  - det1.is_selected? true ✗  ← YA FUE ASIGNADO
  - SKIP (no copiar nada)
  - hd6 queda sin detection

Assignment 7: (hd7 → det2)
  - hd7.is_selected? false ✓
  - det2.is_selected? false ✓
  - Copiar datos ✓
```

**Los flags `is_selected` aseguran 1-to-1**: Una detection solo puede asignarse a UN HD-Map light.

### ¿Qué entrega?

**Lista `hdmap_bboxes` actualizada** (M=8 semáforos):

**Ejemplo de 3 semáforos del mismo grupo (semantic_id=100)**:

**Semáforo detectado exitosamente**:
```cpp
TrafficLight {
  // ✅ Campos del HD-Map (identidad preservada)
  id: "signal_12345"
  semantic: 100  // ← Mantiene semantic_id

  // ✅ Campos de proyección
  region.projection_roi: [850, 300, 40, 80]
  region.crop_roi: [820, 240, 100, 200]
  region.outside_image: false

  // ✅ Campos copiados de la detection asignada
  region.detection_roi: [845, 280, 35, 65]  // ← Bbox real detectado
  region.detect_class_id: TL_VERTICAL_CLASS (0)
  region.detect_score: 0.92
  region.is_detected: true  // ← Tiene detection
  region.is_selected: true

  // ❌ Campos aún vacíos
  status.color: UNKNOWN
  status.confidence: 0.0
  status.blink: false
}
```

**Semáforo NO detectado**:
```cpp
TrafficLight {
  // ✅ Campos del HD-Map (identidad preservada)
  id: "signal_12346"
  semantic: 100  // ← Mismo grupo que el anterior

  // ✅ Campos de proyección
  region.projection_roi: [920, 310, 35, 75]
  region.crop_roi: [895, 225, 87, 187]
  region.outside_image: false

  // ❌ NO tiene detection asignada
  region.detection_roi: [920, 310, 35, 75]  // Mantiene projection_roi
  region.detect_class_id: -1
  region.detect_score: 0.0
  region.is_detected: false  // ← NO detectado
  region.is_selected: false

  // ❌ Campos vacíos
  status.color: UNKNOWN
  status.confidence: 0.0
  status.blink: false
}
```

**Relación clave**:
- M=8 HD-Map lights
- 7 tienen detection asignada (`is_detected = true`)
- 1 no fue detectado (`is_detected = false`)
- 1 detection del buffer original quedó sin asignar

**Observación crítica**: Después de esta etapa, cada `TrafficLight` **mantiene su `id` y `semantic_id` del HD-Map**, que son **persistentes entre frames**.

**Archivo fuente**: `select.cc`

---

## 🔷 ETAPA 4: RECONOCIMIENTO

### ¿Qué recibe?

**Lista de `TrafficLight` objects** (M=8 semáforos):

Estado de cada objeto:
```cpp
TrafficLight {
  // ✅ Campos con identidad (del HD-Map)
  id: "signal_12345"
  semantic: 100

  // ✅ Campos de detección (si fue detectado)
  region.detection_roi: [845, 280, 35, 65]  // O [0,0,0,0] si no detectado
  region.detect_class_id: TL_VERTICAL_CLASS (0)  // O -1 si no detectado
  region.detect_score: 0.92                  // O 0.0 si no detectado
  region.is_detected: true/false

  // ❌ Campos vacíos (se llenarán en esta etapa)
  status.color: UNKNOWN
  status.confidence: 0.0
  status.blink: false
}
```

**También recibe**: La imagen de la cámara (para extraer regiones)

### ¿Qué hace?

**Archivo**: `recognition.cc:48-76`

```cpp
// Línea 51-73
for (base::TrafficLightPtr light : frame->traffic_lights) {

  // CASO 1: NO fue detectado en la etapa anterior
  if (!light->region.is_detected) {
    light->status.color = base::TLColor::TL_UNKNOWN_COLOR;
    light->status.confidence = 0;
    continue;  // Pasar al siguiente
  }

  // CASOS 2-4: SÍ fue detectado → clasificar según tipo
  candidate[0] = light;

  if (light->region.detect_class_id == base::TLDetectionClass::TL_QUADRATE_CLASS) {
    classify_quadrate_->Perform(frame, &candidate);
  } else if (light->region.detect_class_id == base::TLDetectionClass::TL_VERTICAL_CLASS) {
    classify_vertical_->Perform(frame, &candidate);
  } else if (light->region.detect_class_id == base::TLDetectionClass::TL_HORIZONTAL_CLASS) {
    classify_horizontal_->Perform(frame, &candidate);
  } else {
    // ⚠️ ERROR: detect_class_id desconocido
    return false;  // ← Aborta el procesamiento completo
  }
}
```

**⚠️ Validación importante**: Si un semáforo detectado tiene un `detect_class_id` que **NO** es QUADRATE (2), VERTICAL (0), ni HORIZONTAL (1), el sistema **aborta todo el procesamiento** retornando `false`. Esto es una medida de seguridad ante datos corruptos.

#### **Caso 1: NO detectado**

```cpp
// Línea 69-72
if (!light->region.is_detected) {
  light->status.color = base::TLColor::TL_UNKNOWN_COLOR;
  light->status.confidence = 0;
}
```

No hace procesamiento, marca como UNKNOWN.

#### **Caso 2: Detectado como VERTICAL**

**Archivo**: `classify.cc:108-171`

**Paso 1: Extraer región detection_roi de la imagen** (línea 124-127)

```cpp
data_provider_image_option_.crop_roi = light->region.detection_roi;
data_provider_image_option_.do_crop = true;
data_provider_image_option_.target_color = base::Color::BGR;
frame->data_provider->GetImage(data_provider_image_option_, image_.get());
```

Ejemplo: `detection_roi = [845, 280, 35, 65]` → extrae región de 35×65 píxeles en posición (845, 280).

---

**Paso 2: Resize con normalización en GPU** (línea 132-134)

```cpp
const float* mean = mean_.get()->cpu_data();  // [mean_r, mean_g, mean_b]
inference::ResizeGPU(*image_, input_blob_recog,
                     frame->data_provider->src_width(), 0,
                     mean[0], mean[1], mean[2], true, scale_);
```

**Operación interna**:
1. Resize a `[resize_height_, resize_width_]` (típicamente 64×64)
2. Normalización: `pixel_normalized = (pixel - mean) × scale`
3. Transferencia a GPU para inferencia

---

**Paso 3: Inferencia con modelo vertical** (línea 139-141)

```cpp
rt_net_->Infer();
```

**Modelo**: `vert.torch` (red neuronal entrenada para semáforos verticales)
**Output**: `[black_prob, red_prob, yellow_prob, green_prob]`

---

**Paso 4: Convertir probabilidades a color** (línea 146-171)

```cpp
float* out_put_data = output_blob_recog->mutable_cpu_data();
Prob2Color(out_put_data, unknown_threshold_, light);
```

**Función Prob2Color** (línea 151-171):

```cpp
void ClassifyBySimple::Prob2Color(const float* out_put_data, float threshold,
                                  base::TrafficLightPtr light) {
  // Mapping de índices a colores
  std::vector<base::TLColor> status_map = {
      base::TLColor::TL_BLACK,    // 0
      base::TLColor::TL_RED,      // 1
      base::TLColor::TL_YELLOW,   // 2
      base::TLColor::TL_GREEN     // 3
  };

  // Encontrar probabilidad máxima
  std::vector<float> prob(out_put_data, out_put_data + status_map.size());
  auto max_prob = std::max_element(prob.begin(), prob.end());

  // ⚠️ VALIDACIÓN CON THRESHOLD
  int max_color_id = (*max_prob > threshold)
      ? static_cast<int>(std::distance(prob.begin(), max_prob))
      : 0;  // ← Si max_prob <= threshold → BLACK (desconocido)

  light->status.color = status_map[max_color_id];
  light->status.confidence = out_put_data[max_color_id];
}
```

**Ejemplo numérico con alta confianza**:

```
Output del modelo: [0.02, 0.05, 0.08, 0.95]
                   [BLACK, RED,  YELLOW, GREEN]

max_prob = 0.95 (índice 3 = GREEN)
unknown_threshold = 0.5

Validación: 0.95 > 0.5? SÍ
→ max_color_id = 3
→ light->status.color = TL_GREEN
→ light->status.confidence = 0.95
```

**Ejemplo con baja confianza**:

```
Output del modelo: [0.30, 0.25, 0.22, 0.23]
                   [BLACK, RED,  YELLOW, GREEN]

max_prob = 0.30 (índice 0 = BLACK)
unknown_threshold = 0.5

Validación: 0.30 > 0.5? NO
→ max_color_id = 0  ← Forzado a BLACK
→ light->status.color = TL_BLACK
→ light->status.confidence = 0.30
```

**⚠️ Importante**: Si **ninguna** probabilidad supera el `unknown_threshold`, el semáforo se marca como **BLACK** (desconocido/apagado), no como UNKNOWN_COLOR. Esto indica que el modelo no está seguro del color.

---

#### **Caso 3: Detectado como HORIZONTAL**

Mismo proceso pero con modelo `hori.torch` (línea 42-43, 64-65)

#### **Caso 4: Detectado como QUADRATE**

Mismo proceso pero con modelo `quad.torch` (línea 40-41, 56-57)

**¿Por qué modelos separados?**

Los semáforos tienen distribuciones de luces diferentes:
- **Vertical**: Luces apiladas verticalmente (rojo arriba, verde abajo)
- **Horizontal**: Luces en fila horizontal (rojo izquierda, verde derecha)
- **Quadrate**: 4 luces en cuadrado (diferentes patrones)

Cada tipo requiere features visuales distintas → modelos especializados tienen mejor precisión.

---

### 📋 **Configuración de los modelos de reconocimiento**

**Archivo**: `recognition.pb.txt`

| Parámetro | Vertical | Quadrate | Horizontal |
|-----------|----------|----------|------------|
| **Resize** | 32 × 96 | 64 × 64 | 96 × 32 |
| **Mean RGB** | (69.06, 66.58, 66.56) | (69.06, 66.58, 66.56) | (69.06, 66.58, 66.56) |
| **Scale** | 0.01 | 0.01 | 0.01 |
| **Threshold** | 0.5 | 0.5 | 0.5 |
| **Color order** | BGR | BGR | BGR |

**Operación de normalización**:
```python
pixel_normalized = (pixel_bgr - [66.56, 66.58, 69.06]) × 0.01
```

**Observación clave**: Los modelos **vertical** y **horizontal** tienen dimensiones **invertidas**:
- Vertical: 32 ancho × 96 alto (orientación vertical)
- Horizontal: 96 ancho × 32 alto (orientación horizontal)
- Quadrate: 64 × 64 (cuadrado)

Esto permite que cada modelo aprenda features específicas para la orientación de las luces.

### ¿Qué entrega?

**Lista de `TrafficLight` objects actualizada** (M=8 semáforos):

**Ejemplo de semáforos del mismo grupo (semantic_id=100)**:

**Semáforo #1 - Detectado y reconocido**:
```cpp
TrafficLight {
  // ✅ Identidad del HD-Map
  id: "signal_12345"
  semantic: 100

  // ✅ Detección
  region.detection_roi: [845, 280, 35, 65]
  region.detect_class_id: TL_VERTICAL_CLASS (0)
  region.detect_score: 0.92
  region.is_detected: true

  // ✅ Reconocimiento (nuevo)
  status.color: TL_GREEN  // ← Clasificado
  status.confidence: 0.95  // ← Confianza del clasificador

  // ❌ Aún vacío
  status.blink: false
}
```

**Semáforo #2 - Detectado y reconocido**:
```cpp
TrafficLight {
  id: "signal_12346"
  semantic: 100  // ← Mismo grupo
  region.detection_roi: [780, 295, 38, 77]
  region.is_detected: true
  status.color: TL_GREEN  // ← También verde
  status.confidence: 0.88
  status.blink: false
}
```

**Semáforo #3 - NO detectado**:
```cpp
TrafficLight {
  id: "signal_12347"
  semantic: 100  // ← Mismo grupo
  region.is_detected: false
  status.color: TL_UNKNOWN_COLOR  // ← No pudo clasificar
  status.confidence: 0.0
  status.blink: false
}
```

**Archivo fuente**: `recognition.cc`, `classify.cc`

---

## 🔷 ETAPA 5: TRACKING (Semantic Decision)

### ⚠️ NOTA IMPORTANTE SOBRE SEMANTIC_ID

**🔍 HALLAZGO CRÍTICO**: El campo `semantic_id` está **DISEÑADO pero NO IMPLEMENTADO** en Apollo:

```cpp
// traffic_light_region_proposal_component.cc:335-337
int cur_semantic = 0;  // ← SIEMPRE hardcoded a 0
light->semantic = cur_semantic;  // ← Todos los semáforos reciben semantic=0
```

**Evidencia de que NO se usa**:
1. ✅ El código de tracking tiene lógica para manejar `semantic > 0` (voting entre grupos)
2. ✅ La estructura `TrafficLight` tiene el campo `int semantic = 0;`
3. ❌ **PERO** el HD-Map proto de Signal **NO tiene** campo `semantic_id`:
   ```protobuf
   message Signal {
     optional Id id = 1;
     optional Polygon boundary = 2;
     repeated Subsignal subsignal = 3;
     repeated Id overlap_id = 4;  // ← Existe para conectar con lanes
     optional Type type = 5;
     repeated Curve stop_line = 6;
     repeated SignInfo sign_info = 7;
     // ❌ NO existe semantic_id
   }
   ```
4. ❌ El código SIEMPRE asigna `semantic = 0` (nunca lee del mapa)
5. ❌ La documentación oficial de Apollo NO menciona agrupación de semáforos

**Conclusión**: En la práctica, **cada semáforo se trackea de forma individual e independiente**. El mecanismo de voting por grupos está preparado en el código pero nunca se activa porque todos los semáforos tienen `semantic = 0`.

**🎯 Impacto de esta NO-implementación**:

1. **Ventaja perdida de robustez**:
   - Si 3 semáforos del mismo cruce apuntan al vehículo y 2 se clasifican como GREEN y 1 como BLACK (por oclusión), el sistema DEBERÍA usar mayoría (GREEN)
   - En la implementación actual, el semáforo con BLACK mantiene BLACK en su tracking individual

2. **Cada semáforo tiene historial independiente**:
   - Semáforo A puede tener historial: `[GREEN, GREEN, GREEN]`
   - Semáforo B (mismo cruce) puede tener historial: `[BLACK, GREEN, BLACK]`
   - No se ayudan mutuamente a estabilizar la detección

3. **El voting multi-cámara sigue funcionando**:
   - Apollo tiene voting entre múltiples cámaras (6mm, 12mm, 25mm)
   - Esto SÍ ayuda a la robustez (detectar el mismo semáforo desde distintas perspectivas)
   - Pero es diferente al voting por semantic groups (semáforos distintos del mismo cruce)

---

### ¿Qué recibe?

**Lista de `TrafficLight` objects** (M=8 semáforos) con colores actuales:

```cpp
TrafficLight {
  // ✅ Identidad del HD-Map (PERSISTENTE entre frames)
  id: "signal_12345"
  semantic: 0  // ← SIEMPRE 0 (feature no implementada)

  // ✅ Estado actual del frame
  status.color: TL_GREEN  // Clasificación actual
  status.confidence: 0.95

  // ❌ Aún sin revisión temporal
  status.blink: false
}
```

**También recibe**:
- `timestamp`: Momento actual (ej: 1234567890.456 segundos)
- `history_semantic_`: Buffer con historial de estados previos

**Estructura del historial**:
```cpp
std::vector<SemanticTable> history_semantic_ = [
  SemanticTable {
    semantic: "Semantic_0",         // ⚠️ SIEMPRE "Semantic_0" (todos agrupados)
    color: TL_GREEN,                // Último color acordado
    timestamp: 1234567890.400,      // Último update
    light_ids: [0, 1, 2, 3, 4, 5, 6, 7],  // ← TODOS los semáforos del frame
    blink: false,
    last_bright_timestamp: 1234567890.350,
    last_dark_timestamp: 1234567890.100,
    hystertic_window: {
      hysteretic_color: TL_GREEN,
      hysteretic_count: 0
    }
  }
  // ⚠️ En práctica, solo existe UN grupo (semantic_0)
]
```

### ¿Qué hace?

Esta etapa mejora la estabilidad temporal usando el historial. Los semáforos no cambian instantáneamente en el mundo real.

### 📋 **Configuración del tracking**

**Archivo**: `semantic.pb.txt`

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **revise_time_second** | 1.5s | Ventana temporal: si Δt > 1.5s → resetear historial |
| **blink_threshold_second** | 0.55s | Tiempo mínimo oscuro para detectar blink |
| **hysteretic_threshold_count** | 1 | Frames necesarios para salir de BLACK (count > 1 → 2 frames) |
| **non_blink_threshold_second** | 1.1s | Tiempo máximo entre bright/dark para mantener blink (= 2 × blink_threshold) |

---

**REGLAS CLAVE DEL TRACKING:**

1. **~~Voting por Semantic Group~~**: ⚠️ **DISEÑADO pero NO USADO** (todos tienen semantic=0, así que votan TODOS los semáforos juntos independientemente de su intersección)
2. **Hysteresis**: Requiere 2 frames consecutivos para salir del estado BLACK (hysteretic_threshold = 1, condición: count > 1)
3. **Blink Detection**: Detecta intermitencia verde (flecha verde parpadeante)
4. **🚨 REGLA DE SECUENCIA TEMPORAL (Traffic Safety Rule)**:

   > *"Because of the time sequence, yellow only exists after green and before red.
   > Any yellow after red is reset to red for the sake of safety until green displays."*

   **Secuencia válida del mundo real:**
   ```
   GREEN → YELLOW → RED → GREEN → YELLOW → RED ...
         ✅        ✅     ✅      ✅        ✅
   ```

   **Secuencia INVÁLIDA detectada por Apollo:**
   ```
   ... → RED → YELLOW ← ❌ IMPOSIBLE EN EL MUNDO REAL
              └──→ FORZAR A RED (safety override)
   ```

   **Razones para esta regla:**
   - En intersecciones reales, YELLOW solo aparece en la transición GREEN→RED
   - Si detectamos YELLOW después de RED, es un **error de clasificación** o **reflejo**
   - Por **seguridad**, Apollo mantiene RED hasta confirmar GREEN

   **Implementación**: Líneas 174-182 de `semantic_decision.cc`

---

#### **Paso 1: Agrupar por Semantic ID** ⚠️ (EN PRÁCTICA: UN SOLO GRUPO)

**Archivo**: `semantic_decision.cc:239-280`

```cpp
// Línea 252-279
std::vector<SemanticTable> semantic_table;

for (size_t i = 0; i < lights_ref.size(); i++) {
  base::TrafficLightPtr light = lights_ref.at(i);
  int cur_semantic = light->semantic;  // ← ⚠️ SIEMPRE es 0 en la implementación actual

  SemanticTable tmp;
  std::stringstream ss;

  if (cur_semantic > 0) {
    // ❌ Esta rama NUNCA se ejecuta (cur_semantic siempre es 0)
    ss << "Semantic_" << cur_semantic;  // "Semantic_100"
  } else {
    // ✅ SIEMPRE se ejecuta esta rama
    ss << "No_semantic_light_" << light->id;  // "No_semantic_light_signal_12345"
  }

  tmp.semantic = ss.str();
  tmp.light_ids.push_back(static_cast<int>(i));  // Índice en el frame actual
  tmp.color = light->status.color;
  tmp.time_stamp = time_stamp;
  tmp.blink = false;

  // Buscar si ya existe este semantic en la tabla temporal
  auto iter = std::find_if(semantic_table.begin(), semantic_table.end(),
                           boost::bind(compare, _1, tmp));

  if (iter != semantic_table.end()) {
    // ❌ NUNCA se ejecuta (cada light->id es único)
    iter->light_ids.push_back(static_cast<int>(i));  // Agregar al grupo
  } else {
    // ✅ SIEMPRE se ejecuta (cada semáforo crea su propio grupo)
    semantic_table.push_back(tmp);  // Nuevo grupo
  }
}
```

**⚠️ Comportamiento REAL vs DISEÑADO**:

**Ejemplo DISEÑADO** (como debería funcionar si semantic_id se implementara):
```cpp
lights_ref = [
  TrafficLight { id:"signal_12345", semantic:100, color:GREEN },  // idx 0
  TrafficLight { id:"signal_12346", semantic:100, color:GREEN },  // idx 1
  TrafficLight { id:"signal_12347", semantic:100, color:BLACK },  // idx 2
  TrafficLight { id:"signal_12348", semantic:200, color:RED },    // idx 3
]

// Resultado ESPERADO del agrupamiento:
semantic_table = [
  SemanticTable {
    semantic: "Semantic_100",
    light_ids: [0, 1, 2],  // Tres semáforos del mismo cruce
    color: ???  // Se calculará por voting
  },
  SemanticTable {
    semantic: "Semantic_200",
    light_ids: [3],  // Semáforo de otro cruce
    color: ???
  }
]
```

**Ejemplo REAL** (lo que realmente sucede con semantic=0):
```cpp
lights_ref = [
  TrafficLight { id:"signal_12345", semantic:0, color:GREEN },  // idx 0
  TrafficLight { id:"signal_12346", semantic:0, color:GREEN },  // idx 1
  TrafficLight { id:"signal_12347", semantic:0, color:BLACK },  // idx 2
  TrafficLight { id:"signal_12348", semantic:0, color:RED },    // idx 3
]

// Resultado REAL del agrupamiento (cada semáforo en su propio "grupo"):
semantic_table = [
  SemanticTable {
    semantic: "No_semantic_light_signal_12345",
    light_ids: [0],  // ← Solo un semáforo
    color: GREEN
  },
  SemanticTable {
    semantic: "No_semantic_light_signal_12346",
    light_ids: [1],  // ← Solo un semáforo
    color: GREEN
  },
  SemanticTable {
    semantic: "No_semantic_light_signal_12347",
    light_ids: [2],  // ← Solo un semáforo
    color: BLACK
  },
  SemanticTable {
    semantic: "No_semantic_light_signal_12348",
    light_ids: [3],  // ← Solo un semáforo
    color: RED
  }
]

// ⚠️ Cada "grupo" tiene UN SOLO elemento, así que el voting es trivial
```

#### **Paso 2: Voting dentro del grupo** ⚠️ (EN PRÁCTICA: SIEMPRE 1 VOTO)

**Archivo**: `semantic_decision.cc:96-138` (función `ReviseBySemantic`)

```cpp
// Línea 98-106
std::vector<int> vote(static_cast<int>(base::TLColor::TL_TOTAL_COLOR_NUM), 0);

for (size_t i = 0; i < semantic_table.light_ids.size(); ++i) {
  int index = semantic_table.light_ids.at(i);
  base::TrafficLightPtr light = lights_ref[index];
  auto color = light->status.color;
  vote.at(static_cast<int>(color))++;  // Incrementar voto
}
```

**Ejemplo DISEÑADO para grupo "Semantic_100"** (si semantic_id funcionara):

```
light_ids = [0, 1, 2]  // ← Tres semáforos agrupados

Semáforo 0: color = GREEN → vote[GREEN]++
Semáforo 1: color = GREEN → vote[GREEN]++
Semáforo 2: color = BLACK → vote[BLACK]++

Resultado del voting:
vote[RED] = 0
vote[GREEN] = 2  // ← Mayoría clara
vote[YELLOW] = 0
vote[BLACK] = 1
vote[UNKNOWN] = 0
```

**Ejemplo REAL** (lo que realmente pasa con semantic=0):

```
light_ids = [0]  // ← ⚠️ Un solo semáforo en el "grupo"

Semáforo 0: color = GREEN → vote[GREEN]++

Resultado del voting (trivial):
vote[RED] = 0
vote[GREEN] = 1  // ← Único voto
vote[YELLOW] = 0
vote[BLACK] = 0
vote[UNKNOWN] = 0

// ⚠️ El "voting" siempre retorna el color del único semáforo
//    No hay corrección por consenso porque no hay grupo
```

**Determinar color ganador**:

```cpp
// Línea 109-137
if ((vote[RED] == 0) && (vote[GREEN] == 0) && (vote[YELLOW] == 0)) {
  // Solo hay BLACK o UNKNOWN
  if (vote[BLACK] > 0) {
    return base::TLColor::TL_BLACK;
  } else {
    return base::TLColor::TL_UNKNOWN_COLOR;
  }
}

// Ignorar BLACK y UNKNOWN para el voting principal
vote[BLACK] = 0;
vote[UNKNOWN] = 0;

// Encontrar el color con más votos
auto biggest = std::max_element(std::begin(vote), std::end(vote));
int max_color_num = *biggest;
max_color = base::TLColor(std::distance(std::begin(vote), biggest));

// 🚨 IMPORTANTE: Eliminar el ganador del vector para buscar el segundo
vote.erase(biggest);

// Buscar el segundo lugar (ahora es el máximo del vector sin el primero)
auto second_biggest = std::max_element(std::begin(vote), std::end(vote));

// Verificar si hay empate (max == second)
if (max_color_num == *second_biggest) {
  return TL_UNKNOWN_COLOR;  // Empate → no confiar
} else {
  return max_color;  // GREEN en nuestro ejemplo (2 votos)
}
```

**Ejemplo con empate**:
```
vote[RED] = 2
vote[GREEN] = 2
vote[YELLOW] = 0

biggest = RED (2 votos)
vote.erase(RED) → vote ahora tiene [GREEN:2, YELLOW:0]
second_biggest = GREEN (2 votos)

max_color_num (2) == second_biggest (2) ✓
→ return UNKNOWN_COLOR  // No confiar en empates
```

**Resultado**: `cur_color = GREEN` (por mayoría 2 vs 1)

#### **Paso 3: Buscar en historial**

**Archivo**: `semantic_decision.cc:165-169`

```cpp
// Línea 165-169
std::vector<SemanticTable>::iterator iter =
    std::find_if(std::begin(history_semantic_), std::end(history_semantic_),
                 boost::bind(compare, _1, semantic_table));

if (iter != history_semantic_.end()) {
  // Encontró historial previo para este grupo
  pre_color = iter->color;
  ...
}
```

**Si encuentra** (existe en historial):
```cpp
iter->color = TL_GREEN
iter->timestamp = 1234567890.400
iter->blink = false
```

#### **Paso 4: Revisión temporal (si existe historial)**

**Archivo**: `semantic_decision.cc:171-213`

**⚙️ FUNCIONES CLAVE PARA CAMBIAR COLORES:**

Antes de ver el switch, es importante entender las dos funciones que modifican los colores:

1. **`ReviseLights(lights, light_ids, dst_color)`** (línea 140-149):
   ```cpp
   void ReviseLights(std::vector<base::TrafficLightPtr> *lights,
                     const std::vector<int> &light_ids,
                     base::TLColor dst_color) {
     // SOBRESCRIBE el color de todos los semáforos del grupo
     for (auto index : light_ids) {
       lights->at(index)->status.color = dst_color;  // ← Fuerza este color
     }
   }
   ```
   **Propósito:** Rechazar la detección actual y forzar un color específico (seguridad).

2. **`UpdateHistoryAndLights(cur, lights, history)`** (línea 69-94):
   ```cpp
   void UpdateHistoryAndLights(...) {
     (*history)->time_stamp = cur.time_stamp;

     if ((*history)->color == TL_BLACK) {
       // Histéresis para BLACK (requiere 2 frames consecutivos)
       // ...
     } else {
       // Acepta el cambio
       (*history)->color = cur.color;  // ← Actualiza historial al nuevo color
     }
   }
   ```
   **Propósito:** Aceptar la detección actual y actualizar el historial.

**Diferencia clave:**
- `ReviseLights()` = **RECHAZAR** detección (sobrescribir con color anterior/forzado)
- `UpdateHistoryAndLights()` = **ACEPTAR** detección (actualizar historial)

---

```cpp
// Línea 171
if (time_stamp - iter->timestamp < revise_time_s_) {
  // Dentro de ventana temporal (1.5 segundos)

  switch (cur_color) {
    case TL_YELLOW:
      // 🚨 REGLA DE SECUENCIA TEMPORAL (Traffic Safety Rule)
      // "Because of the time sequence, yellow only exists after green and before red.
      //  Any yellow after red is reset to red for the sake of safety until green displays."
      //
      // Secuencia válida del mundo real:
      //   GREEN → YELLOW → RED → GREEN → YELLOW → RED ...
      //
      // Si detectamos YELLOW pero el estado anterior era RED, es un ERROR:
      //   - Puede ser falso positivo (reflejo amarillo)
      //   - Puede ser error de clasificación
      //   - NUNCA puede ser válido en el mundo real
      //
      // Acción: FORZAR a RED por seguridad
      if (iter->color == TL_RED) {
        // Estado anterior: RED
        // Estado detectado: YELLOW ← INVÁLIDO
        // → Mantener RED hasta que veamos GREEN

        // ⚙️ DÓNDE CAMBIA EL COLOR (llamada a ReviseLights):
        ReviseLights(lights, semantic_table.light_ids, iter->color);
        //                                              └─────┬─────┘
        //                                              iter->color = RED
        //
        // Dentro de ReviseLights():
        //   for (auto index : light_ids) {  // [0, 1, 2]
        //     lights->at(index)->status.color = RED;  // ← SOBRESCRIBE a RED
        //   }
        //
        // Resultado: Todos los semáforos del grupo ahora tienen RED,
        //            ignorando el YELLOW que detectó el clasificador.

        iter->time_stamp = time_stamp;
        iter->hystertic_window.hysteretic_count = 0;

        ADEBUG << "YELLOW after RED detected - maintaining RED for safety";

      } else {
        // Estado anterior: GREEN, BLACK, o UNKNOWN
        // Estado detectado: YELLOW ← VÁLIDO (puede venir después de GREEN)
        // → Aceptar el cambio

        // ⚙️ DÓNDE CAMBIA EL COLOR (llamada a UpdateHistoryAndLights):
        UpdateHistoryAndLights(semantic_table, lights, &iter);
        //
        // Dentro de UpdateHistoryAndLights():
        //   (*history)->color = cur.color;  // cur.color = YELLOW
        //                                    // → iter->color = YELLOW
        //
        // Resultado: iter->color actualizado a YELLOW,
        //            los semáforos mantienen el YELLOW detectado.

        ADEBUG << "YELLOW after " << s_color_strs[iter->color] << " - accepted";
      }
      break;

    case TL_RED:
    case TL_GREEN:
      // Alta confianza → aceptar cambio
      UpdateHistoryAndLights(semantic_table, lights, &iter);

      // Actualizar timestamps para blink detection
      if (time_stamp - iter->last_bright_time_stamp > blink_threshold_s_ &&
          iter->last_dark_time_stamp > iter->last_bright_time_stamp) {
        iter->blink = true;
      }
      iter->last_bright_time_stamp = time_stamp;
      break;

    case TL_BLACK:
      // Semáforo "apagado" → resetear y aplicar hysteresis
      iter->last_dark_time_stamp = time_stamp;

      // 🚨 IMPORTANTE: BLACK resetea el contador de histéresis
      // Si estaba en medio de una transición (ej: BLACK→GREEN count=2),
      // al volver a BLACK se pierde el progreso
      iter->hystertic_window.hysteretic_count = 0;

      if (iter->color == TL_UNKNOWN_COLOR || iter->color == TL_BLACK) {
        // Ya estaba apagado/desconocido → aceptar BLACK
        iter->time_stamp = time_stamp;
        UpdateHistoryAndLights(semantic_table, lights, &iter);
      } else {
        // Estaba encendido (RED/GREEN/YELLOW) → mantener color anterior
        // Aplicar hysteresis: esperar 2 frames consecutivos del nuevo color
        // antes de aceptar el cambio desde BLACK
        ReviseLights(lights, semantic_table.light_ids, iter->color);
      }
      break;

    case TL_UNKNOWN_COLOR:
    default:
      // Baja confianza → mantener color anterior
      ReviseLights(lights, semantic_table.light_ids, iter->color);
      break;
  }
} else {
  // 🚨 VENTANA TEMPORAL EXPIRADA (>1.5 segundos sin detecciones)
  // Línea 210-213
  //
  // Si pasó mucho tiempo sin detectar este semantic group:
  // - Puede ser oclusión prolongada (ej: camión bloqueó vista)
  // - Puede ser cambio de escena (giró en intersección)
  //
  // Acción: RESETEAR historial y aceptar color actual SIN VALIDACIÓN
  iter->time_stamp = time_stamp;
  iter->color = cur_color;  // Acepta directamente, sin reglas de secuencia

  ADEBUG << "Temporal window expired, resetting history for semantic "
         << semantic_table.semantic;
}
```

**⚠️ IMPORTANTE**: Cuando la ventana temporal expira:
- ❌ NO se aplica la regla de secuencia YELLOW
- ❌ NO se aplica histéresis
- ❌ NO se valida contra estado anterior
- ✅ Se acepta el color actual como "nuevo comienzo"

**Ejemplo de ventana expirada**:
```
Frame N:
  iter->color = RED
  iter->timestamp = 1234567890.000

[Camión bloquea vista por 2 segundos]

Frame N+60 (2s después):
  cur_color = YELLOW (detectado)
  timestamp = 1234567892.000

  Δt = 2.0s > 1.5s ✓ (ventana expiró)

  → else branch (línea 210-213):
    → iter->color = YELLOW  // Acepta sin validación
    → NO verifica regla de secuencia (RED → YELLOW inválido)
    → Trata como "primer frame" después de oclusión
```

**Función `UpdateHistoryAndLights`** (`semantic_decision.cc:69-94`):

```cpp
// Línea 72-93
iter->time_stamp = cur.time_stamp;

if (iter->color == base::TLColor::TL_BLACK) {
  // Hysteresis para BLACK
  if (iter->hystertic_window.hysteretic_color == cur.color) {
    iter->hystertic_window.hysteretic_count++;
  } else {
    iter->hystertic_window.hysteretic_color = cur.color;
    iter->hystertic_window.hysteretic_count = 1;
  }

  if (iter->hystertic_window.hysteretic_count > hysteretic_threshold_) {
    // Después de 2 frames consecutivos (count > 1) → aceptar cambio
    iter->color = cur.color;
    iter->hystertic_window.hysteretic_count = 0;
  } else {
    // Mantener BLACK
    ReviseLights(lights, cur.light_ids, iter->color);
  }
} else {
  // Transición normal
  iter->color = cur.color;
}
```

**Ejemplo 1: Caso normal (GREEN → GREEN)**

```
Estado previo (historial):
  iter->color = GREEN
  iter->timestamp = 1234567890.400

Estado actual (frame nuevo):
  cur_color = GREEN (por voting: 2 GREEN, 1 BLACK)
  timestamp = 1234567890.456

Δt = 0.456 - 0.400 = 0.056s < 1.5s ✓ (dentro de ventana temporal)

Switch(cur_color = GREEN):
  → Case TL_GREEN:
    → UpdateHistoryAndLights()
    → iter->color = GREEN (acepta)
    → iter->last_bright_timestamp = 1234567890.456
    → Revisar blink detection
```

**Ejemplo 2: 🚨 Regla de secuencia temporal (RED → YELLOW inválido)**

```
Frame N-1:
  iter->color = RED
  iter->timestamp = 1234567890.400

Frame N (actual):
  cur_color = YELLOW (voting detectó YELLOW)
  timestamp = 1234567890.450

Δt = 0.050s < 1.5s ✓ (dentro de ventana temporal)

Switch(cur_color = YELLOW):
  → Case TL_YELLOW:
    → if (iter->color == TL_RED) ← ✅ TRUE
      → ❌ SECUENCIA INVÁLIDA: RED → YELLOW
      → 🚨 SAFETY OVERRIDE: Mantener RED
      → ReviseLights(lights, light_ids, RED)
      → iter->color = RED  (NO cambia)
      → iter->timestamp = 1234567890.450 (actualiza timestamp)

Resultado:
  - Todos los semáforos del grupo reportan: RED
  - El YELLOW detectado se IGNORA por seguridad
  - Sistema esperará hasta ver GREEN antes de aceptar cualquier cambio
```

**Ejemplo 3: Secuencia válida (GREEN → YELLOW)**

```
Frame N-1:
  iter->color = GREEN
  iter->timestamp = 1234567890.400

Frame N (actual):
  cur_color = YELLOW (voting detectó YELLOW)
  timestamp = 1234567890.450

Δt = 0.050s < 1.5s ✓ (dentro de ventana temporal)

Switch(cur_color = YELLOW):
  → Case TL_YELLOW:
    → if (iter->color == TL_RED) ← ❌ FALSE (era GREEN)
    → else:
      → ✅ SECUENCIA VÁLIDA: GREEN → YELLOW
      → UpdateHistoryAndLights()
      → iter->color = YELLOW (acepta el cambio)
      → iter->timestamp = 1234567890.450

Resultado:
  - Todos los semáforos del grupo reportan: YELLOW
  - Transición aceptada (GREEN → YELLOW es normal)
```

**Ejemplo 4: Continuación - esperando GREEN después de override**

```
Frame N (estado actual después de override):
  iter->color = RED (forzado)
  iter->timestamp = 1234567890.450

Frame N+1:
  cur_color = YELLOW (sigue detectando YELLOW erróneo)
  timestamp = 1234567890.500

Switch(cur_color = YELLOW):
  → Case TL_YELLOW:
    → if (iter->color == TL_RED) ← ✅ TRUE
      → 🚨 Mantener RED otra vez

Frame N+2:
  cur_color = RED (ahora detecta correctamente RED)
  timestamp = 1234567890.550

Switch(cur_color = RED):
  → Case TL_RED:
    → UpdateHistoryAndLights()
    → iter->color = RED (confirma RED)

Frame N+10:
  cur_color = GREEN (finalmente cambia a GREEN)
  timestamp = 1234567891.500

Switch(cur_color = GREEN):
  → Case TL_GREEN:
    → UpdateHistoryAndLights()
    → iter->color = GREEN ✅ (AHORA sí puede cambiar)

Resultado:
  - Sistema mantuvo RED hasta confirmar GREEN
  - Secuencia segura: RED → (espera) → GREEN
  - Próximo YELLOW será válido (después de GREEN)
```

#### **Paso 5: Detección de Blink (intermitencia)**

Solo para semáforos VERDES:

```cpp
// Línea 187-190
if (time_stamp - iter->last_bright_time_stamp > blink_threshold_s_ &&
    iter->last_dark_time_stamp > iter->last_bright_time_stamp) {
  iter->blink = true;
}
iter->last_bright_time_stamp = time_stamp;
```

**Lógica**:
```
Patrón normal (no intermitente):
  BRIGHT ───────────────────────> (siempre bright)

Patrón intermitente:
  BRIGHT ─── DARK(>0.4s) ─── BRIGHT ─── DARK ─── BRIGHT
             ↑                           ↑
             last_dark                   detecta blink

Condiciones para blink = true:
  1. last_dark_timestamp > last_bright_timestamp (hubo un periodo oscuro)
  2. time_since_last_dark > 0.4s (suficiente tiempo oscuro)
  3. Ahora está BRIGHT de nuevo
```

```cpp
// Línea 216-225
// Reset blink flag si:
// 1. El color cambió desde el frame anterior
// 2. Pasó mucho tiempo (>0.8s) sin alternancia bright/dark
if (pre_color != iter->color ||
    fabs(iter->last_dark_time_stamp - iter->last_bright_time_stamp) >
        non_blink_threshold_s_) {
  iter->blink = false;
}

// 🚨 REGLA CRÍTICA: Solo semáforos VERDES pueden parpadear
// Línea 222-225
for (auto index : semantic_table.light_ids) {
  lights_ref[index]->status.blink =
      (iter->blink && iter->color == base::TLColor::TL_GREEN);
      //                └────────────────┬───────────────┘
      //                        Blink solo si es GREEN
}
```

**Razón**: En el mundo real, solo las **flechas verdes** parpadean (giro permitido pero con precaución). Los semáforos rojos y amarillos **nunca** parpadean según estándares de tránsito.

#### **Paso 6: Aplicar a todos los semáforos del grupo**

**Archivo**: `semantic_decision.cc:140-149` (función `ReviseLights`)

```cpp
// Línea 143-148
void SemanticReviser::ReviseLights(std::vector<base::TrafficLightPtr> *lights,
                                   const std::vector<int> &light_ids,
                                   base::TLColor dst_color) {
  for (auto index : light_ids) {
    lights->at(index)->status.color = dst_color;
  }
}
```

**Aplicación del voting al grupo**:

```
light_ids = [0, 1, 2]
dst_color = GREEN (decidido por voting/revisión)

lights[0]->status.color = GREEN  ✓
lights[1]->status.color = GREEN  ✓ (ya era GREEN)
lights[2]->status.color = GREEN  ✓ (corrigió desde BLACK)
```

**También aplicar blink status** (`semantic_decision.cc:222-224`):

```cpp
// Línea 222-224
for (auto index : semantic_table.light_ids) {
  lights_ref[index]->status.blink =
      (iter->blink && iter->color == base::TLColor::TL_GREEN);
}
```

#### **Paso 7: Actualizar historial**

```cpp
// Línea 233-235 (si NO existía en historial)
if (iter == history_semantic_.end()) {
  semantic_table.last_dark_time_stamp = semantic_table.time_stamp;
  semantic_table.last_bright_time_stamp = semantic_table.time_stamp;
  history_semantic_.push_back(semantic_table);
}
```

### ¿Qué entrega?

**Lista de `TrafficLight` objects con estados estabilizados** (M=8 semáforos):

**Ejemplo completo del grupo semantic_id=100**:

**Semáforo #1 (signal_12345)**:
```cpp
TrafficLight {
  // ✅ Identidad del HD-Map (PERSISTENTE)
  id: "signal_12345"
  semantic: 100  // ← Grupo para voting

  // ✅ Detección
  region.projection_roi: [850, 300, 40, 80]
  region.detection_roi: [845, 280, 35, 65]
  region.crop_roi: [820, 240, 100, 200]
  region.is_detected: true
  region.detect_class_id: TL_VERTICAL_CLASS (0)
  region.detect_score: 0.92

  // ✅ Reconocimiento + Tracking (FINAL)
  status.color: TL_GREEN  // Clasificado + revisado temporalmente
  status.confidence: 0.95
  status.blink: false     // No intermitente
}
```

**Semáforo #2 (signal_12346)**:
```cpp
TrafficLight {
  id: "signal_12346"
  semantic: 100  // ← Mismo grupo
  region.detection_roi: [780, 295, 38, 77]
  region.is_detected: true
  status.color: TL_GREEN  // Por voting (mismo que grupo)
  status.confidence: 0.88
  status.blink: false
}
```

**Semáforo #3 (signal_12347)**:
```cpp
TrafficLight {
  id: "signal_12347"
  semantic: 100  // ← Mismo grupo
  region.is_detected: false  // NO detectado
  status.color: TL_GREEN  // ← CORREGIDO por voting (era BLACK)
  status.confidence: 0.0   // Baja confianza (no detectado)
  status.blink: false
}
```

**Observación crítica**: El semáforo #3 NO fue detectado (clasificó como UNKNOWN), pero el **voting del grupo** lo corrigió a GREEN porque los otros 2 semáforos del mismo `semantic_id` detectaron GREEN.

**Historial actualizado**:
```cpp
history_semantic_ = [
  SemanticTable {
    semantic: "Semantic_100",
    color: TL_GREEN,                // Acordado por el grupo
    timestamp: 1234567890.456,      // Frame actual
    light_ids: [0, 1, 2],           // Índices en frame actual
    blink: false,
    last_bright_timestamp: 1234567890.456,
    last_dark_timestamp: 1234567890.100,
    hystertic_window: {
      hysteretic_color: TL_GREEN,
      hysteretic_count: 0
    }
  },
  ...
]
```

**Archivo fuente**: `semantic_decision.cc`

---

## 🔗 **PERSISTENCIA DEL HISTORIAL ENTRE FRAMES**

### ¿Cómo se mantiene el historial de cada semáforo?

**CLAVE**: El historial está relacionado al **`id` del HD-Map** (no al índice del frame).

**Archivo**: `semantic_decision.cc:252-279`

```cpp
// Línea 252-264: Crear clave de búsqueda para cada semáforo
for (size_t i = 0; i < lights_ref.size(); i++) {
  base::TrafficLightPtr light = lights_ref.at(i);
  int cur_semantic = light->semantic;  // ⚠️ Siempre 0 en implementación actual

  SemanticTable tmp;
  std::stringstream ss;

  if (cur_semantic > 0) {
    // ❌ Diseñado (NO usado): agrupar por semantic_id
    ss << "Semantic_" << cur_semantic;  // "Semantic_100"
  } else {
    // ✅ Real: usar ID del HD-Map como clave única
    ss << "No_semantic_light_" << light->id;
    //                           └──────┬─────┘
    //                        ID del HD-Map (persistente)
    // Ejemplo: "No_semantic_light_signal_12345"
  }

  tmp.semantic = ss.str();  // ← Esta es la CLAVE de búsqueda
  tmp.light_ids.push_back(static_cast<int>(i));  // Índice EN ESTE frame
  tmp.color = light->status.color;
  tmp.time_stamp = time_stamp;
}
```

**Función de búsqueda en historial** (línea 165-169):

```cpp
std::vector<SemanticTable>::iterator iter =
    std::find_if(std::begin(history_semantic_), std::end(history_semantic_),
                 boost::bind(compare, _1, semantic_table));

// compare (línea 35-37):
bool compare(const SemanticTable &s1, const SemanticTable &s2) {
  return s1.semantic == s2.semantic;  // Compara el STRING generado arriba
}
```

---

### 📊 **Ejemplo: Persistencia frame a frame**

**Frame 1:**
```cpp
lights_ref = [
  TrafficLight { id: "signal_12345", semantic: 0, color: GREEN },  // índice [0]
  TrafficLight { id: "signal_99999", semantic: 0, color: RED }     // índice [1]
]

Agrupamiento:
  semantic_table = [
    { semantic: "No_semantic_light_signal_12345", light_ids: [0], color: GREEN },
    { semantic: "No_semantic_light_signal_99999", light_ids: [1], color: RED }
  ]

Historial creado (primera vez):
  history_semantic_ = [
    {
      semantic: "No_semantic_light_signal_12345",  // ← CLAVE PERSISTENTE
      color: GREEN,
      light_ids: [0],  // ← índice en Frame 1
      timestamp: 1234567890.100,
      ...
    },
    {
      semantic: "No_semantic_light_signal_99999",
      color: RED,
      light_ids: [1],
      timestamp: 1234567890.100,
      ...
    }
  ]
```

**Frame 2 (orden diferente, 0.05s después):**
```cpp
lights_ref = [
  TrafficLight { id: "signal_99999", semantic: 0, color: RED },    // ⚠️ ahora índice [0]
  TrafficLight { id: "signal_12345", semantic: 0, color: GREEN }   // ⚠️ ahora índice [1]
]

Agrupamiento:
  semantic_table = [
    { semantic: "No_semantic_light_signal_99999", light_ids: [0], color: RED },
    { semantic: "No_semantic_light_signal_12345", light_ids: [1], color: GREEN }
  ]

Búsqueda en historial:
  // Para "No_semantic_light_signal_99999":
  iter = find("No_semantic_light_signal_99999" en history_semantic_)
  → ✅ Encuentra: history_semantic_[1]
  → Δt = 0.05s < 1.5s ✓ (ventana temporal válida)
  → Aplica reglas de tracking usando historial previo
  → Actualiza: light_ids = [0] (nuevo índice en Frame 2)

  // Para "No_semantic_light_signal_12345":
  iter = find("No_semantic_light_signal_12345" en history_semantic_)
  → ✅ Encuentra: history_semantic_[0]
  → Δt = 0.05s < 1.5s ✓
  → Aplica reglas de tracking
  → Actualiza: light_ids = [1] (nuevo índice en Frame 2)

Historial actualizado:
  history_semantic_ = [
    {
      semantic: "No_semantic_light_signal_12345",
      color: GREEN,
      light_ids: [1],  // ← ACTUALIZADO al índice del Frame 2
      timestamp: 1234567890.150,  // ← ACTUALIZADO
      ...
    },
    {
      semantic: "No_semantic_light_signal_99999",
      color: RED,
      light_ids: [0],  // ← ACTUALIZADO
      timestamp: 1234567890.150,
      ...
    }
  ]
```

**Frame 3 (signal_12345 no detectado, 0.05s después):**
```cpp
lights_ref = [
  TrafficLight { id: "signal_99999", semantic: 0, color: RED }  // solo este
]

Agrupamiento:
  semantic_table = [
    { semantic: "No_semantic_light_signal_99999", light_ids: [0], color: RED }
  ]

Búsqueda en historial:
  // Para "No_semantic_light_signal_99999":
  → ✅ Encuentra y actualiza

  // ⚠️ "No_semantic_light_signal_12345" NO se busca (no está en lights_ref)

Historial:
  history_semantic_ = [
    {
      semantic: "No_semantic_light_signal_12345",
      color: GREEN,
      light_ids: [1],  // ← NO actualizado (mantiene índice del Frame 2)
      timestamp: 1234567890.150,  // ← NO actualizado
      ...
    },
    {
      semantic: "No_semantic_light_signal_99999",
      color: RED,
      light_ids: [0],
      timestamp: 1234567890.200,  // ← ACTUALIZADO
      ...
    }
  ]
  // ⚠️ El historial de signal_12345 queda "congelado" en Frame 2
```

**Frame 4 (signal_12345 vuelve a aparecer, 2.0s después del Frame 2):**
```cpp
lights_ref = [
  TrafficLight { id: "signal_12345", semantic: 0, color: YELLOW },  // reaparece
  TrafficLight { id: "signal_99999", semantic: 0, color: RED }
]

Agrupamiento:
  semantic_table = [
    { semantic: "No_semantic_light_signal_12345", light_ids: [0], color: YELLOW },
    { semantic: "No_semantic_light_signal_99999", light_ids: [1], color: RED }
  ]

Búsqueda en historial:
  // Para "No_semantic_light_signal_12345":
  iter = find("No_semantic_light_signal_12345" en history_semantic_)
  → ✅ Encuentra: history_semantic_[0]
  → Δt = 2.0s > 1.5s ❌ (ventana temporal EXPIRADA)

  // 🚨 RESETEO DE HISTORIAL (línea 210-213):
  iter->timestamp = timestamp_actual;
  iter->color = cur_color;  // Acepta YELLOW sin validación
  // ⚠️ NO aplica regla de secuencia temporal
  // ⚠️ NO aplica hysteresis
  // Trata como "nuevo comienzo" después de oclusión prolongada

Historial actualizado:
  history_semantic_ = [
    {
      semantic: "No_semantic_light_signal_12345",
      color: YELLOW,  // ← RESETEADO directamente
      light_ids: [0],  // ← índice en Frame 4
      timestamp: 1234567892.150,  // ← ACTUALIZADO
      ...
    },
    ...
  ]
```

---

### ✅ **Conclusión: Persistencia del historial**

**El historial se mantiene usando:**
1. **Clave de búsqueda**: String formado con el `id` del HD-Map
   - Ejemplo: `"No_semantic_light_signal_12345"`
   - **Persistente** entre frames (mismo semáforo físico = mismo ID)

2. **Índices flexibles**: `light_ids` se actualiza cada frame
   - **NO persistente** (cambia según orden de detección)

3. **Ventana temporal**: 1.5 segundos
   - Si Δt > 1.5s → resetea historial (oclusión prolongada)
   - Si Δt ≤ 1.5s → aplica reglas de tracking temporal

**Ventajas**:
- ✅ Tracking robusto ante cambios de orden en `lights_ref`
- ✅ Mantiene historial aunque el semáforo no se detecte temporalmente
- ✅ Reseteo automático después de oclusiones prolongadas

**Limitación actual**:
- ⚠️ Cada semáforo tiene historial **individual** (semantic_id = 0)
- ⚠️ NO hay corrección por consenso entre semáforos del mismo cruce

---

## 📤 SALIDA FINAL

Después de las 5 etapas, Apollo tiene una lista de `TrafficLight` objects con toda la información:

**Estructura completa de TrafficLight (salida final)**:

```cpp
TrafficLight #1 (signal_12345) {
  // ═══════════════════════════════════════════
  // IDENTIDAD (del HD-Map, PERSISTENTE)
  // ═══════════════════════════════════════════
  id: "signal_12345"              // ID único del semáforo
  semantic: 0                     // ⚠️ SIEMPRE 0 (semantic_id NO implementado)

  // ═══════════════════════════════════════════
  // GEOMETRÍA 3D (del HD-Map)
  // ═══════════════════════════════════════════
  region.points: [
    (500.23, 1200.45, 5.12),
    (500.28, 1200.50, 5.12),
    (500.28, 1200.50, 5.92),
    (500.23, 1200.45, 5.92)
  ]

  // ═══════════════════════════════════════════
  // PROYECCIÓN (calculada en preprocesamiento)
  // ═══════════════════════════════════════════
  region.projection_roi: [850, 300, 40, 80]  // Dónde DEBERÍA aparecer
  region.crop_roi: [736, 206, 270, 270]      // ROI cuadrada (max_dim × 2.5, min 270)
  region.outside_image: false                 // Visible en imagen

  // ═══════════════════════════════════════════
  // DETECCIÓN (calculada en detector + asignación)
  // ═══════════════════════════════════════════
  region.detection_roi: [845, 280, 35, 65]   // Dónde se DETECTÓ realmente
  region.detect_class_id: TL_VERTICAL_CLASS (0)  // Tipo detectado
  region.detect_score: 0.92                   // Confianza del detector
  region.is_detected: true                    // Fue detectado exitosamente
  region.is_selected: true                    // Asignado 1-to-1

  // ═══════════════════════════════════════════
  // RECONOCIMIENTO + TRACKING (etapas finales)
  // ═══════════════════════════════════════════
  status.color: TL_GREEN          // Color final (clasificado + revisado)
  status.confidence: 0.95         // Confianza del clasificador
  status.blink: false             // No intermitente
}

TrafficLight #2 (signal_12346) {
  id: "signal_12346"
  semantic: 0                     // ⚠️ Tracking individual (NO hay voting)
  region.projection_roi: [920, 310, 35, 75]
  region.detection_roi: [918, 312, 33, 72]
  region.detect_class_id: TL_VERTICAL_CLASS (0)
  region.is_detected: true
  status.color: TL_GREEN          // Clasificado individualmente
  status.confidence: 0.88
  status.blink: false
}

TrafficLight #3 (signal_12347) {
  id: "signal_12347"
  semantic: 0                     // ⚠️ Tracking individual
  region.projection_roi: [780, 295, 38, 77]
  region.detection_roi: [0, 0, 0, 0]  // NO detectado
  region.is_detected: false
  status.color: TL_UNKNOWN_COLOR  // ⚠️ NO corregido (sin voting)
  status.confidence: 0.0          // Sin detección
  status.blink: false
}

TrafficLight #4 (signal_12348) {
  id: "signal_12348"
  semantic: 0                     // ⚠️ Todos tienen semantic=0
  region.projection_roi: [650, 450, 30, 50]
  region.detection_roi: [652, 448, 28, 52]
  region.detect_class_id: TL_VERTICAL_CLASS (0)
  region.is_detected: true
  status.color: TL_RED
  status.confidence: 0.91
  status.blink: false
}
```

Esta información se publica al resto del sistema Apollo (módulos de planning, control, etc.) para toma de decisiones.

**Mensaje publicado** (formato protobuf - ejemplo conceptual):
```protobuf
TrafficLightDetectionResult {
  header {
    timestamp_sec: 1234567890.456
    camera_name: "front_telephoto"
  }

  traffic_lights: [
    TrafficLight {
      id: "signal_12345"
      // ⚠️ Nota: semantic_id NO existe en el proto real
      bounding_box: { x: 845, y: 280, width: 35, height: 65 }
      color: GREEN
      confidence: 0.95
      blink: false
    },
    ...
  ]
}
```

---

## 🔑 Puntos Clave del Flujo

### 1. HD-Map como fuente de identidad persistente

**El HD-Map provee**:
- `id`: Identificador único de cada semáforo físico
- ⚠️ `semantic_id`: **DISEÑADO pero NO implementado en Apollo** (siempre 0)
- Coordenadas 3D exactas
- Información geométrica (contorno, línea de stop, etc.)

**El ID es persistente**:
- NO cambia entre frames
- Permite tracking robusto individual
- ⚠️ El voting por grupo NO funciona (semantic_id = 0)

**En nuestro sistema** (sin HD-Map):
- Usamos row index del archivo de projections
- Los "IDs" pueden cambiar si se reordena el archivo
- NO tenemos semantic_ids → sin voting por grupo

### 2. ⚠️ Semantic IDs - FEATURE DISEÑADA PERO NO IMPLEMENTADA

**🚨 IMPORTANTE**: Esta sección describe cómo **DEBERÍA** funcionar si semantic_id estuviera implementado, pero **NO FUNCIONA** en Apollo actual.

**Concepto diseñado** (NO funcional): Varios semáforos físicos compartirían el mismo `semantic_id`

**Ejemplo teórico**:
```
Cruce Main St. y 5th Ave (SI estuviera implementado):
  - Semáforo vehicular Norte:  semantic_id = 100
  - Semáforo vehicular Sur:    semantic_id = 100
  - Semáforo vehicular Este:   semantic_id = 100
  - Semáforo peatonal:         semantic_id = 101 (diferente)
```

**Ventajas previstas** (NO disponibles):
- **Voting**: Si 2 detectan GREEN y 1 detecta BLACK → todos quedarían GREEN
- **Robustez**: Compensaría errores en detecciones individuales
- **Coherencia**: Los semáforos del mismo cruce cambiarían coordinadamente

**Realidad en Apollo**:
- ❌ semantic_id SIEMPRE es 0 (hardcoded)
- ❌ El HD-Map Signal proto NO tiene campo semantic_id
- ❌ Cada semáforo tiene tracking individual
- ❌ NO hay voting por grupo
- ✅ Solo funciona el tracking temporal individual

### 3. Multi-detections en la etapa de Detección

**SÍ existe**: La CNN puede generar múltiples detecciones por cada projection box.

**Código**: `detection.cc:363` - `lights->push_back(tmp)`

**Ejemplo**:
- Projection ROI #1 contiene 2 semáforos muy juntos
- CNN detecta ambos → 2 detections agregadas al buffer
- Después de procesar 8 projection ROIs → pueden haber 15 detections

**NO significa "multi-ROI"** en el sentido de asignar múltiples detections a un HD-Map light.

### 4. NMS Global, NO por ROI

El NMS se aplica sobre **todas** las detections juntas (global), no separado por ROI.

**Razón**: Puede haber detections duplicadas de diferentes ROIs que se solapan.

**Ejemplo**:
- Projection ROI #1 genera detection A
- Projection ROI #2 genera detection B
- Si A y B tienen IoU > 0.6 → NMS elimina una (la de menor score)

### 5. Asignación 1-to-1 estricta

**NO existe** asignar múltiples detections a un mismo HD-Map light.

**Mecanismo**:
- Hungarian algorithm produce asignación óptima
- Flags `is_selected` previenen reasignación (`select.cc:99-100`)
- 1 HD-Map light → máximo 1 detection
- 1 detection → máximo 1 HD-Map light

**Ejemplo**:
```
Entrada:
  - 8 HD-Map lights
  - 9 detections (post-NMS)

Salida:
  - 7 HD-Map lights con detection asignada
  - 1 HD-Map light sin detection
  - 1 detection sin asignar (queda descartada)
```

### 6. Peso espacial dominante (70%)

En la asignación húngara: **70% distancia + 30% confianza**

**Razón**: Apollo confía más en la proyección del HD-Map (muy precisa, ±5cm) que en el score de la CNN.

**Ejemplo**:
```
Caso A:
  - Detection muy cerca de projection (5 píxeles)
  - Detector score: 0.70
  - Combined score: 0.7×0.99 + 0.3×0.70 = 0.90

Caso B:
  - Detection lejos de projection (50 píxeles)
  - Detector score: 0.95
  - Combined score: 0.7×0.60 + 0.3×0.90 = 0.69

→ Se elige Caso A (confía más en posición HD-Map)
```

### 7. Tracking con historial individual por semáforo

**Estructura REAL del historial** (usando ID del HD-Map):
```cpp
history_semantic_["No_semantic_light_signal_12345"] = {
  semantic: "No_semantic_light_signal_12345",  // Clave: ID del HD-Map
  color: GREEN,
  timestamp: último_update,
  light_ids: [0],  // ⚠️ Solo 1 índice (semáforo individual)
  blink: false,
  last_bright_timestamp: ...,
  last_dark_timestamp: ...,
  hysteretic_window: { ... }
}
```

**⚠️ Un semáforo = un historial** (NO hay agrupamiento)

**Reglas de transición**:
- YELLOW después de RED → mantener RED (sospechoso)
- BLACK → hysteresis de 2 frames (count > 1)
- Cambios normales → aceptar con update de timestamp

**Blink detection**:
- Solo para GREEN
- Detecta patrón: BRIGHT → DARK(>0.55s) → BRIGHT
- Útil para flechas verdes intermitentes

---

## 📊 Resumen de Cardinalidades

| Etapa | Input | Output | Cambio | Archivos |
|-------|-------|--------|--------|----------|
| **Preprocesamiento** | M signals del HD-Map | M TrafficLight con projection_roi | 1:1 | `traffic_light_region_proposal_component.cc`<br>`tl_preprocessor.cc` |
| **Detección (Inference)** | M TrafficLight con projection_roi | N detections en buffer | 1:N (N≥M, N=M, o N<M) | `detection.cc:142-216` |
| **Detección (NMS)** | N detections | N' detections | N:N' (N'≤N) | `detection.cc:373-422` |
| **Asignación** | M TrafficLight + N' detections | M TrafficLight (algunos con detection) | M+N':M (1-to-1) | `select.cc:42-129` |
| **Reconocimiento** | M TrafficLight | M TrafficLight con color | 1:1 | `recognition.cc`<br>`classify.cc` |
| **Tracking** | M TrafficLight con color | M TrafficLight revisados | 1:1 (tracking individual, NO voting) | `semantic_decision.cc` |

**Ejemplo numérico completo**:
```
Frame N (timestamp: 1234567890.456):

1. Preprocesamiento:
   - Query HD-Map → 8 signals
   - Generar 8 TrafficLight objects
   - Proyectar 3D→2D → 8 projection_roi

2. Detección Inference:
   - Procesar 8 projection_roi (loop serial)
   - ROI #1 → 2 detections
   - ROI #2 ��� 1 detection
   - ROI #3 → 0 detections
   - ROI #4 → 3 detections
   - ROI #5 → 2 detections
   - ROI #6 → 2 detections
   - ROI #7 → 3 detections
   - ROI #8 → 2 detections
   - Total: 15 detections en buffer

3. Detección NMS:
   - Entrada: 15 detections
   - NMS global (IoU>0.6)
   - Salida: 9 detections (eliminó 6 duplicadas)

4. Asignación:
   - Entrada: 8 HD-Map lights + 9 detections
   - Hungarian 8×9
   - Salida:
     * 7 HD-Map lights con detection asignada
     * 1 HD-Map light sin detection
     * 1 detection sin asignar

5. Reconocimiento:
   - Entrada: 8 HD-Map lights
   - 7 detectados → clasificar color
   - 1 no detectado → UNKNOWN

6. Tracking:
   - ⚠️ Agrupamiento individual (semantic_id = 0 para todos):
     * light [0] (signal_12345): GREEN → revisar con historial → GREEN
     * light [1] (signal_12346): GREEN → revisar con historial → GREEN
     * light [2] (signal_12347): UNKNOWN → revisar con historial → UNKNOWN
     * light [3] (signal_12348): RED → revisar con historial → RED
     * ... (cada uno independiente)
   - Aplicar revisión temporal individual
   - Actualizar historial por semáforo
```

---

## 🗂️ Archivos Fuente del Código Original Verificados

### Preprocesamiento (Region Proposal)
- **`traffic_light_region_proposal_component.cc`** (555 líneas)
  - Query HD-Map: líneas 343-377
  - Generación de TrafficLight: líneas 319-341
  - Selección de cámara: líneas 408-448

- **`tl_preprocessor.cc`** (358 líneas)
  - Proyección 3D→2D: líneas 236-272
  - Selección de cámara multi-focal: líneas 180-234

- **`multi_camera_projection.cc`** (194 líneas)
  - Transformaciones geométricas 3D→2D

### Detección
- **`detection.cc`** (429 líneas)
  - Loop serial inference: líneas 142-216
  - ROI expansion (crop_scale=2.5): línea 175
  - CNN inference: líneas 202-206
  - SelectOutputBoxes (push_back): líneas 278-371
  - NMS global: líneas 373-422

### Asignación
- **`select.cc`** (134 líneas)
  - Construcción matriz costos: líneas 42-86
  - Cálculo Gaussian 2D: líneas 34-40
  - Hungarian algorithm: línea 88
  - Post-procesamiento 1-to-1: líneas 90-129

### Reconocimiento
- **`recognition.cc`** (83 líneas)
  - Switch por detect_class_id: líneas 48-76
  - Llamadas a modelos especializados

- **`classify.cc`**
  - Clasificación por modelo (vert/hori/quad)

### Tracking
- **`semantic_decision.cc`** (296 líneas)
  - Agrupamiento por semantic_id: líneas 239-280
  - Voting: líneas 96-138
  - Revisión temporal: líneas 151-237
  - Blink detection: líneas 187-190
  - Hysteresis: líneas 72-93

---

**FIN DEL DOCUMENTO NARRATIVO DETALLADO**
