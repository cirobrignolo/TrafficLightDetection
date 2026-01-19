# Métricas Comparativas: Apollo Original vs Nuestro Sistema

Este documento presenta una comparación cuantitativa entre el sistema original de Apollo y nuestra implementación, evidenciando la complejidad del primer approach (recortar Apollo) versus el segundo (recrear desde cero).

---

## 1. Resumen Ejecutivo

| Métrica | Apollo (para Traffic Light) | Nuestro Sistema | Reducción |
|---------|----------------------------|-----------------|-----------|
| **Líneas de código necesarias** | ~174,208 | 1,879 | **99%** |
| **Archivos de código** | ~1,300+ | 13 | **99%** |
| **Dependencias externas** | 54 librerías | 2 (PyTorch, NumPy) | **96%** |
| **Lenguajes** | C++, Python, Proto, Bash | Python | - |
| **Framework requerido** | CyberRT (64,949 líneas) | Ninguno | **100%** |

---

## 2. Estructura del Sistema Apollo

### 2.1 Módulos de Traffic Light en Apollo

Apollo divide la funcionalidad de semáforos en **4 módulos separados**:

| Módulo | Archivos | Líneas C++ | Función |
|--------|----------|------------|---------|
| `traffic_light_region_proposal` | 7 | 1,720 | Query HD-Map, proyección 3D→2D |
| `traffic_light_detection` | 9 | 1,230 | CNN detector, NMS, validación |
| `traffic_light_recognition` | 7 | 683 | Clasificación de color |
| `traffic_light_tracking` | 5 | 1,192 | Tracking temporal, reglas de seguridad |
| **SUBTOTAL Traffic Light** | **28** | **4,825** | - |

### 2.2 Dependencias Obligatorias

Para ejecutar los módulos de traffic light, Apollo requiere:

| Dependencia | Líneas C++ | Archivos | Propósito |
|-------------|------------|----------|-----------|
| **CyberRT** | 64,949 | 475 | Framework de comunicación y orquestación |
| **Perception Common** | 63,772 | ~200 | Utilidades de percepción, inference, algoritmos |
| **Modules Common** | 19,922 | ~100 | Utilidades generales, math, configs |
| **Map Module** | 20,740 | ~100 | HD-Map, consulta de signals |
| **SUBTOTAL Dependencias** | **169,383** | **~875** | - |

### 2.3 Total para Traffic Light en Apollo

```
Traffic Light Modules:        4,825 líneas
+ CyberRT:                   64,949 líneas
+ Perception Common:         63,772 líneas
+ Modules Common:            19,922 líneas
+ Map Module:                20,740 líneas
─────────────────────────────────────────
TOTAL NECESARIO:            174,208 líneas C++
```

**Nota:** Esto no incluye archivos proto (~111 líneas), archivos de configuración, BUILD files (~384 líneas), ni las 54 librerías de third_party.

---

## 3. Estructura de Nuestro Sistema

### 3.1 Archivos del Sistema

| Archivo | Líneas | Función |
|---------|--------|---------|
| `pipeline.py` | 250 | Pipeline principal, orquestación |
| `tracking.py` | 215 | Tracking temporal, reglas de seguridad |
| `hungarian_optimizer.py` | 260 | Algoritmo de asignación |
| `tools/utils.py` | 475 | Preprocesamiento, NMS, utilidades |
| `rpn_proposal.py` | 152 | Region Proposal Network |
| `feature_net.py` | 139 | Backbone de features |
| `faster_rcnn.py` | 118 | Detector Faster R-CNN |
| `dfmb_roi_align.py` | 94 | ROI Align layer |
| `selector.py` | 67 | Asignación Hungarian |
| `recognizer.py` | 63 | Clasificador de color |
| `detector.py` | 46 | Wrapper del detector |
| `__init__.py` (x2) | 0 | Módulos Python |
| **TOTAL** | **1,879** | - |

### 3.2 Dependencias

| Dependencia | Propósito |
|-------------|-----------|
| PyTorch | Framework de deep learning, inference |
| NumPy | Operaciones numéricas |

---

## 4. Comparación Detallada por Componente

### 4.1 Framework de Comunicación

| Aspecto | Apollo (CyberRT) | Nuestro Sistema |
|---------|------------------|-----------------|
| Líneas de código | 64,949 | 0 |
| Archivos | 475 | 0 |
| Componentes | Node, Channel, Reader, Writer, Scheduler, Service Discovery, Transport, etc. | N/A |
| Configuración | DAG files, launch files, conf files | N/A |
| Complejidad | Sistema distribuido con múltiples procesos | Ejecución secuencial simple |

**Justificación de la diferencia:** CyberRT es necesario en Apollo para coordinar múltiples módulos ejecutándose en paralelo en un vehículo real. Nuestro sistema procesa frames secuencialmente, no requiere comunicación entre procesos.

### 4.2 Inference Framework

| Aspecto | Apollo | Nuestro Sistema |
|---------|--------|-----------------|
| Framework | Custom (Caffe-based) | PyTorch |
| Líneas propias | 11,642 | 0 (usa PyTorch) |
| Backends soportados | Caffe, TensorRT, ONNX, PaddlePaddle, MIGraphX | PyTorch |
| Configuración | Prototxt, proto files | Ninguna (carga .torch) |

### 4.3 HD-Map Integration

| Aspecto | Apollo | Nuestro Sistema |
|---------|--------|-----------------|
| Módulo | `modules/map` | N/A |
| Líneas | 20,740 | 0 |
| Función | Query geoespacial de signals | Archivo de texto con boxes |
| Proyección 3D→2D | Sí (con calibración de cámara) | No (boxes pre-calculadas) |

**Justificación:** Apollo necesita consultar el HD-Map en tiempo real según la posición del vehículo. Nuestro sistema usa boxes estáticas definidas por video.

---

## 5. Complejidad de Integración

### 5.1 Dependencias Externas de Apollo

Apollo requiere **54 librerías externas** en `third_party/`:

```
absl, adolc, ad_rss_lib, adv_plat, atlas, bazel, benchmark, boost,
caddn_infer_op, camera_library, can_card_library, centerpoint_infer_op,
civetweb, cpplint, eigen3, fastdds, ffmpeg, fftw3, gflags, glew, glog,
gpus, gtest, intrinsics_translation, ipopt, libtorch, localization_msf,
ncurses5, nlohmann_json, npp, nvjpeg, opencv, opengl, openh264, openssl,
osqp, paddleinference, pcl, portaudio, proj, protobuf, py, qt5, rtklib,
sqlite3, sse2neon, tensorrt, tf2, tinyxml2, uuid, vtk, yaml_cpp
```

**Dependencias críticas para traffic light:**
- `libtorch` - Inference
- `opencv` - Procesamiento de imágenes
- `eigen3` - Álgebra lineal
- `protobuf` - Serialización de mensajes
- `gflags/glog` - Configuración y logging
- `cuda/cudnn/tensorrt` - GPU acceleration

### 5.2 Dependencias de Nuestro Sistema

```
pytorch  - Inference + operaciones de tensor
numpy    - Operaciones numéricas auxiliares
```

---

## 6. Archivos de Configuración

### 6.1 Apollo

| Tipo | Cantidad | Propósito |
|------|----------|-----------|
| `.proto` | 7 (111 líneas) | Definición de estructuras de datos |
| `.pb.txt` | 7 | Configuración de modelos y parámetros |
| `BUILD` | 11 (384 líneas) | Sistema de build Bazel |
| `cyberfile.xml` | 4 | Dependencias de módulos |
| `.dag` | 4 | Grafos de ejecución |
| `.launch` | 4 | Archivos de lanzamiento |

### 6.2 Nuestro Sistema

| Tipo | Cantidad | Propósito |
|------|----------|-----------|
| `.json` | 5 | Configuración de capas del detector |
| `.torch` | 4 | Pesos de modelos |

---

## 7. Requisitos de Configuración

### 7.1 Requisitos de Hardware

| Componente | Apollo | Nuestro Sistema |
|------------|--------|-----------------|
| **CPU** | 8 núcleos (mínimo) | Cualquier CPU moderna |
| **RAM** | 16 GB (mínimo) | 4 GB suficiente |
| **GPU** | NVIDIA Turing+ obligatoria | Opcional (soporta CPU) |
| **Almacenamiento** | ~50 GB (código + Docker) | ~500 MB |
| **Sistema Operativo** | Ubuntu 18.04/20.04/22.04 | Cualquiera con Python |

### 7.2 Requisitos de Software de Apollo

Según la documentación oficial de Apollo, se requiere:

| Software | Versión Requerida | Propósito |
|----------|-------------------|-----------|
| **Ubuntu** | 18.04, 20.04, o 22.04 | Sistema operativo base |
| **NVIDIA Driver** | >= 520.61.05 | Soporte GPU |
| **CUDA** | 11.8 | Aceleración GPU |
| **Docker** | >= 19.03 | Contenedorización |
| **NVIDIA Container Toolkit** | Última versión | GPU en Docker |

**Nota:** El módulo de percepción (que incluye Traffic Light) **requiere obligatoriamente GPU**. La documentación indica que sin GPU disponible, el sistema se ejecutará sin percepción ya que está basado en CUDA.

### 7.3 Proceso de Instalación de Apollo

Según la guía oficial, el proceso incluye:

```bash
# 1. Prerequisitos (1-2 horas)
# - Instalar NVIDIA driver correcto
# - Instalar Docker
# - Instalar NVIDIA Container Toolkit
# - Verificar compatibilidad GPU

# 2. Clonar repositorio (~2GB)
git clone https://github.com/ApolloAuto/apollo.git

# 3. Iniciar container Docker (~15GB imagen)
bash docker/scripts/dev_start.sh
bash docker/scripts/dev_into.sh

# 4. Compilar (2-4 horas)
./apollo.sh clean
./apollo.sh build_opt_gpu

# 5. Ejecutar
./scripts/bootstrap.sh
# Acceder a http://localhost:8888
```

### 7.4 Sistema de Build Bazel

Apollo utiliza **Bazel** como sistema de build:

- Múltiples variantes de compilación: `build_dbg`, `build_opt`, `build_cpu`, `build_gpu`, `build_opt_gpu`
- Configuración CPU vs GPU automática según capacidad del container
- Archivos BUILD en cada directorio con dependencias explícitas
- Sistema de tests integrado (`apollo.sh test`)

### 7.5 Proceso de Instalación de Nuestro Sistema

```bash
# Todo el proceso
pip install torch numpy
python example.py  # Listo
```

---

## 8. Tiempo de Setup Estimado

### 8.1 Apollo (Primer Approach)

| Tarea | Tiempo Estimado |
|-------|-----------------|
| Clonar repositorio (~2GB) | 10-30 min |
| Instalar Docker/dependencias | 1-2 horas |
| Compilar Apollo completo | 2-4 horas |
| Configurar CUDA/cuDNN/TensorRT | 1-2 horas |
| Entender CyberRT | 1-2 semanas |
| Entender estructura de módulos | 1-2 semanas |
| Extraer módulo de traffic light | 2-4 semanas |
| Debuggear dependencias rotas | Indefinido |
| **TOTAL** | **1-2 meses** (mínimo) |

### 8.2 Nuestro Sistema (Segundo Approach)

| Tarea | Tiempo Estimado |
|-------|-----------------|
| Clonar repositorio | 1 min |
| `pip install pytorch numpy` | 5 min |
| Ejecutar ejemplo | Inmediato |
| **TOTAL** | **< 10 minutos** |

---

## 9. Métricas de Mantenibilidad

### 9.1 Complejidad Ciclomática Aproximada

| Sistema | Archivos a mantener | Lenguajes | Build System |
|---------|---------------------|-----------|--------------|
| Apollo TL | ~1,300 | C++, Python, Proto | Bazel |
| Nuestro | 13 | Python | Ninguno (pip) |

### 9.2 Curva de Aprendizaje

| Concepto | Apollo | Nuestro Sistema |
|----------|--------|-----------------|
| Entender el flujo | 2-4 semanas | 1-2 días |
| Modificar un parámetro | Buscar en múltiples archivos | 1 archivo Python |
| Agregar un log de debug | Recompilar | Agregar print() |
| Ejecutar un test | Configurar DAG, launch, etc. | `python script.py` |

---

## 10. Visualización de la Diferencia

### Líneas de Código Necesarias

```
Apollo (para Traffic Light):
████████████████████████████████████████████████████████████████████████████████████████████████████ 174,208

Nuestro Sistema:
█ 1,879

Reducción: 99%
```

### Archivos de Código

```
Apollo:
████████████████████████████████████████████████████████████████████████████████████████████████████ ~1,300

Nuestro Sistema:
█ 13

Reducción: 99%
```

### Dependencias Externas

```
Apollo:
██████████████████████████████████████████████████████ 54 librerías

Nuestro Sistema:
██ 2 librerías (PyTorch, NumPy)

Reducción: 96%
```

---

## 11. Conclusiones

### 11.1 Por qué el Primer Approach Falló

1. **Acoplamiento extremo:** Los módulos de traffic light están profundamente integrados con CyberRT, perception common, y el sistema de build.

2. **Dependencias en cascada:** Extraer traffic light requería traer ~170,000 líneas de código de soporte.

3. **Configuración compleja:** El sistema de build (Bazel), configuración de GPU, y setup de CyberRT representaban semanas de trabajo.

4. **Curva de aprendizaje empinada:** Entender CyberRT solo para poder debuggear era desproporcionado al objetivo.

### 11.2 Por qué el Segundo Approach Funcionó

1. **Código mínimo:** 1,879 líneas vs 174,208 (99% menos).

2. **Sin framework:** Ejecución directa en Python sin orquestación compleja.

3. **Dependencias mínimas:** Solo PyTorch y NumPy, instalables con pip.

4. **Iteración rápida:** Cambios inmediatos sin recompilación.

5. **Foco en lo esencial:** Solo se implementó la lógica del pipeline, no la infraestructura.

### 11.3 Trade-offs

| Aspecto | Apollo | Nuestro Sistema |
|---------|--------|-----------------|
| Producción en vehículo real | ✅ Diseñado para eso | ❌ No aplica |
| Testing y experimentación | ❌ Complejo | ✅ Ideal |
| Tiempo real multi-sensor | ✅ Soportado | ❌ No diseñado |
| Modificabilidad | ❌ Difícil | ✅ Trivial |
| Portabilidad | ❌ Requiere setup específico | ✅ Cualquier máquina con Python |

---

## 12. Datos Crudos

### Líneas por Módulo de Apollo (Traffic Light)

```
traffic_light_region_proposal/
├── traffic_light_region_proposal_component.cc    554
├── traffic_light_region_proposal_component.h     174
├── preprocessor/tl_preprocessor.cc               358
├── preprocessor/tl_preprocessor.h                177
├── preprocessor/multi_camera_projection.cc       193
├── preprocessor/multi_camera_projection.h        119
└── interface/base_tl_preprocessor.h              145
TOTAL: 1,720

traffic_light_detection/
├── traffic_light_detection_component.cc          111
├── traffic_light_detection_component.h            84
├── detector/caffe_detection/detection.cc         428
├── detector/caffe_detection/detection.h          134
├── algorithm/select.cc                           133
├── algorithm/select.h                             71
├── algorithm/cropbox.cc                          106
├── algorithm/cropbox.h                            89
└── interface/base_traffic_light_detector.h        74
TOTAL: 1,230

traffic_light_recognition/
├── traffic_light_recognition_component.cc        110
├── traffic_light_recognition_component.h          84
├── recognition/caffe_recognizer/classify.cc      175
├── recognition/caffe_recognizer/classify.h        82
├── recognition/caffe_recognizer/recognition.cc    82
├── recognition/caffe_recognizer/recognition.h     75
└── interface/base_traffic_light_recognitor.h      75
TOTAL: 683

traffic_light_tracking/
├── traffic_light_tracking_component.cc           581
├── traffic_light_tracking_component.h            115
├── tracker/semantic_decision.cc                  295
├── tracker/semantic_decision.h                   129
└── interface/base_traffic_light_tracker.h         72
TOTAL: 1,192
```

### Líneas por Archivo de Nuestro Sistema

```
src/tlr/
├── tools/utils.py           475
├── hungarian_optimizer.py   260
├── pipeline.py              250
├── tracking.py              215
├── rpn_proposal.py          152
├── feature_net.py           139
├── faster_rcnn.py           118
├── dfmb_roi_align.py         94
├── selector.py               67
├── recognizer.py             63
├── detector.py               46
└── __init__.py (x2)           0
TOTAL: 1,879
```
