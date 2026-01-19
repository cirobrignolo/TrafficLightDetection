# Evolución del Sistema de Detección de Semáforos

Este documento narra la evolución del proyecto desde su concepción hasta el estado actual, incluyendo las decisiones tomadas, los problemas encontrados, los errores corregidos y los aprendizajes obtenidos.

---

## 1. Origen del Proyecto

### 1.1 Motivación Inicial

El proyecto nació a partir del paper **"A First Look at the Integration of Machine Learning Models in Complex Autonomous Driving Systems"** (Peng et al., FSE 2020). Este trabajo analiza cómo se integran los modelos de machine learning en sistemas complejos de conducción autónoma, identificando desafíos y oportunidades de mejora.

**Referencia:** https://petertsehsun.github.io/papers/fse2020_ADS.pdf

A partir de las discusiones con los profesores, surgió la idea de crear un **subsistema especializado** en una sección particular del sistema completo de conducción autónoma. El objetivo era poder realizar **tests más específicos y controlados** sobre componentes individuales, algo difícil de lograr en el sistema monolítico original.

### 1.2 Elección del Subsistema

El sistema de referencia elegido fue **Apollo** de Baidu, uno de los sistemas de conducción autónoma más completos disponibles como open source.

**Repositorio:** https://github.com/ApolloAuto/apollo

Para determinar qué subsistema atacar, se evaluaron dos candidatos:

| Opción | Ventajas | Desventajas |
|--------|----------|-------------|
| **Detección de Semáforos** | Solo usa cámaras, más simple de recrear | Menos componentes para testear |
| **Detección de Obstáculos** | Más complejo, más casos de test | Requiere LiDAR, difícil de simular |

**Decisión:** Se eligió el subsistema de **detección de semáforos** por una razón fundamental: no utiliza LiDAR. Los sensores LiDAR son significativamente más complejos de simular y recrear para generar casos de test controlados. Las cámaras, en cambio, permiten usar videos e imágenes estáticas de manera mucho más sencilla.

---

## 2. Primera Aproximación: Recortar el Sistema Original

### 2.1 Estrategia Inicial

La primera estrategia fue tomar el código original de Apollo y **recortar progresivamente** las secciones no necesarias, manteniendo solo el módulo de detección de semáforos.

### 2.2 Problemas Encontrados

Esta aproximación resultó en **múltiples problemas**:

1. **Código extenso:** El sistema completo de Apollo tiene millones de líneas de código. Identificar qué era necesario y qué no requería un entendimiento profundo de todo el sistema.

2. **Dependencias complejas:** El módulo de percepción dependía de:
   - Librerías externas específicas (Caffe, CUDA, cuDNN)
   - Frameworks propios de Apollo (CyberRT)
   - Configuraciones específicas del hardware
   - Herramientas de debug y visualización integradas

3. **Framework CyberRT:** Todo el sistema estaba fuertemente entrelazado con CyberRT, un framework de comunicación y orquestación desarrollado por Baidu. Modularizar el subsistema de semáforos requería entender este framework completo.

4. **Herramientas de debug:** Apollo incluye interfaces gráficas y herramientas de visualización propias que complicaban aún más la extracción del módulo.

5. **Requisitos de hardware:** Se descubrió que el módulo de percepción requería **obligatoriamente** una GPU NVIDIA con CUDA configurado correctamente.

### 2.3 Complejidad de Configuración de Apollo

Para dimensionar la complejidad del primer approach, Apollo requiere una configuración específica y extensa:

#### Requisitos de Hardware (según documentación oficial)

| Componente | Requisito Mínimo |
|------------|------------------|
| **CPU** | Procesador de 8 núcleos |
| **RAM** | 16 GB |
| **GPU** | NVIDIA con arquitectura Turing o superior (RTX 20xx+) |
| **Almacenamiento** | SSD con espacio suficiente (~50 GB solo para el código) |

#### Requisitos de Software

| Software | Versión Requerida |
|----------|-------------------|
| **Sistema Operativo** | Ubuntu 18.04, 20.04, o 22.04 |
| **NVIDIA Driver** | >= 520.61.05 |
| **CUDA** | 11.8 |
| **Docker** | >= 19.03 |
| **NVIDIA Container Toolkit** | Requerido |

#### Proceso de Instalación (según guía oficial)

1. **Instalación de prerequisites:**
   - Instalar Ubuntu con kernel específico
   - Instalar NVIDIA driver correcto
   - Instalar Docker y configurar permisos
   - Instalar NVIDIA Container Toolkit
   - Verificar compatibilidad GPU

2. **Clonar y configurar:**
   ```bash
   git clone https://github.com/ApolloAuto/apollo.git  # ~2GB
   cd apollo
   bash docker/scripts/dev_start.sh  # Descarga imagen Docker ~15GB
   bash docker/scripts/dev_into.sh   # Entrar al container
   ```

3. **Compilación:**
   ```bash
   ./apollo.sh clean
   ./apollo.sh build_opt_gpu  # 2-4 horas en hardware típico
   ```

4. **Ejecución:**
   ```bash
   ./scripts/bootstrap.sh  # Iniciar Dreamview
   # Acceder a http://localhost:8888
   # Configurar modo de conducción y mapa
   # Cargar archivos de calibración específicos del vehículo
   ```

#### Sistema de Build Bazel

Apollo utiliza **Bazel** como sistema de build, lo cual agrega otra capa de complejidad:

- Archivos BUILD en cada directorio
- Configuraciones específicas para CPU/GPU
- Múltiples variantes: `build_dbg`, `build_opt`, `build_cpu`, `build_gpu`, `build_opt_gpu`
- Dependencias transitivas complejas

#### Conclusión sobre el Primer Approach

El esfuerzo necesario para configurar Apollo correctamente (driver, CUDA, Docker, Bazel, CyberRT) era desproporcionado al objetivo de testear el módulo de semáforos. Cada paso de configuración tenía potencial de fallar con errores crípticos difíciles de debuggear.

### 2.4 Tiempo Invertido

Durante este período, mientras se luchaba con la infraestructura, se trabajó en paralelo en **entender el pipeline** de detección de semáforos a nivel conceptual:

- Se identificaron las 5 etapas: Preprocesamiento, Detección, Asignación, Reconocimiento y Tracking
- Se entendió qué hacía cada etapa "por arriba"
- Sin embargo, la imposibilidad de ejecutar el código impedía un entendimiento profundo a nivel de implementación

### 2.5 Decisión de Cambio de Estrategia

Después de invertir considerable tiempo en esta aproximación, se tomó la decisión de **abandonar la estrategia de recorte** y comenzar desde cero.

**Razones:**
- El tiempo invertido en entender el framework no aportaba al objetivo final
- La complejidad del sistema original no justificaba el esfuerzo
- Era más eficiente recrear solo lo necesario que extraer de un sistema gigante

---

## 3. Segunda Aproximación: Recreación desde Cero

### 3.1 Nueva Estrategia

Se decidió **escribir el código desde cero** en Python/PyTorch, tomando como referencia únicamente:
- El pipeline de detección de semáforos
- Los archivos de configuración y pesos de los modelos
- La lógica documentada en el código original

### 3.2 Elección de Python/PyTorch

**¿Por qué Python?**
- Era el lenguaje que mejor se manejaba en el momento
- Permitía acelerar el desarrollo
- PyTorch facilitaba la conversión de los modelos de Caffe
- Mejor para prototipado y experimentación

### 3.3 Proceso de Construcción

Se recreó cada etapa del pipeline:

1. **Preprocesamiento:** Funciones de crop, resize y normalización
2. **Detección:** Conversión del modelo Faster R-CNN de Caffe a PyTorch
3. **Asignación:** Implementación del algoritmo Húngaro
4. **Reconocimiento:** Conversión de los 3 clasificadores (vertical, horizontal, quad)
5. **Tracking:** Implementación de las reglas temporales

**Principio guía:** Solo implementar funcionalidades necesarias, evitando interfaces gráficas, herramientas de debug integradas y cualquier elemento que no aportara al objetivo de testing.

---

## 4. Primera Versión Funcional y Tests Iniciales

### 4.1 Primer Test: Video de Semáforo Cambiando

Una vez que hubo una primera versión funcional, se realizó el primer test básico:

- **Input:** Video de un semáforo cambiando de color
- **Proceso:** Aplicar el pipeline completo al video
- **Output:** Resultados del sistema frame a frame

Este test sirvió para verificar que el sistema funcionaba end-to-end.

### 4.2 Mejoras de Observabilidad

A partir de los primeros tests, se realizaron modificaciones para mejorar la **observabilidad** del sistema:

- **Resultados por etapa:** Modificar el pipeline para obtener outputs intermedios de cada etapa
- **Información visual:** Dibujar bounding boxes, colores detectados y scores sobre las imágenes
- **Información textual:** Logs con datos de ROIs, resultados de modelos, asignaciones, etc.

Esto permitió debuggear el sistema paso a paso y entender mejor su comportamiento.

---

## 5. Errores Descubiertos y Corregidos

### 5.1 Errores de Comprensión Funcional

Al recrear el sistema desde cero, surgieron **errores de interpretación** del sistema original:

#### Error 1: Interpretación del Blink (Parpadeo)

- **Error:** Se interpretó que la funcionalidad de "blink" (parpadeo) era sobre el **amarillo**
- **Realidad:** El blink se detecta unicamente en el color **verde**, no en el amarillo, informando su proximo cambio
- **Impacto:** La lógica de detección de parpadeo estaba incorrecta
- **Corrección:** Se reescribió la lógica de blink detection siguiendo el código original

#### Error 2: Regla de Secuencia de Colores

- **Error:** Se asumió que la secuencia era RED → YELLOW → GREEN
- **Realidad:** La secuencia correcta es GREEN → YELLOW → RED (el amarillo solo puede venir después del verde)
- **Impacto:** La regla de seguridad "YELLOW after RED → mantener RED" no tenía sentido con la interpretación errónea
- **Corrección:** Se invirtió la lógica de la regla de secuencia

#### Error 3: Inputs del Algoritmo Húngaro

- **Error:** No se pasaban los inputs correctos al algoritmo de asignación
- **Impacto:** Las asignaciones projection→detection eran incorrectas
- **Corrección:** Se revisó la construcción de la matriz de costos y los parámetros

### 5.2 Errores de Implementación

#### Error 4: Orden de Means BGR

- **Error:** Los means del recognizer estaban en orden RGB en lugar de BGR
- **Realidad:** OpenCV (cv2.imread) devuelve imágenes en formato BGR
- **Impacto:** La normalización era incorrecta, afectando el reconocimiento de colores
- **Corrección:** Se invirtió el orden de los means a [66.56, 66.58, 69.06]

### 5.3 Metodología de Corrección

Para cada error descubierto:
1. Se identificaba el síntoma (resultado incorrecto)
2. Se investigaba el código original de Apollo
3. Se corregía la implementación
4. Se creaba un **test específico** para ese caso

---

## 6. Test de Cruce de Historiales

### 6.1 Hipótesis Inicial

Con el entendimiento (aún incorrecto) del sistema, se planteó un test:

**Hipótesis:** Si se cruzan los historiales de dos semáforos (uno en verde, otro en amarillo), el sistema debería fallar por detectar una transición inválida según la regla de secuencia de colores.

### 6.2 Primer Intento: Modificación Manual

Se realizó un test donde se **modificaban manualmente los IDs** de los semáforos para cruzar sus historiales.

**Resultado:** El test apoyó la hipótesis, pero la metodología no era coherente con lo que sucedería en un escenario real. Modificar IDs a mano no representa un caso de uso válido.

### 6.3 Segundo Intento: Movimiento de Semáforos

Se buscó un caso donde el cruce de historiales pudiera ocurrir **naturalmente**.

**Idea:** Si entre un frame y el siguiente se mueven las posiciones de los semáforos en la imagen, el sistema podría confundir los IDs basándose en las coordenadas espaciales (dado que el peso de la distancia es 70% vs 30% de confianza).

**Test diseñado:**
- Frame N: Semáforo A en posición X (verde), Semáforo B en posición Y (amarillo)
- Frame N+1: Ambos semáforos se mueven varios píxeles a la derecha
- Semáforo A ahora está en la posición donde estaba B

**Resultado:** El cruce ocurría **a veces sí, a veces no**. Esto llevó a una investigación más profunda.

### 6.4 Descubrimientos Importantes

La investigación reveló varios **errores de concepto**:

1. **Dos tipos de ID:** El sistema maneja dos IDs distintos:
   - **Signal ID:** ID físico del semáforo en el HD-Map (persistente)
   - **Detection ID:** ID de la detección en el frame actual (temporal)
   - El que prevalece para el tracking es siempre el **Signal ID**

2. **Coordenadas del HD-Map:** Apollo guarda las coordenadas 3D reales del semáforo físico. No importa cuántos píxeles se mueva en la imagen, el sistema sabe dónde debería estar basándose en el HD-Map y la pose del vehículo.

3. **El test no tenía sentido:** Mover semáforos píxeles en la imagen no puede confundir al sistema real de Apollo, porque la identidad viene del HD-Map, no de la posición en el frame.

---

## 7. Creación del Diagrama de Flujo Completo

### 7.1 Necesidad de Documentación Detallada

Los descubrimientos anteriores evidenciaron que faltaba un **entendimiento profundo y documentado** del sistema original.

### 7.2 Proceso de Documentación

Se creó un diagrama de flujo exhaustivo del pipeline de Apollo:

- Análisis línea por línea del código original
- Documentación de cada etapa con referencias a archivos y líneas específicas
- Identificación de todos los parámetros y sus valores
- Mapeo del flujo de datos entre etapas

### 7.3 Errores Descubiertos Durante la Documentación

- Corrección de la lógica de asignación (N detecciones → M semáforos)
- Corrección del modelo de tracking completo
- Entendimiento correcto de las reglas de secuencia
- Descubrimiento del Semantic ID (y su no-uso)

---

## 8. Descubrimiento del Semantic ID

### 8.1 ¿Qué es el Semantic ID?

El código de Apollo contiene un campo `semantic_id` que **conceptualmente** debería agrupar semáforos del mismo cruce para aplicar voting (si 2 de 3 semáforos del cruce son verdes, el tercero también debería serlo).

### 8.2 Investigación

Al investigar el código original:

```cpp
// traffic_light_region_proposal_component.cc:335
int cur_semantic = 0;  // ← SIEMPRE 0, hardcodeado
light->semantic = cur_semantic;
```

**Descubrimiento:** El `semantic_id` está **diseñado pero no implementado**. Siempre se asigna 0 a todos los semáforos, por lo que el voting nunca se ejecuta realmente.

### 8.3 Implicaciones para Testing

Este descubrimiento abrió nuevas oportunidades de test:

- **No hay priorización por distancia entre semáforos del mismo cruce**
- Todos los semáforos se tratan de manera independiente
- Un semáforo lejano de otra intersección puede influir en el sistema

---

## 9. Tests Basados en Nuevos Descubrimientos

### 9.1 Test de Confusión por Atardecer

**Contexto:** El amarillo/naranja del atardecer tiene colores similares al amarillo del semáforo.

**Hipótesis:** Un atardecer podría confundir al recognizer y hacer que clasifique incorrectamente.

**Resultado:** El test fue exitoso - el sistema efectivamente se confundía en ciertas condiciones de iluminación.

### 9.2 Test de Semáforo Lejano

**Contexto:** Sin semantic grouping, todos los semáforos se procesan independientemente sin importar su distancia o relevancia.

**Hipótesis:** Un semáforo en rojo de una intersección lejana (no relevante para el vehículo) podría influir en las decisiones del sistema.

**Estado:** Test diseñado basado en el descubrimiento del semantic_id no implementado.
