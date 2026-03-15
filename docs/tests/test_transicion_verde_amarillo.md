# Test de Transición Verde → Amarillo

## Objetivo

Determinar el punto exacto donde el sistema TLR de Apollo comienza a confundir el color verde con amarillo, mediante la modificación gradual del color del semáforo en el video.

## Metodología

1. **Video base**: Video con efecto de atardecer simulado y semáforo en verde
2. **Variaciones**: 6 versiones del video donde se modifica el color verde del semáforo hacia amarillo usando el efecto "Cambiar color" de Adobe Premiere Pro 2021
3. **Métricas a registrar**:
   - Porcentaje de frames clasificados como verde
   - Porcentaje de frames clasificados como amarillo
   - Confianza promedio de cada clasificación
   - Frame específico donde ocurre el primer error

## Modificación del Color del Semáforo

### Efecto utilizado
- **Software**: Adobe Premiere Pro 2021
- **Efecto**: Efectos → Efectos de video → Corrección de color → Cambiar color

### Configuración base
| Parámetro | Valor |
|-----------|-------|
| Color que cambiar | Verde del semáforo (seleccionado con gotero) |
| Tolerancia coincidente | 15 - 30% |
| Suavizado coincidente | 10 - 15% |
| Hacer coincidir colores | Con RGB |

### Parámetros de transformación por video

| Video | Tono | Luminosidad | Saturación | Color Resultante |
|-------|------|-------------|------------|------------------|
| 1 | 0 | 0 | 0 | Verde original |
| 2 | -20 | +10 | +10 | Verde-amarillento |
| 3 | -65 | 0 | +30 | Amarillo-verdoso |
| 4 | -80 | 0 | +30 | Amarillo |
| 5 | -100 | +10 | +20 | Amarillo intenso |
| 6 | -110 | +10 | +20 | Ámbar |

**Nota**: El parámetro "Tono" controla la rotación del color en el espectro. Valores negativos rotan el verde hacia el amarillo/naranja.

---

## Resultados

### Contexto importante

El video contiene un ciclo completo de semáforo real:
- **Frames 0-132**: Semáforo en VERDE
- **Frames 133-150**: Semáforo en AMARILLO (transición real)
- **Frames 151-355**: Semáforo en ROJO
- **Frames 356-371**: Semáforo vuelve a VERDE

La modificación de color solo afecta al verde del semáforo, por lo que las fases de amarillo y rojo reales permanecen iguales en todos los videos. El análisis se centra en cómo el sistema clasifica la **fase verde modificada**.

### Resumen General

| Video | Color Aplicado | Detecciones Verde (inicio) | Verde Final | Observaciones |
|-------|----------------|---------------------------|-------------|---------------|
| 1 | Verde original (Tono: 0) | 100% verde | 100% verde | Funcionamiento correcto |
| 2 | Verde-amarillento (Tono: -20) | 100% verde | 100% verde | Sin diferencia con original |
| 3 | Amarillo-verdoso (Tono: -65) | 100% verde | 100% verde | Sistema robusto |
| 4 | Amarillo (Tono: -80) | 100% verde | 100% verde | Sistema robusto |
| 5 | Amarillo intenso (Tono: -100) | 100% verde | 100% verde | Sistema robusto |
| 6 | Ámbar (Tono: -110) | GREEN | **YELLOW/RED** | Punto de falla: detecta amarillo y rojo |

### Análisis Detallado por Video

#### Video 1 - Verde original (Tono: 0)
- **Total detecciones**: 714 (2 semáforos detectados en la mayoría de frames)
- **Fase verde inicial (frames 0-132)**: 266 detecciones, 100% clasificadas como GREEN
- **Fase verde final (frames 356-371)**: 16 detecciones, 100% clasificadas como GREEN
- **Observaciones**: Comportamiento de referencia correcto

#### Video 2 - Verde-amarillento (Tono: -20)
- **Total detecciones**: 714
- **Fase verde inicial**: 100% GREEN
- **Fase verde final**: 100% GREEN
- **Observaciones**: Modificación de tono -20 no afecta la clasificación

#### Video 3 - Amarillo-verdoso (Tono: -65)
- **Total detecciones**: 714
- **Fase verde inicial**: 100% GREEN
- **Fase verde final**: 100% GREEN
- **Observaciones**: Incluso con tono -65, el sistema mantiene clasificación correcta

#### Video 4 - Amarillo (Tono: -80)
- **Total detecciones**: 714
- **Fase verde inicial**: 100% GREEN
- **Fase verde final**: 100% GREEN
- **Observaciones**: El sistema sigue siendo robusto a esta modificación

#### Video 5 - Amarillo intenso (Tono: -100)
- **Total detecciones**: 722 (ligera variación por detecciones adicionales)
- **Fase verde inicial**: 100% GREEN
- **Fase verde final**: 100% GREEN
- **Observaciones**: Aún funciona correctamente

#### Video 6 - Ámbar (Tono: -110) ⚠️ PUNTO DE FALLA
- **Total detecciones**: 927
- **Total frames**: 485

**Distribución de detecciones por color:**

| Fase del Video | Frames | Color Detectado | Observaciones |
|----------------|--------|-----------------|---------------|
| Fase 1 | 0-165 | GREEN (100%) | El sistema aún detecta verde al inicio |
| Fase 2 | 166-188 | **YELLOW** (100%) | ⚠️ El verde modificado se detecta como amarillo |
| Fase 3 | 189-443 | **RED** (~95%) | El verde modificado se detecta como rojo |
| Fase 4 | 444-484 | BLACK/RED mezclado | Inestabilidad al final del video |

**Hallazgos importantes:**
- **Frame 166**: Primera detección de YELLOW → **Este es el punto de falla**
- **Frame 189**: Primera detección de RED
- El sistema comienza clasificando correctamente como GREEN pero luego transiciona a YELLOW y RED
- Se detectan correctamente dos semáforos (det_idx 0 y 1) durante la mayor parte del video

**Observaciones:**
- A diferencia de los videos 1-5, aquí el sistema **sí confunde** el verde modificado con otros colores
- La transición ocurre gradualmente: GREEN → YELLOW → RED
- El umbral de falla está entre Tono -100 (funciona) y Tono -110 (falla)

---

## Conclusiones

### Hallazgo Principal

El sistema TLR de Apollo es **sorprendentemente robusto** a cambios de color en el rango verde → amarillo. No se produce confusión verde/amarillo como se esperaba originalmente.

### Punto de Falla

El sistema falla en **Tono -110 (Ámbar)**:
- **Sí confunde verde con amarillo** a partir del frame 166
- Posteriormente clasifica como **RED** a partir del frame 189
- El umbral de falla está entre **Tono -100** (funciona) y **Tono -110** (falla)

### Robustez del Sistema

| Escenario | Comportamiento |
|-----------|----------------|
| Tono 0 a -100 | Clasificación correcta como GREEN |
| Tono -110 | Falla: clasifica como YELLOW → RED |

### Implicaciones

1. **El sistema es robusto** para variaciones de color moderadas (hasta Tono -100)
2. **El punto de falla está bien definido**: entre Tono -100 y -110
3. **El modo de falla es gradual**: GREEN → YELLOW → RED, no un salto abrupto

### Trabajo Futuro

- Realizar pruebas con valores intermedios (Tono -105, -107, -108) para encontrar el punto exacto de transición

## Fecha de Ejecución

- Inicio: 2026-02-26
- Última actualización: 2026-02-27