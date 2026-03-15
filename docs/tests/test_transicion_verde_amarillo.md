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

### Resumen General

| Video | Color Aplicado | % Verde | % Amarillo | % Rojo | Confianza Prom | Observaciones |
|-------|----------------|---------|------------|--------|----------------|---------------|
| 1 | Verde original | | | | | |
| 2 | Verde-amarillento | | | | | |
| 3 | Amarillo-verdoso | | | | | |
| 4 | Amarillo | | | | | |
| 5 | Amarillo intenso | | | | | |
| 6 | Ámbar | | | | | |

### Análisis Detallado por Video

#### Video 1 - Verde original (Tono: 0)
- Total frames analizados:
- Clasificaciones:
- Observaciones:

#### Video 2 - Verde-amarillento (Tono: -20)
- Total frames analizados:
- Clasificaciones:
- Observaciones:

#### Video 3 - Amarillo-verdoso (Tono: -65)
- Total frames analizados:
- Clasificaciones:
- Observaciones:

#### Video 4 - Amarillo (Tono: -80)
- Total frames analizados:
- Clasificaciones:
- Observaciones:

#### Video 5 - Amarillo intenso (Tono: -100)
- Total frames analizados:
- Clasificaciones:
- Observaciones:

#### Video 6 - Ámbar (Tono: -110)
- Total frames analizados:
- Clasificaciones:
- Observaciones:

---

## Conclusiones

(Por completar tras el análisis)

## Fecha de Ejecución

- Inicio: 2026-02-26
- Última actualización: 2026-02-26