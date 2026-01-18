import cv2
import numpy as np

# Cargar frame con amarillo real
yellow_frame = cv2.imread('input_frames/frame_0141.jpg')

# Coordenadas aproximadas del semáforo amarillo (ajustar según imagen)
# Semáforo está aproximadamente en x=240-260, y=190-210
x_start, x_end = 240, 260
y_start, y_end = 190, 210

# Extraer región del semáforo amarillo
yellow_region = yellow_frame[y_start:y_end, x_start:x_end]

# Calcular valores promedio BGR
mean_bgr = cv2.mean(yellow_region)[:3]

# Convertir a HSV
yellow_region_hsv = cv2.cvtColor(yellow_region, cv2.COLOR_BGR2HSV)
mean_hsv = cv2.mean(yellow_region_hsv)[:3]

print("=" * 60)
print("ANÁLISIS DE SEMÁFORO AMARILLO REAL (frame_0141.jpg)")
print("=" * 60)
print(f"\n📊 Valores promedio BGR:")
print(f"   B (Azul):  {mean_bgr[0]:.1f}")
print(f"   G (Verde): {mean_bgr[1]:.1f}")
print(f"   R (Rojo):  {mean_bgr[2]:.1f}")
print(f"\n📊 Valores promedio HSV:")
print(f"   H (Hue):        {mean_hsv[0]:.1f} (rango 0-179)")
print(f"   S (Saturation): {mean_hsv[1]:.1f} (rango 0-255)")
print(f"   V (Value):      {mean_hsv[2]:.1f} (rango 0-255)")

# Analizar verde del frame_0000 para comparar
green_frame = cv2.imread('input_frames/frame_0000.jpg')
green_region = green_frame[y_start:y_end, x_start:x_end]
green_mean_hsv = cv2.mean(cv2.cvtColor(green_region, cv2.COLOR_BGR2HSV))[:3]

print(f"\n📊 Valores promedio HSV del VERDE (frame_0000 para comparación):")
print(f"   H (Hue):        {green_mean_hsv[0]:.1f}")
print(f"   S (Saturation): {green_mean_hsv[1]:.1f}")
print(f"   V (Value):      {green_mean_hsv[2]:.1f}")

print("\n" + "=" * 60)
print("RECOMENDACIÓN PARA FILTRO:")
print("=" * 60)
print(f"Target Hue para amarillo: {mean_hsv[0]:.0f}")
print(f"Target Saturation mínima: {max(mean_hsv[1] - 20, 180):.0f}")
print(f"Target Value (brillo): mantener original o {mean_hsv[2]:.0f}")
print("=" * 60)

# Guardar muestra visual
sample = np.zeros((100, 300, 3), dtype=np.uint8)
sample[:, :100] = mean_bgr  # Color amarillo real
sample[:, 100:200] = [0, 255, 255]  # Amarillo puro BGR
sample[:, 200:] = cv2.cvtColor(np.uint8([[[mean_hsv[0], 255, 255]]]),
                                cv2.COLOR_HSV2BGR)[0][0]  # Amarillo saturado

cv2.imwrite('yellow_color_samples.jpg', sample)
print("\n✅ Guardado: yellow_color_samples.jpg")
print("   [Izq: Amarillo real | Centro: Amarillo puro | Der: HSV saturado]")
