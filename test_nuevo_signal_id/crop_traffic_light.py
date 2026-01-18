import cv2
import numpy as np

# Cargar frame con amarillo real
yellow_frame = cv2.imread('frame_0141.jpg')

# Recortar exactamente como indicaste:
# x: 190 a 290 (píxeles verticales/columnas)
# y: desde el centro (424) 200 píxeles para arriba = 224, y subir más

x_start, x_end = 190, 290
y_start = 424 - 200  # Centro menos 200 = 224
y_end = y_start + 60  # Agregar altura del semáforo

# Recortar semáforo
traffic_light = yellow_frame[y_start:y_end, x_start:x_end]

# Guardar recorte
cv2.imwrite('traffic_light_crop_yellow.jpg', traffic_light)

# Analizar píxeles del semáforo recortado
print("="*60)
print("ANÁLISIS DEL SEMÁFORO RECORTADO")
print("="*60)

# Convertir a HSV
tl_hsv = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)

# Estadísticas de toda la región del semáforo
mean_bgr = cv2.mean(traffic_light)[:3]
mean_hsv = cv2.mean(tl_hsv)[:3]

print(f"\n📊 Promedio TOTAL del semáforo amarillo:")
print(f"   BGR: B={mean_bgr[0]:.1f}, G={mean_bgr[1]:.1f}, R={mean_bgr[2]:.1f}")
print(f"   HSV: H={mean_hsv[0]:.1f}, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")

# Encontrar el píxel MÁS BRILLANTE (parte encendida del semáforo)
max_brightness_idx = np.unravel_index(tl_hsv[:, :, 2].argmax(), tl_hsv[:, :, 2].shape)
y_max, x_max = max_brightness_idx

pixel_bgr = traffic_light[y_max, x_max]
pixel_hsv = tl_hsv[y_max, x_max]

print(f"\n🔆 Píxel MÁS BRILLANTE (luz amarilla encendida):")
print(f"   Coordenadas en recorte: ({x_max}, {y_max})")
print(f"   BGR: B={pixel_bgr[0]}, G={pixel_bgr[1]}, R={pixel_bgr[2]}")
print(f"   HSV: H={pixel_hsv[0]}, S={pixel_hsv[1]}, V={pixel_hsv[2]}")

# Región 5x5 alrededor del píxel más brillante
region_5x5 = traffic_light[max(0, y_max-2):min(traffic_light.shape[0], y_max+3),
                           max(0, x_max-2):min(traffic_light.shape[1], x_max+3)]
mean_5x5_bgr = cv2.mean(region_5x5)[:3]

region_hsv = cv2.cvtColor(region_5x5, cv2.COLOR_BGR2HSV)
mean_5x5_hsv = cv2.mean(region_hsv)[:3]

print(f"\n📊 Promedio 5x5 píxeles centrales (luz amarilla):")
print(f"   BGR: B={mean_5x5_bgr[0]:.1f}, G={mean_5x5_bgr[1]:.1f}, R={mean_5x5_bgr[2]:.1f}")
print(f"   HSV: H={mean_5x5_hsv[0]:.1f}, S={mean_5x5_hsv[1]:.1f}, V={mean_5x5_hsv[2]:.1f}")

# Hacer lo mismo con el verde
green_frame = cv2.imread('input_frames/frame_0000.jpg')
if green_frame is not None:
    green_tl = green_frame[y_start:y_end, x_start:x_end]
    cv2.imwrite('traffic_light_crop_green.jpg', green_tl)

    green_tl_hsv = cv2.cvtColor(green_tl, cv2.COLOR_BGR2HSV)
    max_green_idx = np.unravel_index(green_tl_hsv[:, :, 2].argmax(), green_tl_hsv[:, :, 2].shape)

    green_pixel_bgr = green_tl[max_green_idx]
    green_pixel_hsv = green_tl_hsv[max_green_idx]

    print(f"\n🟢 Para comparación - Verde brillante:")
    print(f"   BGR: B={green_pixel_bgr[0]}, G={green_pixel_bgr[1]}, R={green_pixel_bgr[2]}")
    print(f"   HSV: H={green_pixel_hsv[0]}, S={green_pixel_hsv[1]}, V={green_pixel_hsv[2]}")

print("\n" + "="*60)
print("ARCHIVOS GENERADOS:")
print("="*60)
print("✅ traffic_light_crop_yellow.jpg - Recorte del semáforo amarillo")
print("✅ traffic_light_crop_green.jpg - Recorte del semáforo verde")
print("="*60)
