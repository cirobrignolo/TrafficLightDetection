import cv2
import numpy as np

# Cargar frame con verde
green_frame = cv2.imread('frame_0000.jpg')

# Coordenadas del recorte final
x_start, x_end = 226, 242
y_start = 306
y_end = 321

# Recortar semáforo verde
traffic_light_green = green_frame[y_start:y_end, x_start:x_end]

# Guardar recorte
cv2.imwrite('semaforo_recorte_verde.jpg', traffic_light_green)

print("="*60)
print("ANÁLISIS DEL SEMÁFORO VERDE RECORTADO")
print("="*60)
print(f"Coordenadas: x[{x_start}:{x_end}], y[{y_start}:{y_end}]")

# Convertir a HSV
tl_hsv = cv2.cvtColor(traffic_light_green, cv2.COLOR_BGR2HSV)

# Encontrar píxel más brillante
max_brightness_idx = np.unravel_index(tl_hsv[:, :, 2].argmax(), tl_hsv[:, :, 2].shape)
y_max, x_max = max_brightness_idx

pixel_bgr = traffic_light_green[y_max, x_max]
pixel_hsv = tl_hsv[y_max, x_max]

print(f"\n🔆 Píxel MÁS BRILLANTE (luz verde encendida):")
print(f"   BGR: B={pixel_bgr[0]}, G={pixel_bgr[1]}, R={pixel_bgr[2]}")
print(f"   HSV: H={pixel_hsv[0]}, S={pixel_hsv[1]}, V={pixel_hsv[2]}")

# Promedio de píxeles brillantes (V>150)
bright_mask = tl_hsv[:, :, 2] > 150
if np.any(bright_mask):
    bright_pixels_bgr = traffic_light_green[bright_mask]
    bright_pixels_hsv = tl_hsv[bright_mask]

    mean_bright_bgr = np.mean(bright_pixels_bgr, axis=0)
    mean_bright_hsv = np.mean(bright_pixels_hsv, axis=0)

    print(f"\n💡 Promedio píxeles BRILLANTES (V>150) - luz verde activa:")
    print(f"   Cantidad de píxeles: {np.sum(bright_mask)}")
    print(f"   BGR: B={mean_bright_bgr[0]:.1f}, G={mean_bright_bgr[1]:.1f}, R={mean_bright_bgr[2]:.1f}")
    print(f"   HSV: H={mean_bright_hsv[0]:.1f}, S={mean_bright_hsv[1]:.1f}, V={mean_bright_hsv[2]:.1f}")

# Promedio total del recorte
mean_total_bgr = cv2.mean(traffic_light_green)[:3]
mean_total_hsv = cv2.mean(tl_hsv)[:3]

print(f"\n📊 Promedio TOTAL del recorte:")
print(f"   BGR: B={mean_total_bgr[0]:.1f}, G={mean_total_bgr[1]:.1f}, R={mean_total_bgr[2]:.1f}")
print(f"   HSV: H={mean_total_hsv[0]:.1f}, S={mean_total_hsv[1]:.1f}, V={mean_total_hsv[2]:.1f}")

print("\n" + "="*60)
print("✅ Guardado: semaforo_recorte_verde.jpg")
print("="*60)
