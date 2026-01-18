import cv2
import numpy as np

# Cargar el recorte del semáforo
semaforo = cv2.imread('semaforo_recorte.jpg')

# Convertir a HSV
hsv = cv2.cvtColor(semaforo, cv2.COLOR_BGR2HSV)

print("="*60)
print("ANÁLISIS DEL SEMÁFORO AMARILLO (RECORTE PRECISO)")
print("="*60)

# Encontrar el píxel más brillante (la luz amarilla encendida)
max_idx = np.unravel_index(hsv[:, :, 2].argmax(), hsv[:, :, 2].shape)
y_max, x_max = max_idx

pixel_bgr = semaforo[y_max, x_max]
pixel_hsv = hsv[y_max, x_max]

print(f"\n🔆 Píxel MÁS BRILLANTE (luz amarilla):")
print(f"   BGR: B={pixel_bgr[0]}, G={pixel_bgr[1]}, R={pixel_bgr[2]}")
print(f"   HSV: H={pixel_hsv[0]}, S={pixel_hsv[1]}, V={pixel_hsv[2]}")

# Promedio de toda la región de la luz amarilla
# Filtrar solo píxeles brillantes (valor > 150)
bright_mask = hsv[:, :, 2] > 150
if np.any(bright_mask):
    bright_pixels_bgr = semaforo[bright_mask]
    bright_pixels_hsv = hsv[bright_mask]

    mean_bright_bgr = np.mean(bright_pixels_bgr, axis=0)
    mean_bright_hsv = np.mean(bright_pixels_hsv, axis=0)

    print(f"\n💡 Promedio píxeles BRILLANTES (V>150) - luz amarilla activa:")
    print(f"   Cantidad de píxeles: {np.sum(bright_mask)}")
    print(f"   BGR: B={mean_bright_bgr[0]:.1f}, G={mean_bright_bgr[1]:.1f}, R={mean_bright_bgr[2]:.1f}")
    print(f"   HSV: H={mean_bright_hsv[0]:.1f}, S={mean_bright_hsv[1]:.1f}, V={mean_bright_hsv[2]:.1f}")

# Promedio de TODO el recorte
mean_total_bgr = cv2.mean(semaforo)[:3]
mean_total_hsv = cv2.mean(hsv)[:3]

print(f"\n📊 Promedio TOTAL del recorte:")
print(f"   BGR: B={mean_total_bgr[0]:.1f}, G={mean_total_bgr[1]:.1f}, R={mean_total_bgr[2]:.1f}")
print(f"   HSV: H={mean_total_hsv[0]:.1f}, S={mean_total_hsv[1]:.1f}, V={mean_total_hsv[2]:.1f}")

print("\n" + "="*60)
print("VALORES RECOMENDADOS PARA EL FILTRO:")
print("="*60)
print("Usar los valores de 'Promedio píxeles BRILLANTES' para")
print("convertir verde -> amarillo en el filtro")
print("="*60)
