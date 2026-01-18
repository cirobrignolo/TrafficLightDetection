import cv2
import numpy as np

# Cargar recorte de la luz amarilla
img = cv2.imread('semaforo_recorte.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, w = img.shape[:2]
center_y, center_x = h // 2, w // 2

print("="*60)
print("ANÁLISIS DEL AMARILLO - CENTRO DE LA LUZ")
print("="*60)
print(f"Dimensiones del recorte: {w}x{h}")
print(f"Centro: x={center_x}, y={center_y}")

# Analizar píxeles en diferentes radios desde el centro
for radius in [1, 2, 3]:
    print(f"\n📍 Radio {radius} desde el centro:")

    # Crear máscara circular
    y_coords, x_coords = np.ogrid[:h, :w]
    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2

    # Extraer píxeles dentro del círculo
    pixels_bgr = img[mask]
    pixels_hsv = hsv[mask]

    if len(pixels_bgr) > 0:
        mean_bgr = np.mean(pixels_bgr, axis=0)
        mean_hsv = np.mean(pixels_hsv, axis=0)

        print(f"   Cantidad de píxeles: {len(pixels_bgr)}")
        print(f"   BGR promedio: B={mean_bgr[0]:.1f}, G={mean_bgr[1]:.1f}, R={mean_bgr[2]:.1f}")
        print(f"   HSV promedio: H={mean_hsv[0]:.1f}, S={mean_hsv[1]:.1f}, V={mean_hsv[2]:.1f}")

# Píxel central exacto
center_pixel_bgr = img[center_y, center_x]
center_pixel_hsv = hsv[center_y, center_x]

print(f"\n🎯 Píxel CENTRAL exacto (x={center_x}, y={center_y}):")
print(f"   BGR: B={center_pixel_bgr[0]}, G={center_pixel_bgr[1]}, R={center_pixel_bgr[2]}")
print(f"   HSV: H={center_pixel_hsv[0]}, S={center_pixel_hsv[1]}, V={center_pixel_hsv[2]}")

print("\n" + "="*60)
print("RECOMENDACIÓN:")
print("="*60)
print("Usar los valores del radio 2 o 3 como el amarillo objetivo")
print("="*60)
