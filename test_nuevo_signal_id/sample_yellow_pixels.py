import cv2
import numpy as np

# Cargar frame con amarillo real
yellow_frame = cv2.imread('frame_0141.jpg')

# Mostrar frame para identificar coordenadas exactas del semáforo
# Voy a hacer click manual en el centro del semáforo amarillo
h, w = yellow_frame.shape[:2]
print(f"Dimensiones del frame: {w}x{h}")

# Buscar píxel más brillante y amarillento en la región del semáforo
# El semáforo está en centro-superior de la imagen
# Imagen 480x848 (ancho x alto), semáforo aprox en x=235-250, y=240-255
roi_x_start, roi_x_end = 230, 255
roi_y_start, roi_y_end = 235, 260

roi = yellow_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

# Convertir a HSV para encontrar el píxel más brillante
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Buscar píxel con mayor Value (brillo) en la ROI
max_brightness_idx = np.unravel_index(roi_hsv[:, :, 2].argmax(), roi_hsv[:, :, 2].shape)
y_roi, x_roi = max_brightness_idx

# Coordenadas absolutas
y_abs = roi_y_start + y_roi
x_abs = roi_x_start + x_roi

# Extraer valores BGR exactos del píxel más brillante
pixel_bgr = yellow_frame[y_abs, x_abs]
pixel_hsv = roi_hsv[y_roi, x_roi]

print(f"\n🎯 Píxel más brillante del semáforo amarillo:")
print(f"   Coordenadas: ({x_abs}, {y_abs})")
print(f"   BGR: B={pixel_bgr[0]}, G={pixel_bgr[1]}, R={pixel_bgr[2]}")
print(f"   HSV: H={pixel_hsv[0]}, S={pixel_hsv[1]}, V={pixel_hsv[2]}")

# Tomar muestra 3x3 alrededor del píxel más brillante
sample_3x3 = yellow_frame[y_abs-1:y_abs+2, x_abs-1:x_abs+2]
mean_bgr_3x3 = cv2.mean(sample_3x3)[:3]

sample_hsv = cv2.cvtColor(sample_3x3, cv2.COLOR_BGR2HSV)
mean_hsv_3x3 = cv2.mean(sample_hsv)[:3]

print(f"\n📊 Promedio 3x3 píxeles centrales del amarillo:")
print(f"   BGR: B={mean_bgr_3x3[0]:.1f}, G={mean_bgr_3x3[1]:.1f}, R={mean_bgr_3x3[2]:.1f}")
print(f"   HSV: H={mean_hsv_3x3[0]:.1f}, S={mean_hsv_3x3[1]:.1f}, V={mean_hsv_3x3[2]:.1f}")

# Comparar con verde del frame_0000
green_frame = cv2.imread('input_frames/frame_0000.jpg')
green_roi = green_frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
green_hsv = cv2.cvtColor(green_roi, cv2.COLOR_BGR2HSV)
max_green_idx = np.unravel_index(green_hsv[:, :, 2].argmax(), green_hsv[:, :, 2].shape)
green_pixel_hsv = green_hsv[max_green_idx]
green_pixel_bgr = green_roi[max_green_idx]

print(f"\n🟢 Para comparación - Verde brillante (frame_0000):")
print(f"   BGR: B={green_pixel_bgr[0]}, G={green_pixel_bgr[1]}, R={green_pixel_bgr[2]}")
print(f"   HSV: H={green_pixel_hsv[0]}, S={green_pixel_hsv[1]}, V={green_pixel_hsv[2]}")

# Crear visualización
vis = np.zeros((150, 400, 3), dtype=np.uint8)

# Píxel amarillo exacto
vis[:50, :100] = pixel_bgr

# Promedio 3x3 amarillo
vis[:50, 100:200] = mean_bgr_3x3

# Verde para comparar
vis[:50, 200:300] = green_pixel_bgr

# Amarillo puro teórico BGR (0, 255, 255)
vis[:50, 300:] = (0, 255, 255)

# Etiquetas
cv2.putText(vis, "Amarillo", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "pixel", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, "Amarillo", (110, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "3x3", (110, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, "Verde", (210, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "real", (210, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, "Amarillo", (310, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
cv2.putText(vis, "puro", (310, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

# Valores BGR
cv2.putText(vis, f"B:{int(pixel_bgr[0])}", (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"G:{int(pixel_bgr[1])}", (5, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"R:{int(pixel_bgr[2])}", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

cv2.putText(vis, f"B:{int(mean_bgr_3x3[0])}", (105, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"G:{int(mean_bgr_3x3[1])}", (105, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"R:{int(mean_bgr_3x3[2])}", (105, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

cv2.putText(vis, f"B:{green_pixel_bgr[0]}", (205, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"G:{green_pixel_bgr[1]}", (205, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, f"R:{green_pixel_bgr[2]}", (205, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

cv2.putText(vis, "B:0", (305, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, "G:255", (305, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.putText(vis, "R:255", (305, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

cv2.imwrite('yellow_analysis_visual.jpg', vis)
print(f"\n✅ Guardado: yellow_analysis_visual.jpg")

print("\n" + "="*60)
print("RECOMENDACIÓN PARA FILTRO:")
print("="*60)
print(f"Copiar exactamente estos valores BGR del amarillo real:")
print(f"  B = {int(mean_bgr_3x3[0])}")
print(f"  G = {int(mean_bgr_3x3[1])}")
print(f"  R = {int(mean_bgr_3x3[2])}")
print("="*60)
