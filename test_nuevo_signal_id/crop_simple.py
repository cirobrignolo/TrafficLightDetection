import cv2

# Cargar frame amarillo
img = cv2.imread('frame_0141.jpg')

# Recortar exactamente:
# x: 190 a 290
# y: desde centro (424) - 200 = 224
x_start, x_end = 225, 240  # Sacado 5 izq (220+5), 10 der (250-10)
y_start, y_end = 306, 322  # Shift abajo: +2 arriba (304+2), +2 abajo (320+2)

crop = img[y_start:y_end, x_start:x_end]

# Guardar
cv2.imwrite('semaforo_recorte.jpg', crop)

print(f"✅ Recorte guardado: semaforo_recorte.jpg")
print(f"   Dimensiones: {crop.shape[1]}x{crop.shape[0]} píxeles")
print(f"   Coordenadas: x=[{x_start}:{x_end}], y=[{y_start}:{y_end}]")
