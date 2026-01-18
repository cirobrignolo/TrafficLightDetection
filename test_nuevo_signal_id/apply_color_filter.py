import cv2
import numpy as np
import os

# Configuración
input_dir = 'input_frames/'
output_dir = 'frames_with_filter/'

print("🎨 APLICANDO FILTRO DE ATARDECER/TONALIDAD CÁLIDA A TODOS LOS FRAMES")
print(f"📂 Input: {input_dir}")
print(f"📁 Output: {output_dir}")
print(f"🌅 Simulando cambio de iluminación (atardecer/nublado)")

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

def warm_sunset_filter(image):
    """
    Convierte verde del semáforo en amarillo.
    Basado en análisis real:
    - Verde: H=91, S=142, V=216
    - Amarillo: H=26, S=161, V=199
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Máscara para verdes del semáforo (H: 70-110, incluye el 91)
    green_mask = (hue >= 70) & (hue <= 110)

    # SHIFT de hue: 91 → 35 = -56 (más amarillo, menos naranja)
    hue = np.where(green_mask, np.clip(hue - 56, 0, 179), hue)

    # Subir saturación: 142 → 161 (+13%)
    sat = np.where(green_mask, np.clip(sat * 1.13, 0, 255), sat)

    # Bajar value: 216 → 199 (-8%)
    val = np.where(green_mask, np.clip(val * 0.92, 0, 255), val)

    hsv[:, :, 0] = hue
    hsv[:, :, 1] = sat
    hsv[:, :, 2] = val

    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return result

# Obtener lista de frames ordenados
frame_files = sorted([f for f in os.listdir(input_dir) if f.startswith('frame_') and f.endswith('.jpg')])

print(f"📊 Total frames encontrados: {len(frame_files)}")

processed_count = 0
filtered_count = 0

for frame_file in frame_files:
    # Extraer número de frame
    frame_num = int(frame_file.split('_')[1].split('.')[0])

    # Leer imagen
    input_path = os.path.join(input_dir, frame_file)
    image = cv2.imread(input_path)

    if image is None:
        print(f"⚠️  No se pudo leer: {frame_file}")
        continue

    # Aplicar filtro de atardecer a TODOS los frames
    output_image = warm_sunset_filter(image)
    filtered_count += 1
    status = "🌅 FILTRADO"

    # Guardar resultado
    output_path = os.path.join(output_dir, frame_file)
    cv2.imwrite(output_path, output_image)

    processed_count += 1

    # Mostrar progreso cada 50 frames
    if processed_count % 50 == 0:
        print(f"{status} {frame_file} ({processed_count}/{len(frame_files)})")

print(f"\n✅ Procesamiento completado:")
print(f"   📊 Total frames procesados: {processed_count}")
print(f"   🌅 Frames con filtro aplicado: {filtered_count}")
print(f"   📁 Guardados en: {output_dir}")
print(f"\n🔍 Para verificar el filtro:")
print(f"   Compara cualquier frame original vs filtrado, por ejemplo:")
print(f"   input_frames/frame_0000.jpg vs frames_with_filter/frame_0000.jpg")
