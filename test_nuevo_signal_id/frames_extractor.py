import cv2
import os
import math

# Configuración para el video de flecha roja
video_path = 'input/campana_semaforo.mp4'  # Ruta relativa al video
output_dir = 'input_frames/'  # Guardar en esta misma carpeta (reemplazará los frames existentes)

print("🎥 AUTO-EXTRACCIÓN DE FRAMES PARA VIDEO DOBLE CHICO")
print(f"📂 Input: {video_path}")
print(f"📁 Output: {output_dir}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error: No se pudo abrir el video {video_path}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"📊 Video info:")
print(f"   FPS: {fps:.2f}")
print(f"   Total frames: {total_frames}")
print(f"   Duración: {duration:.2f}s")

# CONFIGURACIÓN PARA PRUEBAS: Extraer 1 de cada 5 frames (reduce cantidad para testing)
# Para producción: usar frame_interval = 1 (todos los frames)
frame_interval = 5

print(f"📸 Guardando 1 frame cada {frame_interval} frames (modo pruebas)")
print(f"📈 Frames estimados: ~{total_frames // frame_interval}")

# Limpiar frames existentes
existing_frames = [f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.jpg')]
if existing_frames:
    print(f"🧹 Eliminando {len(existing_frames)} frames existentes...")
    for frame_file in existing_frames:
        os.remove(os.path.join(output_dir, frame_file))

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Rotar frame 90 grados a la derecha (para corregir orientación del video)
        # El video tiene metadata de rotación que OpenCV no respeta automáticamente
        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        filename = f"frame_{saved_count:04d}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, rotated_frame)

        # Obtener dimensiones del frame rotado
        h, w = rotated_frame.shape[:2]
        print(f"🖼️  Guardado: {filename} (dimensiones: {w}x{h}, rotado 90°)")
        saved_count += 1

    # Mostrar progreso cada 30 frames
    if frame_count % 30 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"📈 Progreso: {progress:.1f}% ({frame_count}/{total_frames})")

    frame_count += 1

cap.release()
print(f"\n✅ Extracción completada:")
print(f"   📊 Total frames procesados: {frame_count}")
print(f"   🖼️  Total frames guardados: {saved_count}")
print(f"   📁 Guardados en: {output_dir}")
print(f"\n🎯 Próximo paso:")
print(f"   1. Ejecutar: python3 select_projection_and_append.py frame_0000.jpg")
print(f"   2. Marcar manualmente las proyecciones en algunos frames clave")
print(f"   3. Ejecutar: python3 run_pipeline_debug_stages_fixed.py")