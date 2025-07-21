import cv2
import os
import math

video_path = 'inputTesis/semaforo_rojo_a_verde_cuadrado.mp4'
output_dir = 'frames_auto_labeled'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1.0:
    print("⚠️ No se pudo obtener el FPS correctamente. Usando 30 por defecto.")
    fps = 30.0
target_fps = 8
frame_interval = max(1, math.ceil(fps / target_fps))

print(f"🎥 Video FPS: {fps:.2f}")
print(f"📸 Guardando 1 frame cada {frame_interval} frames para lograr ~{target_fps} fps\n")

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"frame_{saved_count:04d}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, frame)
        print(f"🖼️ Guardado: {filename}")
        saved_count += 1

    display = frame.copy()
    cv2.putText(display, f"Frame {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Video", display)

    key = cv2.waitKey(60) & 0xFF
    if key == 27:  # ESC
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Total de frames guardados: {saved_count}")
