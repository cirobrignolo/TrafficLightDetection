import cv2
import os

video_path = 'inputTesis/semaforo_rojo_a_verde_cuadrado.mp4'
output_dir = 'frames_labeled'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
saved_count = 0

print("▶️ Reproduciendo video")
print("Presioná 's' para guardar el frame actual")
print("Presioná 'ESC' para salir\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, f"Frame {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("Video", display)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        filename = f"frame_{saved_count:04d}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, frame)
        print(f"🖼️ Guardado: {filename}")
        saved_count += 1
    elif key == 27:  # ESC
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Total de frames guardados: {saved_count}")
