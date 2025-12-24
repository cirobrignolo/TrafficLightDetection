import cv2
import sys
import os
import time
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tlr.pipeline import load_pipeline

# === CONFIGURACIÓN ===
image_path         = "inputTesis/IMG_2252.JPEG"
output_image_path  = "examples/output_debug_tracking.jpg"
output_boxes_path  = "examples/output_boxes_tracking.txt"
os.makedirs("examples", exist_ok=True)

# Leer proyecciones desde archivo
projection_bboxes = []
with open("projection_bboxes.txt", "r") as f:
    for line in f:
        projection_bboxes.append(eval(line.strip()))

# === CARGA DE IMAGEN ===
image_np = cv2.imread(image_path)
if image_np is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

# Convertir imagen a tensor (C, H, W) y pasar a GPU
image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')

print("Image shape (final):", image_tensor.shape)

# Timestamp del frame en segundos
frame_ts = time.time()

print("Image tensor shape:", image_tensor.shape)

# === CARGA DE PIPELINE Y EJECUCIÓN ===
pipeline = load_pipeline('cuda:0')
valid_detections, recognitions, assignments, invalid_detections, revised = \
    pipeline(image_tensor, projection_bboxes, frame_ts)

print("Valid detections:", valid_detections)
print("Recognitions     :", recognitions)
print("Assignments      :", assignments)
print("Revised states   :", revised)

# === DIBUJOS Y GUARDADO ===
color_map = { 0: "black", 1: "red", 2: "yellow", 3: "green" }

# 1) Proyecciones (azul)
for box in projection_bboxes:
    x1, y1, x2, y2, pid = box
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image_np, f"proj {pid}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

# 2) Detecciones y reconocimientos (verde)
with open(output_boxes_path, "w") as f:
    for i, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        scores = recognitions[i].tolist()
        cls = int(torch.argmax(recognitions[i]))
        col = color_map[cls]

        # Dibujar bbox y etiqueta
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_np, col, (x1, y1-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Guardar línea en el txt
        line = f"{x1},{y1},{x2},{y2},{col}," + ",".join(f"{s:.4f}" for s in scores)
        f.write(line + "\n")

# 3) Estados revisados (amarillo)
if revised:
    for signal_id, (col, blink) in revised.items():
        # signal_id es del formato "signal_0", "signal_1", etc.
        # Extraer el número y buscar la caja correspondiente
        # El box_id en projection_bboxes es el quinto elemento (índice 4)
        try:
            # Extraer número del signal_id (ej: "signal_0" → 0)
            if signal_id.startswith("signal_"):
                box_id_num = int(signal_id.split("_")[1])
            else:
                continue  # Skip si no es formato esperado

            # Buscar la projection box con ese ID
            for x1, y1, x2, y2, box_id in projection_bboxes:
                if box_id == box_id_num:
                    text = f"{col}{' *' if blink else ''}"
                    cv2.putText(image_np, text, (x1, y2+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    break
        except (ValueError, IndexError):
            continue  # Skip si hay error parseando el signal_id

# Guardar resultados
cv2.imwrite(output_image_path, image_np)
print(f"📄 Boxes guardados en {output_boxes_path}")
print(f"🖼️ Imagen anotada guardada en {output_image_path}")
