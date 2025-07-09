import cv2
import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tlr.pipeline import load_pipeline

# === CONFIGURACIÓN ===
image_path = "inputTesis/IMG_2252.JPEG"
output_image_path = "examples/output_debug.jpg"
output_boxes_path = "examples/output_boxes.txt"

# Leer proyecciones desde archivo
projection_bboxes = []
with open("projection_bboxes.txt", "r") as f:
    for line in f:
        projection_bboxes.append(eval(line.strip()))

# === CARGA DE IMAGEN ===
image_np = cv2.imread(image_path)
if image_np is None:
    raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

# Convertir imagen a tensor (H, W, C) y pasar a GPU
image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')

print("Image shape (final):", image_tensor.shape)

# === CARGA DE PIPELINE Y EJECUCIÓN ===
pipeline = load_pipeline('cuda:0')
valid_detections, recognitions, assignments, invalid_detections = pipeline(image_tensor, projection_bboxes)

print("Valid detections:", valid_detections)
print("Recognitions:", recognitions)
print("Assignments:", assignments)

# === COLORES Y OUTPUT ===
color_map = { 0: "black", 1: "red", 2: "yellow", 3: "green" }
os.makedirs("examples", exist_ok=True)

# Dibujar bounding boxes proyectadas (azul)
for box in projection_bboxes:
    x1, y1, x2, y2, _ = box
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image_np, "projection", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Dibujar detecciones válidas (verde)
with open(output_boxes_path, "w") as f:
    for i, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        scores = recognitions[i].tolist()
        predicted_class = int(torch.argmax(recognitions[i]))
        predicted_color = color_map[predicted_class]

        # Dibujar
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{predicted_color}"
        cv2.putText(image_np, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Guardar línea
        line = f"{x1},{y1},{x2},{y2},{predicted_color}," + ",".join(f"{s:.4f}" for s in scores) + "\n"
        f.write(line)

print(f"📄 Boxes guardados en {output_boxes_path}")
cv2.imwrite(output_image_path, image_np)
print(f"🖼️ Imagen anotada guardada en {output_image_path}")
