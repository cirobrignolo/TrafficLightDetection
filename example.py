import cv2
import sys
import os
import torch
import numpy as np

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tlr.pipeline import load_pipeline

# Load image (deberías tenerla en examples/)
image_np = cv2.imread("inputTesis/IMG_2252.JPEG")

# Validar que se haya cargado correctamente
if image_np is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verificá la ruta.")

# Convertir imagen a tensor (C, H, W) y enviar a GPU
#image = torch.from_numpy(image_np).permute(2, 0, 1).float()
image = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
print("Image shape (final):", image.shape)

# Dummy projection bounding boxes (x1, y1, x2, y2, id)
projection_bboxes = [
    [727, 702, 1087, 944, 0]
]

# Load the traffic light recognition pipeline
#pipeline = load_pipeline('cpu')

pipeline = load_pipeline('cuda:0')

# Run detection and recognition
valid_detections, recognitions, assignments, invalid_detections = pipeline(image, projection_bboxes)

# Output results
print("Valid detections:", valid_detections)
print("Recognitions:", recognitions)
print("Assignments:", assignments)
print("Invalid detections:", invalid_detections)

# Guardar los bounding boxes en un archivo
output_boxes_path = 'examples/output_boxes.txt'

# Mapear índice de clase a color
color_map = { 0: "black", 1: "red", 2: "yellow", 3: "green" }

with open(output_boxes_path, "w") as f:
    for i, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        scores = recognitions[i].tolist()
        predicted_class = int(torch.argmax(recognitions[i]))
        predicted_color = color_map[predicted_class]

        line = f"{x1},{y1},{x2},{y2},{predicted_color}," + ",".join(f"{s:.4f}" for s in scores) + "\n"
        f.write(line)

print(f"📄 Bounding boxes guardados en '{output_boxes_path}'")

# Optional: draw and save results
for det in valid_detections:
    x1, y1, x2, y2 = map(int, det[1:5])  # el bbox está en columnas 1 a 4
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite('examples/output.jpg', image_np)
print("🖼️ Imagen guardada en 'examples/output.jpg'")
