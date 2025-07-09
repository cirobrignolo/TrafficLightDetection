import cv2
import os
import sys
import torch
import numpy as np

# 🔧 Fix para importar tlr.pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tlr.pipeline import load_pipeline

# Config
input_dir = 'frames_labeled'
bbox_file = os.path.join(input_dir, 'projection_bboxes_master.txt')
output_dir = os.path.join(input_dir, 'outputs')
os.makedirs(output_dir, exist_ok=True)

# Cargar pipeline
pipeline = load_pipeline('cuda:0')  # o 'cpu'
print("🚀 Pipeline cargado")

# Leer archivo de bboxes
entries = {}
with open(bbox_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        frame = parts[0]
        bbox = list(map(int, parts[1:]))
        entries.setdefault(frame, []).append(bbox)

# Salida global
results_path = os.path.join(output_dir, "results.csv")
with open(results_path, "w") as out:
    out.write("frame,x1,y1,x2,y2,color,score_red,score_yellow,score_green,score_unknown\n")

    for frame_name, bboxes in entries.items():
        frame_path = os.path.join(input_dir, frame_name)
        image_np = cv2.imread(frame_path)

        if image_np is None:
            print(f"❌ No se pudo cargar {frame_name}, se salta.")
            continue

        image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
        #image_tensor = image_tensor.permute(1, 2, 0)

        # Ejecutar pipeline
        valid_detections, recognitions, assignments, invalid_detections = pipeline(image_tensor, bboxes)

        # Dibujar detecciones
        for i, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            scores = recognitions[i].tolist()
            predicted_class = int(torch.argmax(recognitions[i]))
            color = ['black', 'red', 'yellow', 'green'][predicted_class]

            # Guardar fila en results
            out.write(f"{frame_name},{x1},{y1},{x2},{y2},{color}," +
                      ",".join(f"{s:.4f}" for s in scores) + "\n")

            # Dibujar caja
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, color, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Guardar imagen de salida
        out_img_path = os.path.join(output_dir, frame_name.replace(".jpg", "_output.jpg"))
        cv2.imwrite(out_img_path, image_np)
        print(f"✅ {frame_name} procesado → {out_img_path}")

print(f"\n📄 Resultados guardados en: {results_path}")
