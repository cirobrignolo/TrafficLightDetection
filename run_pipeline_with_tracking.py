import cv2
import os
import sys
import time
import torch
import numpy as np
import ast

# 🔧 Fix para importar tlr.pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tlr.pipeline import load_pipeline

# ─── CONFIG ───────────────────────────────────────────────────────────────────
#input_dir = 'frames_labeled'
input_dir = 'frames_auto_labeled'
bbox_file = os.path.join(input_dir, 'projection_bboxes_master.txt')
output_dir = os.path.join(input_dir, 'outputs_with_tracking')
os.makedirs(output_dir, exist_ok=True)

# ─── CARGA DEL PIPELINE (CON TRACKING) ─────────────────────────────────────────
pipeline = load_pipeline('cuda:0')  # o 'cpu'
print("🚀 Pipeline cargado con tracking")

# ─── LEER PROYECCIONES POR FRAME ───────────────────────────────────────────────
# Formato CSV: frame_name,x1,y1,x2,y2,proj_id
entries = {}
with open(bbox_file, 'r') as f:
    for line in f:
        parts = line.strip().split(",")
        frame = parts[0]
        bbox = list(map(int, parts[1:]))
        entries.setdefault(frame, []).append(bbox)

# ─── PROCESAR CADA FRAME ───────────────────────────────────────────────────────
results_path = os.path.join(output_dir, 'results.csv')
with open(results_path, 'w') as out:
    # Cabecera: agregamos pred_color, revised_color, blink_flag y luego scores
    out.write("frame,x1,y1,x2,y2,pred_color,revised_color,blink,score_black,score_red,score_yellow,score_green\n")

    for frame_idx, (frame_name, bboxes) in enumerate(entries.items()):
        frame_path = os.path.join(input_dir, frame_name)
        image_np   = cv2.imread(frame_path)
        if image_np is None:
            print(f"❌ No se pudo cargar {frame_name}, salto.")
            continue
    
        # Convertir a tensor H×W×3 float32 (tal como espera preprocess4det) y pasar a GPU
        image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
    
        # Timestamp para el tracker (puede ser time.time() o frame_idx/fps)
        frame_ts = time.time()
    
        # Ejecutar pipeline
        valid, recognitions, assignments, invalid, revised = pipeline(
            image_tensor, bboxes, frame_ts
        )
    
        # Construir map: det_idx → proj_id
        assign_list = assignments.cpu().tolist()  # [[proj_id, det_idx], ...]
        assign_map  = {det_idx: proj_id for proj_id, det_idx in assign_list}
    
        # Escribir resultados por detección
        with open(results_path, 'a') as out:
            for det_idx, det in enumerate(valid):
                x1, y1, x2, y2 = map(int, det[1:5])
                scores         = recognitions[det_idx].tolist()
                pred_cls       = int(torch.argmax(recognitions[det_idx]))
                pred_color     = ['black','red','yellow','green'][pred_cls]
    
                proj_id        = assign_map.get(det_idx)
                rev_color, blink = revised.get(proj_id, (pred_color, False))
    
                out.write(
                    f"{frame_name},{x1},{y1},{x2},{y2},"
                    f"{pred_color},{rev_color},{int(blink)},"
                    f"{scores[0]:.4f},{scores[1]:.4f},"
                    f"{scores[2]:.4f},{scores[3]:.4f}\n"
                )
    
                # Dibujar bbox y etiqueta predicha
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image_np, pred_color, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
        # Dibujar color revisado en la caja de proyección
        for x1, y1, x2, y2, pid in bboxes:
            rev_color, blink = revised.get(pid, ('none', False))
            txt = f"{rev_color}{'*' if blink else ''}"
            cv2.putText(image_np, txt, (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    
        # Guardar imagen anotada
        out_img = os.path.join(output_dir, frame_name.replace('.jpg','_out.jpg'))
        cv2.imwrite(out_img, image_np)
        print(f"✅ {frame_name} procesado → {out_img}")

print(f"\n📄 Resultados completos en: {results_path}")
