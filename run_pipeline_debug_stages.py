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
input_dir = 'frames_auto_labeled'
bbox_file = os.path.join(input_dir, 'projection_bboxes_master.txt')
output_dir = os.path.join(input_dir, 'outputs_debug_stages')
os.makedirs(output_dir, exist_ok=True)

# ─── CARGA DEL PIPELINE (CON TRACKING) ─────────────────────────────────────────
pipeline = load_pipeline('cuda:0')  # o 'cpu'
print("🚀 Pipeline cargado con tracking y debug de etapas")

# ─── LEER PROYECCIONES POR FRAME ───────────────────────────────────────────────
entries = {}
with open(bbox_file, 'r') as f:
    for line in f:
        parts = line.strip().split(",")
        frame = parts[0]
        bbox = list(map(int, parts[1:]))
        entries.setdefault(frame, []).append(bbox)

# ─── SEPARATE CSV FILES FOR EACH STAGE ──────────────────────────────────────
detection_csv = os.path.join(output_dir, '1_detection_results.csv')
recognition_csv = os.path.join(output_dir, '2_recognition_results.csv')
final_csv = os.path.join(output_dir, '3_final_results.csv')

# Clean start: remove existing CSVs
for csv_file in [detection_csv, recognition_csv, final_csv]:
    if os.path.exists(csv_file):
        os.remove(csv_file)

# Write headers for each file
with open(detection_csv, 'w') as out:
    out.write("frame,det_idx,x1,y1,x2,y2,tl_type,det_vert,det_quad,det_hori,det_bg\n")

with open(recognition_csv, 'w') as out:
    out.write("frame,det_idx,x1,y1,x2,y2,tl_type,pred_color,rec_black,rec_red,rec_yellow,rec_green\n")

with open(final_csv, 'w') as out:
    out.write("frame,det_idx,proj_id,x1,y1,x2,y2,tl_type,pred_color,revised_color,blink,det_vert,det_quad,det_hori,det_bg,rec_black,rec_red,rec_yellow,rec_green\n")

# Process each frame
for frame_idx, (frame_name, bboxes) in enumerate(entries.items()):
    frame_path = os.path.join(input_dir, frame_name)
    image_np = cv2.imread(frame_path)
    if image_np is None:
        print(f"❌ No se pudo cargar {frame_name}, salto.")
        continue

    # Convertir a tensor y pasar a GPU
    image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
    frame_ts = time.time()

    # Ejecutar pipeline y capturar resultados intermedios
    valid, recognitions, assignments, invalid, revised = pipeline(
        image_tensor, bboxes, frame_ts
    )

    # 📊 ANÁLISIS ETAPA POR ETAPA
    print(f"\n🔍 FRAME: {frame_name}")
    print(f"   Input projections: {len(bboxes)}")
    print(f"   Valid detections: {len(valid)}")
    print(f"   Invalid detections: {len(invalid)}")
    print(f"   Assignments: {len(assignments)}")
    print(f"   Revised states: {len(revised)}")
    
    # Construir mapas de asignación
    assign_list = assignments.cpu().tolist()
    assign_map = {det_idx: proj_id for proj_id, det_idx in assign_list}
    
    # ═══ ETAPA 1: DETECCIÓN CRUDA ═══
    detection_lines = []
        for det_idx, det in enumerate(valid):
            x1, y1, x2, y2 = map(int, det[1:5])
            det_scores = det[5:9].tolist()  # [vert, quad, hori, bg]
            tl_type = int(torch.argmax(det[5:9]))
            type_names = ['bg', 'vert', 'quad', 'hori']
            
            detection_lines.append(
                f"{frame_name},{det_idx},{x1},{y1},{x2},{y2},"
                f"{type_names[tl_type]},"
                f"{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f}\n"
            )
        
        if detection_lines:
            with open(detection_csv, 'a') as out:
                out.writelines(detection_lines)
        
        # ═══ ETAPA 2: RECONOCIMIENTO ═══
        recognition_lines = []
        for det_idx, det in enumerate(valid):
            x1, y1, x2, y2 = map(int, det[1:5])
            rec_scores = recognitions[det_idx].tolist() if det_idx < len(recognitions) else [0,0,0,0]
            pred_cls = int(torch.argmax(recognitions[det_idx])) if det_idx < len(recognitions) else 0
            pred_color = ['black','red','yellow','green'][pred_cls]
            tl_type = int(torch.argmax(det[5:9]))
            type_names = ['bg', 'vert', 'quad', 'hori']
            
            recognition_lines.append(
                f"{frame_name},{det_idx},{x1},{y1},{x2},{y2},"
                f"{type_names[tl_type]},{pred_color},"
                f"{rec_scores[0]:.4f},{rec_scores[1]:.4f},{rec_scores[2]:.4f},{rec_scores[3]:.4f}\n"
            )
        
        if recognition_lines:
            with open(recognition_csv, 'a') as out:
                out.writelines(recognition_lines)
        
        # ═══ ETAPA 3: ASIGNACIÓN + TRACKING ═══
        final_lines = []
        for det_idx, det in enumerate(valid):
            x1, y1, x2, y2 = map(int, det[1:5])
            det_scores = det[5:9].tolist()
            rec_scores = recognitions[det_idx].tolist() if det_idx < len(recognitions) else [0,0,0,0]
            pred_cls = int(torch.argmax(recognitions[det_idx])) if det_idx < len(recognitions) else 0
            pred_color = ['black','red','yellow','green'][pred_cls]
            tl_type = int(torch.argmax(det[5:9]))
            type_names = ['bg', 'vert', 'quad', 'hori']
            
            # Obtener resultado de tracking
            proj_id = assign_map.get(det_idx, -1)
            rev_color, blink = revised.get(proj_id, (pred_color, False)) if proj_id != -1 else (pred_color, False)
            
            final_lines.append(
                f"{frame_name},{det_idx},{proj_id},{x1},{y1},{x2},{y2},"
                f"{type_names[tl_type]},{pred_color},{rev_color},{int(blink)},"
                f"{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f},"
                f"{rec_scores[0]:.4f},{rec_scores[1]:.4f},{rec_scores[2]:.4f},{rec_scores[3]:.4f}\n"
            )
        
        if final_lines:
            with open(final_csv, 'a') as out:
                out.writelines(final_lines)
    
        # ═══ VISUALIZACIÓN MEJORADA ═══
        image_debug = image_np.copy()
        
        # Dibujar proyecciones originales (azul)
        for x1, y1, x2, y2, pid in bboxes:
            cv2.rectangle(image_debug, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Azul
            cv2.putText(image_debug, f"P{pid}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Dibujar detecciones válidas (verde)
        for det_idx, det in enumerate(valid):
            x1, y1, x2, y2 = map(int, det[1:5])
            cv2.rectangle(image_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde
            
            # Obtener resultado de tracking
            proj_id = assign_map.get(det_idx, -1)
            rev_color, blink = revised.get(proj_id, ('none', False)) if proj_id != -1 else ('none', False)
            
            # Mostrar información completa
            pred_cls = int(torch.argmax(recognitions[det_idx])) if det_idx < len(recognitions) else 0
            pred_color = ['black','red','yellow','green'][pred_cls]
            
            info_text = f"{pred_color}→{rev_color}"
            if blink:
                info_text += "*"
            if proj_id != -1:
                info_text += f" (P{proj_id})"
                
            cv2.putText(image_debug, info_text, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Dibujar detecciones inválidas (rojo)
        for det_idx, det in enumerate(invalid):
            x1, y1, x2, y2 = map(int, det[1:5])
            cv2.rectangle(image_debug, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Rojo
            cv2.putText(image_debug, "INVALID", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
        # Guardar imagen con debug completo
        out_img = os.path.join(output_dir, frame_name.replace('.jpg','_debug.jpg'))
        cv2.imwrite(out_img, image_debug)
        print(f"✅ {frame_name} procesado → {out_img}")

print(f"\n📄 Resultados detallados en: {results_path}")
print(f"🎨 Imágenes con debug en: {output_dir}/")
print(f"\n📊 Formato CSV:")
print(f"   stage=detection: Detecciones crudas del detector")
print(f"   stage=recognition: Después de clasificación de color") 
print(f"   stage=final: Después de asignación y tracking")