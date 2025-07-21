import cv2
import os
import sys
import time
import torch
import numpy as np

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
print("🚀 Pipeline cargado con tracking y debug de etapas separadas")

# ─── LEER PROYECCIONES POR FRAME ───────────────────────────────────────────────
entries = {}
with open(bbox_file, 'r') as f:
    for line in f:
        parts = line.strip().split(",")
        frame = parts[0]
        bbox = list(map(int, parts[1:]))
        entries.setdefault(frame, []).append(bbox)

# ─── CREATE ORGANIZED FOLDER STRUCTURE ──────────────────────────────────────
detection_dir = os.path.join(output_dir, '1_detection')
recognition_dir = os.path.join(output_dir, '2_recognition')
final_dir = os.path.join(output_dir, '3_final')

# Create directories
for stage_dir in [detection_dir, recognition_dir, final_dir]:
    os.makedirs(stage_dir, exist_ok=True)

# CSV files in main output directory
detection_csv = os.path.join(output_dir, '1_detection_results.csv')
recognition_csv = os.path.join(output_dir, '2_recognition_results.csv')
final_csv = os.path.join(output_dir, '3_final_results.csv')
all_detections_csv = os.path.join(output_dir, '0_all_detections.csv')  # NEW: Shows ALL detections

# Clean start: remove existing files
for csv_file in [detection_csv, recognition_csv, final_csv, all_detections_csv]:
    if os.path.exists(csv_file):
        os.remove(csv_file)

# Write headers for each file
with open(detection_csv, 'w') as out:
    out.write("frame,det_idx,x1,y1,x2,y2,tl_type,det_vert,det_quad,det_hori,det_bg\n")

with open(recognition_csv, 'w') as out:
    out.write("frame,det_idx,x1,y1,x2,y2,tl_type,pred_color,rec_black,rec_red,rec_yellow,rec_green\n")

with open(final_csv, 'w') as out:
    out.write("frame,det_idx,proj_id,x1,y1,x2,y2,tl_type,pred_color,revised_color,blink,det_vert,det_quad,det_hori,det_bg,rec_black,rec_red,rec_yellow,rec_green\n")

with open(all_detections_csv, 'w') as out:
    out.write("frame,status,det_idx,x1,y1,x2,y2,tl_type,det_vert,det_quad,det_hori,det_bg\n")

# ─── PROCESS EACH FRAME ──────────────────────────────────────────────────────
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
    
    # ═══ ALL DETECTIONS (VALID + INVALID) ═══
    all_detection_lines = []
    
    # Add valid detections
    for det_idx, det in enumerate(valid):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']
        
        all_detection_lines.append(
            f"{frame_name},VALID,{det_idx},{x1},{y1},{x2},{y2},"
            f"{type_names[tl_type]},"
            f"{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f}\n"
        )
    
    # Add invalid detections
    for det_idx, det in enumerate(invalid):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']
        
        all_detection_lines.append(
            f"{frame_name},INVALID,{det_idx},{x1},{y1},{x2},{y2},"
            f"{type_names[tl_type]},"
            f"{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f}\n"
        )
    
    if all_detection_lines:
        with open(all_detections_csv, 'a') as out:
            out.writelines(all_detection_lines)
    
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

    # ═══ VISUALIZACIÓN POR ETAPAS ═══
    
    # ────── IMAGEN 1: ETAPA DETECCIÓN ──────
    img_detection = image_np.copy()
    
    # Título de la etapa
    cv2.putText(img_detection, "STAGE 1: DETECTION", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Proyecciones originales (azul)
    for x1, y1, x2, y2, pid in bboxes:
        cv2.rectangle(img_detection, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_detection, f"Proj{pid}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Detecciones válidas con scores de orientación
    for det_idx, det in enumerate(valid):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']
        
        cv2.rectangle(img_detection, (x1, y1), (x2, y2), (0, 255, 0), 3)
        info_text = f"Det{det_idx}: {type_names[tl_type]} ({det_scores[tl_type]:.2f})"
        cv2.putText(img_detection, info_text, (x1, y1-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Detecciones inválidas
    for det_idx, det in enumerate(invalid):
        x1, y1, x2, y2 = map(int, det[1:5])
        cv2.rectangle(img_detection, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img_detection, f"Invalid{det_idx}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Guardar imagen de detección en carpeta específica
    out_img1 = os.path.join(detection_dir, frame_name.replace('.jpg','_detection.jpg'))
    cv2.imwrite(out_img1, img_detection)
    
    # ────── IMAGEN 2: ETAPA RECONOCIMIENTO ──────
    img_recognition = image_np.copy()
    
    # Título de la etapa
    cv2.putText(img_recognition, "STAGE 2: RECOGNITION", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Proyecciones (azul claro)
    for x1, y1, x2, y2, pid in bboxes:
        cv2.rectangle(img_recognition, (x1, y1), (x2, y2), (200, 100, 0), 1)
        cv2.putText(img_recognition, f"P{pid}", (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 0), 1)
    
    # Detecciones con colores predichos
    for det_idx, det in enumerate(valid):
        x1, y1, x2, y2 = map(int, det[1:5])
        rec_scores = recognitions[det_idx].tolist() if det_idx < len(recognitions) else [0,0,0,0]
        pred_cls = int(torch.argmax(recognitions[det_idx])) if det_idx < len(recognitions) else 0
        pred_color = ['black','red','yellow','green'][pred_cls]
        max_conf = max(rec_scores)
        
        # Color del rectángulo según predicción
        if pred_color == 'red':
            box_color = (0, 0, 255)
        elif pred_color == 'green':
            box_color = (0, 255, 0)
        elif pred_color == 'yellow':
            box_color = (0, 255, 255)
        else:
            box_color = (128, 128, 128)
        
        cv2.rectangle(img_recognition, (x1, y1), (x2, y2), box_color, 3)
        info_text = f"{pred_color.upper()} ({max_conf:.2f})"
        cv2.putText(img_recognition, info_text, (x1, y1-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    # Guardar imagen de reconocimiento en carpeta específica
    out_img2 = os.path.join(recognition_dir, frame_name.replace('.jpg','_recognition.jpg'))
    cv2.imwrite(out_img2, img_recognition)
    
    # ────── IMAGEN 3: ETAPA FINAL (TRACKING) ──────
    img_final = image_np.copy()
    
    # Título de la etapa
    cv2.putText(img_final, "STAGE 3: FINAL (TRACKING)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Proyecciones con estado revisado
    for x1, y1, x2, y2, pid in bboxes:
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (100, 100, 255), 1)
        
        # Mostrar estado revisado para esta proyección
        rev_color, blink = revised.get(pid, ('none', False))
        proj_text = f"P{pid}: {rev_color.upper()}"
        if blink:
            proj_text += " *BLINK*"
        cv2.putText(img_final, proj_text, (x1, y2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)
    
    # Detecciones con asignación y tracking
    for det_idx, det in enumerate(valid):
        x1, y1, x2, y2 = map(int, det[1:5])
        pred_cls = int(torch.argmax(recognitions[det_idx])) if det_idx < len(recognitions) else 0
        pred_color = ['black','red','yellow','green'][pred_cls]
        
        # Obtener resultado de tracking
        proj_id = assign_map.get(det_idx, -1)
        rev_color, blink = revised.get(proj_id, (pred_color, False)) if proj_id != -1 else (pred_color, False)
        
        # Color del rectángulo según color revisado
        if rev_color == 'red':
            box_color = (0, 0, 255)
        elif rev_color == 'green':
            box_color = (0, 255, 0)
        elif rev_color == 'yellow':
            box_color = (0, 255, 255)
        else:
            box_color = (128, 128, 128)
        
        # Rectángulo más grueso si hay cambio en tracking
        thickness = 4 if pred_color != rev_color else 3
        cv2.rectangle(img_final, (x1, y1), (x2, y2), box_color, thickness)
        
        # Información completa
        info_text = f"Det{det_idx}>P{proj_id}: {pred_color}>{rev_color}"
        if blink:
            info_text += " *BLINK*"
        cv2.putText(img_final, info_text, (x1, y1-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    # Guardar imagen final en carpeta específica
    out_img3 = os.path.join(final_dir, frame_name.replace('.jpg','_final.jpg'))
    cv2.imwrite(out_img3, img_final)
    
    print(f"✅ {frame_name}: {len(valid)} valid, {len(invalid)} invalid, {len(bboxes)} projections → imágenes guardadas")

print(f"\n📄 RESULTADOS ORGANIZADOS POR ETAPAS:")
print(f"📊 CSV Files:")
print(f"   📋 0_all_detections.csv: TODAS las detecciones (válidas + inválidas)")
print(f"   🔍 1_detection_results.csv: Solo detecciones válidas del detector")
print(f"   🎨 2_recognition_results.csv: Después de clasificación de color") 
print(f"   🏁 3_final_results.csv: Después de asignación y tracking")
print(f"\n🖼️  Image Folders:")
print(f"   📁 1_detection/: {len(entries)} imágenes mostrando detecciones")
print(f"   📁 2_recognition/: {len(entries)} imágenes mostrando clasificación de colores")
print(f"   📁 3_final/: {len(entries)} imágenes mostrando tracking final")
print(f"\n📂 Todo en: {output_dir}/")