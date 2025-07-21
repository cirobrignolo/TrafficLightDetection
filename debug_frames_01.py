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

# ─── CARGA DEL PIPELINE ─────────────────────────────────────────────────────────
pipeline = load_pipeline('cuda:0')
print("🚀 Pipeline cargado para debug de frames 0 y 1")

# ─── LEER PROYECCIONES ───────────────────────────────────────────────────────────
entries = {}
with open(bbox_file, 'r') as f:
    for line in f:
        parts = line.strip().split(",")
        frame = parts[0]
        bbox = list(map(int, parts[1:]))
        entries.setdefault(frame, []).append(bbox)

print(f"📊 Total frames en bbox file: {len(entries)}")
print(f"🔍 Primeros 5 frames: {list(entries.keys())[:5]}")

# ─── DEBUG ESPECÍFICO DE FRAMES 0 Y 1 ──────────────────────────────────────────
target_frames = ['frame_0000.jpg', 'frame_0001.jpg', 'frame_0002.jpg']

for frame_name in target_frames:
    print(f"\n🔍 ═══ DEBUGGING {frame_name} ═══")
    
    if frame_name not in entries:
        print(f"❌ {frame_name} NO está en projection_bboxes_master.txt")
        continue
    
    bboxes = entries[frame_name]
    print(f"📦 Projection boxes: {bboxes}")
    
    frame_path = os.path.join(input_dir, frame_name)
    image_np = cv2.imread(frame_path)
    if image_np is None:
        print(f"❌ No se pudo cargar la imagen: {frame_path}")
        continue
    
    print(f"🖼️  Imagen cargada: {image_np.shape}")
    
    # Convertir a tensor y procesar
    image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
    frame_ts = time.time()
    
    # Ejecutar pipeline y capturar TODOS los resultados
    valid, recognitions, assignments, invalid, revised = pipeline(
        image_tensor, bboxes, frame_ts
    )
    
    print(f"📊 RESULTADOS DEL PIPELINE:")
    print(f"   📍 Input bboxes: {len(bboxes)}")
    print(f"   ✅ Valid detections: {len(valid)}")
    print(f"   ❌ Invalid detections: {len(invalid)}")  
    print(f"   🔗 Assignments: {len(assignments)}")
    print(f"   🕒 Revised states: {len(revised)}")
    
    if len(valid) > 0:
        print(f"   🎯 Valid detection boxes:")
        for i, det in enumerate(valid):
            x1, y1, x2, y2 = map(int, det[1:5])
            scores = det[5:9]
            print(f"      Det {i}: [{x1},{y1},{x2},{y2}] scores={scores.tolist()}")
    else:
        print(f"   🚫 NO HAY DETECCIONES VÁLIDAS")
    
    if len(invalid) > 0:
        print(f"   ❌ Invalid detection boxes:")
        for i, det in enumerate(invalid):
            x1, y1, x2, y2 = map(int, det[1:5])  
            scores = det[5:9]
            print(f"      Invalid {i}: [{x1},{y1},{x2},{y2}] scores={scores.tolist()}")
    
    if len(assignments) > 0:
        assign_list = assignments.cpu().tolist()
        print(f"   🔗 Assignments: {assign_list}")
    
    # Crear imagen debug para visualizar
    debug_img = image_np.copy()
    
    # Dibujar projection boxes (azul)
    for x1, y1, x2, y2, pid in bboxes:
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(debug_img, f"P{pid}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Dibujar valid detections (verde)  
    for i, det in enumerate(valid):
        x1, y1, x2, y2 = map(int, det[1:5])
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_img, f"V{i}", (x1, y2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Dibujar invalid detections (rojo)
    for i, det in enumerate(invalid):
        x1, y1, x2, y2 = map(int, det[1:5])
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(debug_img, f"X{i}", (x1, y2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    # Guardar imagen debug
    debug_path = f"debug_{frame_name}"
    cv2.imwrite(debug_path, debug_img)
    print(f"💾 Debug image saved: {debug_path}")
    
    print(f"═══ END {frame_name} ═══")