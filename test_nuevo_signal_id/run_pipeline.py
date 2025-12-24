import cv2
import os
import sys
import torch
import numpy as np

# 🔧 Fix para importar tlr.pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from tlr.pipeline import load_pipeline

# ─── CONFIGURACIÓN SIMPLIFICADA ────────────────────────────────────────────────
input_dir = 'input_frames'
bbox_file = 'projection_bboxes_master.txt'
output_dir = 'output'

# Crear carpeta de output
os.makedirs(output_dir, exist_ok=True)

# Crear subdirectorios para las etapas
os.makedirs(os.path.join(output_dir, '1_detection'), exist_ok=True)
os.makedirs(os.path.join(output_dir, '2_recognition'), exist_ok=True)
os.makedirs(os.path.join(output_dir, '3_final'), exist_ok=True)

print(f"🚀 PIPELINE DE DETECCIÓN DE SEMÁFOROS")
print(f"📂 Input: {input_dir}/")
print(f"📄 Projection boxes: {bbox_file}")
print(f"📁 Output: {output_dir}/")

# ─── CARGA DEL PIPELINE (CON TRACKING) ─────────────────────────────────────────
pipeline = load_pipeline('cuda:0')  # o 'cpu'
print("🚀 Pipeline cargado con tracking y debug de etapas separadas")

# ─── LEER PROYECCIONES POR FRAME ───────────────────────────────────────────────
print(f"📋 Leyendo projection boxes desde: {bbox_file}")

entries = {}
with open(bbox_file, 'r') as f:
    for line in f:
        parts = line.strip().split(",")
        frame = parts[0]
        bbox = list(map(int, parts[1:]))
        entries.setdefault(frame, []).append(bbox)

print(f"📊 Frames con proyecciones: {len(entries)}")

# ─── PROCESAR SOLO LOS PRIMEROS 20 FRAMES ──────────────────────────────────────
max_frames = 400
frame_count = 0

# ─── LOGS DE RESULTADOS ────────────────────────────────────────────────────────
all_detection_lines = []
detection_lines = []
recognition_lines = []
final_lines = []

for frame_idx, (frame_name, bboxes) in enumerate(entries.items()):
    if frame_count >= max_frames:
        break
        
    print(f"\n🔍 FRAME {frame_count}: {frame_name}")
    
    # Leer imagen
    frame_path = os.path.join(input_dir, frame_name)
    if not os.path.exists(frame_path):
        print(f"❌ No se encontró: {frame_path}")
        continue
        
    image_np = cv2.imread(frame_path)
    image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
    
    print(f"   📊 Input projections: {len(bboxes)}")
    for i, bbox in enumerate(bboxes):
        print(f"      Proj {i}: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}] ID:{bbox[4]}")
    
    # Ejecutar pipeline
    valid_detections, recognitions, assignments, invalid_detections, revised_states = pipeline(
        image_tensor, bboxes, frame_ts=frame_count * (1.0/29)
    )
    
    print(f"   ✅ Valid detections: {len(valid_detections)}")
    print(f"   ❌ Invalid detections: {len(invalid_detections)}")
    print(f"   🔗 Assignments: {len(assignments)}")
    print(f"   🔄 Revised states: {len(revised_states) if revised_states else 0}")
    
    # ─── LOGS DE TODAS LAS DETECCIONES ──────────────────────────────────────────
    for det_idx, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        # FIXED: Correct order matching Apollo detector output [bg, vert, quad, hori]
        type_names = ['bg', 'vert', 'quad', 'hori']

        all_detection_lines.append(
            f"{frame_name},VALID,{det_idx},{x1},{y1},{x2},{y2},"
            f"{type_names[tl_type]},{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f}"
        )

    for det_idx, det in enumerate(invalid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        # FIXED: Correct order matching Apollo detector output [bg, vert, quad, hori]
        type_names = ['bg', 'vert', 'quad', 'hori']

        all_detection_lines.append(
            f"{frame_name},INVALID,{det_idx},{x1},{y1},{x2},{y2},"
            f"{type_names[tl_type]},{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f}"
        )
    
    # ─── ETAPA 1: DETECCIÓN ─────────────────────────────────────────────────────
    img_detection = image_np.copy()
    
    for det_idx, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()  # [bg, vert, quad, hori]
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']

        detection_lines.append(
            f"{frame_name},{det_idx},{x1},{y1},{x2},{y2},"
            f"{type_names[tl_type]},{det_scores[0]:.4f},{det_scores[1]:.4f},{det_scores[2]:.4f},{det_scores[3]:.4f}"
        )
    
    # Visualizar detecciones
    for det_idx, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']

        # Color verde para detecciones
        color = (0, 255, 0)

        cv2.rectangle(img_detection, (x1, y1), (x2, y2), color, 3)
        info_text = f"Det{det_idx}: {type_names[tl_type]} ({det_scores[tl_type]:.2f})"
        cv2.putText(img_detection, info_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Dibujar projection boxes como referencia
    for bbox in bboxes:
        x1, y1, x2, y2, proj_id = bbox
        cv2.rectangle(img_detection, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Azul para proyecciones
        cv2.putText(img_detection, f'Proj{proj_id}', (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Agregar título
    cv2.putText(img_detection, "DETECCION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Guardar imagen de detección
    cv2.imwrite(os.path.join(output_dir, '1_detection', f'{frame_name}_detection.jpg'), img_detection)
    
    # ─── ETAPA 2: RECONOCIMIENTO ────────────────────────────────────────────────
    img_recognition = image_np.copy()
    
    for det_idx, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        pred_cls = int(torch.argmax(recognitions[det_idx])) if det_idx < len(recognitions) else 0
        pred_color = ['black','red','yellow','green'][pred_cls]
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']

        recognition_lines.append(
            f"{frame_name},{det_idx},{x1},{y1},{x2},{y2},"
            f"{type_names[tl_type]},{pred_color}"
        )
        
        # Visualizar reconocimiento con colores reales
        color_map = {
            'black': (0, 0, 0), 
            'red': (0, 0, 255), 
            'yellow': (0, 255, 255), 
            'green': (0, 255, 0)
        }
        color = color_map.get(pred_color, (255, 255, 255))
        cv2.rectangle(img_recognition, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img_recognition, pred_color.upper(), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Agregar título
    cv2.putText(img_recognition, "RECONOCIMIENTO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Guardar imagen de reconocimiento
    cv2.imwrite(os.path.join(output_dir, '2_recognition', f'{frame_name}_recognition.jpg'), img_recognition)
    
    # ─── ETAPA 3: TRACKING FINAL ────────────────────────────────────────────────
    img_final = image_np.copy()
    # FIXED: assignments = [[proj_idx, det_idx], ...] → necesitamos det_idx → proj_idx
    # Crear mapa INVERSO: det_idx → proj_idx
    assign_map = {int(assignment[1]): int(assignment[0]) for assignment in assignments}

    for det_idx, det in enumerate(valid_detections):
        x1, y1, x2, y2 = map(int, det[1:5])
        pred_cls = int(torch.argmax(recognitions[det_idx])) if det_idx < len(recognitions) else 0
        pred_color = ['black','red','yellow','green'][pred_cls]
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']

        # Obtener proj_idx desde det_idx (ahora el mapa es correcto)
        proj_idx = assign_map.get(det_idx, -1)

        # NUEVO: Convertir proj_idx → signal_id para buscar en revised_states
        signal_id = f"signal_{bboxes[proj_idx][4]}" if proj_idx >= 0 and proj_idx < len(bboxes) else "unknown"

        # Obtener estado revisado si existe
        final_color = pred_color
        blink_status = ""
        is_blinking = False  # Inicializar la variable
        if revised_states and signal_id in revised_states:
            revised_color, is_blinking = revised_states[signal_id]
            final_color = revised_color
            blink_status = " (BLINK)" if is_blinking else ""

        final_lines.append(
            f"{frame_name},{det_idx},{x1},{y1},{x2},{y2},"
            f"{type_names[tl_type]},{pred_color},{final_color},{signal_id},{blink_status.strip()}"
        )
        
        # Visualizar resultado final con colores apropiados
        color_map = {
            'black': (0, 0, 0), 
            'red': (0, 0, 255), 
            'yellow': (0, 255, 255), 
            'green': (0, 255, 0),
            'BLACK': (0, 0, 0), 
            'RED': (0, 0, 255), 
            'YELLOW': (0, 255, 255), 
            'GREEN': (0, 255, 0)
        }
        color = color_map.get(final_color, (255, 255, 255))
        
        # Usar color especial si está parpadeando
        if is_blinking:
            color = (255, 0, 255)  # Magenta para parpadeo
        
        cv2.rectangle(img_final, (x1, y1), (x2, y2), color, 3)
        final_text = f"{signal_id} {final_color}{blink_status}"
        cv2.putText(img_final, final_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Agregar título final
    cv2.putText(img_final, "TRACKING FINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Guardar imagen final
    cv2.imwrite(os.path.join(output_dir, '3_final', f'{frame_name}_final.jpg'), img_final)
    
    frame_count += 1
    print(f"✅ {frame_name}: {len(valid_detections)} valid, {len(invalid_detections)} invalid, {len(bboxes)} projections → imágenes guardadas")

# ─── GUARDAR RESULTADOS CSV ─────────────────────────────────────────────────────
# Todas las detecciones
with open(os.path.join(output_dir, '0_all_detections.csv'), 'w') as f:
    # FIXED: Correct header order matching Apollo detector output [bg, vert, quad, hori]
    f.write('frame,status,det_idx,x1,y1,x2,y2,tl_type,det_bg,det_vert,det_quad,det_hori\n')
    for line in all_detection_lines:
        f.write(line + '\n')

# Solo detecciones válidas
with open(os.path.join(output_dir, '1_detection_results.csv'), 'w') as f:
    # FIXED: Correct header order matching Apollo detector output [bg, vert, quad, hori]
    f.write('frame,det_idx,x1,y1,x2,y2,tl_type,det_bg,det_vert,det_quad,det_hori\n')
    for line in detection_lines:
        f.write(line + '\n')

# Reconocimiento
with open(os.path.join(output_dir, '2_recognition_results.csv'), 'w') as f:
    f.write('frame,det_idx,x1,y1,x2,y2,tl_type,pred_color\n')
    for line in recognition_lines:
        f.write(line + '\n')

# Final con tracking
with open(os.path.join(output_dir, '3_final_results.csv'), 'w') as f:
    f.write('frame,det_idx,x1,y1,x2,y2,tl_type,pred_color,final_color,signal_id,blink_status\n')
    for line in final_lines:
        f.write(line + '\n')

print(f"\n📄 RESULTADOS ORGANIZADOS POR ETAPAS:")
print(f"📊 CSV Files:")
print(f"   📋 0_all_detections.csv: TODAS las detecciones (válidas + inválidas)")
print(f"   🔍 1_detection_results.csv: Solo detecciones válidas del detector")
print(f"   🎨 2_recognition_results.csv: Después de clasificación de color")
print(f"   🏁 3_final_results.csv: Después de asignación y tracking (con signal_id)")
print(f"")
print(f"🖼️  Image Folders:")
print(f"   📁 1_detection/: {frame_count} imágenes mostrando detecciones")
print(f"   📁 2_recognition/: {frame_count} imágenes mostrando clasificación de colores")
print(f"   📁 3_final/: {frame_count} imágenes mostrando tracking final")
print(f"")
print(f"📂 Todo en: {output_dir}/")