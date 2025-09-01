import cv2
import os
import sys
import torch
import numpy as np

# 🔧 Fix para importar tlr.pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from tlr.pipeline import load_pipeline

def expand_bbox(bbox, expansion_factor=1.2, img_shape=None):
    """
    Expande una bounding box por un factor dado
    """
    x1, y1, x2, y2, proj_id = bbox
    
    # Calcular centro y dimensiones
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    
    # Expandir
    new_width = width * expansion_factor
    new_height = height * expansion_factor
    
    # Nuevas coordenadas
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)
    
    # Limitar a los bordes de la imagen si se proporciona img_shape
    if img_shape is not None:
        h, w = img_shape
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(w, new_x2)
        new_y2 = min(h, new_y2)
    
    return [new_x1, new_y1, new_x2, new_y2, proj_id]

def bbox_from_detection(detection):
    """
    Convierte una detección del pipeline a formato bbox
    """
    proj_id, x1, y1, x2, y2 = detection[:5].int().tolist()
    return [x1, y1, x2, y2, proj_id]

def main():
    # ─── CONFIG ───────────────────────────────────────────────────────────────────
    # Usar frames conocidos que funcionan bien
    input_dir = '../../frames_auto_labeled'
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    print("🧪 TEST: Retroalimentación adaptiva con frames CONOCIDOS")
    print(f"📂 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Leer SOLO las proyecciones del primer frame del archivo conocido
    initial_projections = []
    projections_file = 'projection_bboxes_master.txt'
    first_frame_name = None
    
    with open(projections_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    frame_name, x1, y1, x2, y2, proj_id = parts[:6]
                    
                    # Si es la primera línea, guardar el nombre del frame
                    if first_frame_name is None:
                        first_frame_name = frame_name
                    
                    # Solo agregar proyecciones del primer frame
                    if frame_name == first_frame_name:
                        initial_projections.append([int(x1), int(y1), int(x2), int(y2), int(proj_id)])
                    else:
                        # Ya pasamos al segundo frame, salir del loop
                        break
    
    print(f"📍 Proyecciones iniciales del {first_frame_name}: {len(initial_projections)} regiones")
    for i, proj in enumerate(initial_projections):
        print(f"   Región {i}: [{proj[0]}, {proj[1]}, {proj[2]}, {proj[3]}] ID:{proj[4]}")
    
    expansion_factor = 1.2  # 20% más grande para el siguiente frame
    max_frames = 20  # Procesar solo los primeros 20 frames para comparar
    
    # ─── CARGA DEL PIPELINE ─────────────────────────────────────────────────────────
    print("🚀 Cargando pipeline...")
    pipeline = load_pipeline('cuda:0')
    print("✅ Pipeline cargado")
    
    # ─── PROCESAR FRAMES ───────────────────────────────────────────────────────────
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])[:max_frames]
    print(f"📹 Procesando {len(frame_files)} frames con retroalimentación adaptiva")
    print("🔍 Comparando con el método de proyecciones fijas")
    
    current_projections = initial_projections.copy()
    results_log = []
    regions_log = []  # Para guardar las regiones usadas en cada frame
    
    for frame_idx, frame_file in enumerate(frame_files):
        print(f"\\n🔍 FRAME {frame_idx}: {frame_file}")
        
        # Cargar imagen
        frame_path = os.path.join(input_dir, frame_file)
        image_np = cv2.imread(frame_path)
        image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
        h, w = image_np.shape[:2]
        
        print(f"   📊 Proyecciones actuales: {len(current_projections)}")
        for i, proj in enumerate(current_projections):
            print(f"      Proj {i}: [{proj[0]}, {proj[1]}, {proj[2]}, {proj[3]}] ID:{proj[4]}")
        
        # Guardar regiones usadas en este frame
        for proj_idx, proj in enumerate(current_projections):
            regions_log.append(f"{frame_file},{proj_idx},{proj[0]},{proj[1]},{proj[2]},{proj[3]},{proj[4]}")
        
        # Procesar con pipeline
        valid_detections, recognitions, assignments, invalid_detections, revised_states = pipeline(
            image_tensor, current_projections, frame_ts=frame_idx * (1.0/29)
        )
        
        print(f"   ✅ Detecciones válidas: {len(valid_detections)}")
        print(f"   ❌ Detecciones inválidas: {len(invalid_detections)}")
        
        # Visualización detallada de cada detección
        for det_idx, det in enumerate(valid_detections):
            x1, y1, x2, y2 = map(int, det[1:5])
            tl_type = int(torch.argmax(det[5:9]))
            type_names = ['vert', 'quad', 'hori', 'bg']
            det_scores = det[5:9].tolist()
            
            if det_idx < len(recognitions):
                pred_cls = int(torch.argmax(recognitions[det_idx]))
                pred_color = ['black','red','yellow','green'][pred_cls]
            else:
                pred_color = 'unknown'
            
            print(f"      Det {det_idx}: [{x1},{y1},{x2},{y2}] {type_names[tl_type]} {pred_color}")
            print(f"                Scores: vert={det_scores[0]:.3f} quad={det_scores[1]:.3f} hori={det_scores[2]:.3f} bg={det_scores[3]:.3f}")
        
        # PROBLEMA IDENTIFICADO: ¿Qué hace con múltiples detecciones?
        if len(valid_detections) > len(current_projections):
            print(f"   ⚠️  PROBLEMA: {len(valid_detections)} detecciones vs {len(current_projections)} proyecciones")
            print(f"   🤔 El sistema está creando nuevas proyecciones de más")
        
        # RETROALIMENTACIÓN: Actualizar proyecciones para el próximo frame
        next_projections = []
        
        if len(valid_detections) > 0:
            for det_idx, det in enumerate(valid_detections):
                # Convertir detección a bbox
                bbox = bbox_from_detection(det)
                
                # Expandir para el próximo frame
                expanded_bbox = expand_bbox(bbox, expansion_factor, (h, w))
                next_projections.append(expanded_bbox)
                
                # Log info
                x1, y1, x2, y2, proj_id = bbox
                results_log.append(f"{frame_file},{det_idx},{x1},{y1},{x2},{y2},{type_names[tl_type]},{pred_color}")
        
        # Actualizar proyecciones para el próximo frame
        current_projections = next_projections if next_projections else initial_projections.copy()
        
        # Guardar imagen con visualización
        img_copy = image_np.copy()
        
        # Dibujar proyecciones usadas (azul)
        for proj in (initial_projections if frame_idx == 0 else current_projections):
            cv2.rectangle(img_copy, (proj[0], proj[1]), (proj[2], proj[3]), (255, 0, 0), 2)
            cv2.putText(img_copy, f'Proj-{proj[4]}', (proj[0], proj[1]-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Dibujar detecciones válidas (verde)
        if len(valid_detections) > 0:
            for det_idx, det in enumerate(valid_detections):
                x1, y1, x2, y2 = map(int, det[1:5])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Color detectado
                if det_idx < len(recognitions):
                    pred_cls = int(torch.argmax(recognitions[det_idx]))
                    color_name = ['black','red','yellow','green'][pred_cls]
                    cv2.putText(img_copy, f'Det{det_idx}: {color_name}', (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Dibujar información del frame
        cv2.putText(img_copy, f'Frame {frame_idx}: {len(current_projections)} proj -> {len(valid_detections)} det', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Guardar
        output_path = os.path.join(output_dir, f'adaptive_{frame_file}')
        cv2.imwrite(output_path, img_copy)
    
    # ─── GUARDAR RESULTADOS ─────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, 'adaptive_results.csv'), 'w') as f:
        f.write('frame,det_idx,x1,y1,x2,y2,tl_type,pred_color\\n')
        for line in results_log:
            f.write(line + '\\n')
    
    # Guardar regiones usadas en cada frame
    with open(os.path.join(output_dir, 'regions_used.csv'), 'w') as f:
        f.write('frame,proj_idx,x1,y1,x2,y2,proj_id\\n')
        for line in regions_log:
            f.write(line + '\\n')
    
    print(f"\\n🎯 RESULTADOS:")
    print(f"   📊 Total frames procesados: {len(frame_files)}")
    print(f"   📁 Imágenes guardadas: {output_dir}")
    print(f"   📄 Detecciones: {output_dir}/adaptive_results.csv")
    print(f"   📍 Regiones usadas: {output_dir}/regions_used.csv")
    print("\\n🔍 COMPARAR CON:")
    print("   📊 frames_auto_labeled/outputs_debug_stages/ (método fijo)")
    print("   🆚 Verificar si hay diferencias en cantidad de detecciones")
    print("   📈 Analizar evolución de regiones en regions_used.csv")

if __name__ == "__main__":
    main()