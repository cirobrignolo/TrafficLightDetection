import cv2
import numpy as np
import os
import random
import shutil
import subprocess
from pathlib import Path

def apply_sepia(image):
    """Aplica efecto sepia (luz cálida/atardecer)"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia_img = cv2.transform(image, kernel)
    return np.clip(sepia_img, 0, 255).astype(np.uint8)

def apply_blue_tint(image, intensity=0.4):
    """Aplica tinte azul (simula noche/sombra)"""
    blue_tinted = image.copy().astype(np.float32)
    blue_tinted[:,:,0] = blue_tinted[:,:,0] * (1 + intensity)  # Aumentar canal azul
    blue_tinted[:,:,1] = blue_tinted[:,:,1] * (1 - intensity*0.3)  # Reducir verde
    blue_tinted[:,:,2] = blue_tinted[:,:,2] * (1 - intensity*0.5)  # Reducir rojo
    return np.clip(blue_tinted, 0, 255).astype(np.uint8)

def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Ajusta brillo y contraste"""
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

def add_fog_effect(image, intensity=0.3):
    """Simula efecto niebla"""
    fog = np.ones_like(image, dtype=np.float32) * 255 * intensity
    fog = cv2.GaussianBlur(fog, (15, 15), 0)
    foggy = cv2.addWeighted(image.astype(np.float32), 1-intensity, fog, intensity, 0)
    return np.clip(foggy, 0, 255).astype(np.uint8)

def add_rain_effect(image, intensity=0.3):
    """Simula lluvia con líneas verticales"""
    rain_image = image.copy()
    height, width = image.shape[:2]
    
    num_drops = int(width * height * intensity / 1000)
    
    for _ in range(num_drops):
        x = random.randint(0, width-1)
        y1 = random.randint(0, height//2)
        y2 = y1 + random.randint(10, 30)
        
        cv2.line(rain_image, (x, y1), (x, min(y2, height-1)), 
                (200, 200, 200), 1, cv2.LINE_AA)
    
    return rain_image

def add_gaussian_noise(image, intensity=20):
    """Añade ruido gaussiano"""
    noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def compress_jpeg(image, quality=25):
    """Simula compresión JPEG agresiva"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed = cv2.imdecode(encimg, 1)
    return compressed

# Diccionario de efectos para testing
TEST_EFFECTS = {
    'original': lambda img: img,  # Sin modificación
    'sepia': apply_sepia,
    'blue_night': apply_blue_tint,
    'dark': lambda img: adjust_brightness_contrast(img, brightness=-40, contrast=0.8),
    'bright': lambda img: adjust_brightness_contrast(img, brightness=25, contrast=1.1),
    'low_contrast': lambda img: adjust_brightness_contrast(img, brightness=0, contrast=0.6),
    'fog_light': lambda img: add_fog_effect(img, 0.25),
    'rain_light': lambda img: add_rain_effect(img, 0.25),
    'noise_light': lambda img: add_gaussian_noise(img, 15),
    'jpeg_compression': lambda img: compress_jpeg(img, 30)
}

def create_test_case(source_dir, test_name, effect_function, base_output_dir):
    """
    Crea un caso de prueba completo:
    1. Crea estructura de carpetas
    2. Aplica efecto a imágenes
    3. Copia projection_bboxes_master.txt
    4. Crea script personalizado para este caso
    """
    
    print(f"\n🎨 Creando caso de prueba: {test_name}")
    
    # Crear estructura de carpetas
    test_dir = Path(base_output_dir) / test_name
    input_dir = test_dir / "input"
    output_dir = test_dir / "output"
    
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    
    # Procesar imágenes
    image_files = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
    
    print(f"   📸 Procesando {len(image_files)} imágenes...")
    
    for img_file in image_files:
        # Leer imagen original
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # Aplicar efecto
        try:
            modified_image = effect_function(image)
            
            # Guardar en input del caso de prueba
            output_file = input_dir / img_file.name
            cv2.imwrite(str(output_file), modified_image)
            
        except Exception as e:
            print(f"     ❌ Error procesando {img_file.name}: {e}")
    
    # Copiar projection_bboxes_master.txt
    bbox_source = source_path / 'projection_bboxes_master.txt'
    bbox_dest = input_dir / 'projection_bboxes_master.txt'
    
    if bbox_source.exists():
        shutil.copy2(bbox_source, bbox_dest)
        print(f"   📋 Copiado projection_bboxes_master.txt")
    else:
        print(f"   ⚠️  No encontrado projection_bboxes_master.txt en {source_dir}")
    
    # Crear script personalizado para este caso
    create_custom_pipeline_script(test_name, input_dir, output_dir, test_dir)
    
    print(f"   ✅ Caso {test_name} creado en: {test_dir}")
    
    return test_dir

def create_custom_pipeline_script(test_name, input_dir, output_dir, test_dir):
    """
    Crea un script personalizado del pipeline para este caso específico
    """
    
    script_content = f'''import cv2
import os
import sys
import time
import torch
import numpy as np

# 🔧 Fix para importar tlr.pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from tlr.pipeline import load_pipeline

# ─── CONFIG PARA CASO: {test_name.upper()} ─────────────────────────────────
test_name = "{test_name}"
input_dir = r'{input_dir.absolute()}'
bbox_file = os.path.join(input_dir, 'projection_bboxes_master.txt')
output_dir = r'{output_dir.absolute()}'
os.makedirs(output_dir, exist_ok=True)

print(f"🧪 EJECUTANDO CASO DE PRUEBA: {{test_name.upper()}}")
print(f"📥 Input: {{input_dir}}")
print(f"📤 Output: {{output_dir}}")

# ─── CARGA DEL PIPELINE (CON TRACKING) ─────────────────────────────────────────
pipeline = load_pipeline('cuda:0')  # o 'cpu'
print("🚀 Pipeline cargado con tracking y debug de etapas separadas")

# ─── LEER PROYECCIONES POR FRAME ───────────────────────────────────────────────
entries = {{}}
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
all_detections_csv = os.path.join(output_dir, '0_all_detections.csv')

# Clean start: remove existing files
for csv_file in [detection_csv, recognition_csv, final_csv, all_detections_csv]:
    if os.path.exists(csv_file):
        os.remove(csv_file)

# Write headers for each file
with open(detection_csv, 'w') as out:
    out.write("frame,det_idx,x1,y1,x2,y2,tl_type,det_vert,det_quad,det_hori,det_bg\\n")

with open(recognition_csv, 'w') as out:
    out.write("frame,det_idx,x1,y1,x2,y2,tl_type,pred_color,rec_black,rec_red,rec_yellow,rec_green\\n")

with open(final_csv, 'w') as out:
    out.write("frame,det_idx,proj_id,x1,y1,x2,y2,tl_type,pred_color,revised_color,blink,det_vert,det_quad,det_hori,det_bg,rec_black,rec_red,rec_yellow,rec_green\\n")

with open(all_detections_csv, 'w') as out:
    out.write("frame,status,det_idx,x1,y1,x2,y2,tl_type,det_vert,det_quad,det_hori,det_bg\\n")

# ─── PROCESS EACH FRAME ──────────────────────────────────────────────────────
for frame_idx, (frame_name, bboxes) in enumerate(entries.items()):
    frame_path = os.path.join(input_dir, frame_name)
    image_np = cv2.imread(frame_path)
    if image_np is None:
        print(f"❌ No se pudo cargar {{frame_name}}, salto.")
        continue

    # Convertir a tensor y pasar a GPU
    image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
    frame_ts = time.time()

    # Ejecutar pipeline y capturar resultados intermedios
    valid, recognitions, assignments, invalid, revised = pipeline(
        image_tensor, bboxes, frame_ts
    )

    # 📊 ANÁLISIS ETAPA POR ETAPA
    print(f"\\n🔍 FRAME: {{frame_name}}")
    print(f"   Input projections: {{len(bboxes)}}")
    print(f"   Valid detections: {{len(valid)}}")
    print(f"   Invalid detections: {{len(invalid)}}")
    print(f"   Assignments: {{len(assignments)}}")
    print(f"   Revised states: {{len(revised)}}")
    
    # ═══ ALL DETECTIONS (VALID + INVALID) ═══
    all_detection_lines = []
    
    # Add valid detections
    for det_idx, det in enumerate(valid):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']
        
        all_detection_lines.append(
            f"{{frame_name}},VALID,{{det_idx}},{{x1}},{{y1}},{{x2}},{{y2}},"
            f"{{type_names[tl_type]}},"
            f"{{det_scores[0]:.4f}},{{det_scores[1]:.4f}},{{det_scores[2]:.4f}},{{det_scores[3]:.4f}}\\n"
        )
    
    # Add invalid detections
    for det_idx, det in enumerate(invalid):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']
        
        all_detection_lines.append(
            f"{{frame_name}},INVALID,{{det_idx}},{{x1}},{{y1}},{{x2}},{{y2}},"
            f"{{type_names[tl_type]}},"
            f"{{det_scores[0]:.4f}},{{det_scores[1]:.4f}},{{det_scores[2]:.4f}},{{det_scores[3]:.4f}}\\n"
        )
    
    if all_detection_lines:
        with open(all_detections_csv, 'a') as out:
            out.writelines(all_detection_lines)
    
    # Construir mapas de asignación
    assign_list = assignments.cpu().tolist()
    assign_map = {{det_idx: proj_id for proj_id, det_idx in assign_list}}
    
    # ═══ ETAPA 1: DETECCIÓN CRUDA ═══
    detection_lines = []
    for det_idx, det in enumerate(valid):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()  # [vert, quad, hori, bg]
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']
        
        detection_lines.append(
            f"{{frame_name}},{{det_idx}},{{x1}},{{y1}},{{x2}},{{y2}},"
            f"{{type_names[tl_type]}},"
            f"{{det_scores[0]:.4f}},{{det_scores[1]:.4f}},{{det_scores[2]:.4f}},{{det_scores[3]:.4f}}\\n"
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
            f"{{frame_name}},{{det_idx}},{{x1}},{{y1}},{{x2}},{{y2}},"
            f"{{type_names[tl_type]}},{{pred_color}},"
            f"{{rec_scores[0]:.4f}},{{rec_scores[1]:.4f}},{{rec_scores[2]:.4f}},{{rec_scores[3]:.4f}}\\n"
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
            f"{{frame_name}},{{det_idx}},{{proj_id}},{{x1}},{{y1}},{{x2}},{{y2}},"
            f"{{type_names[tl_type]}},{{pred_color}},{{rev_color}},{{int(blink)}},"
            f"{{det_scores[0]:.4f}},{{det_scores[1]:.4f}},{{det_scores[2]:.4f}},{{det_scores[3]:.4f}},"
            f"{{rec_scores[0]:.4f}},{{rec_scores[1]:.4f}},{{rec_scores[2]:.4f}},{{rec_scores[3]:.4f}}\\n"
        )
    
    if final_lines:
        with open(final_csv, 'a') as out:
            out.writelines(final_lines)

    # ═══ VISUALIZACIÓN POR ETAPAS ═══
    
    # ────── IMAGEN 1: ETAPA DETECCIÓN ──────
    img_detection = image_np.copy()
    
    # Título de la etapa
    cv2.putText(img_detection, f"STAGE 1: DETECTION - {{test_name.upper()}}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Proyecciones originales (azul)
    for x1, y1, x2, y2, pid in bboxes:
        cv2.rectangle(img_detection, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_detection, f"Proj{{pid}}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Detecciones válidas con scores de orientación
    for det_idx, det in enumerate(valid):
        x1, y1, x2, y2 = map(int, det[1:5])
        det_scores = det[5:9].tolist()
        tl_type = int(torch.argmax(det[5:9]))
        type_names = ['bg', 'vert', 'quad', 'hori']
        
        cv2.rectangle(img_detection, (x1, y1), (x2, y2), (0, 255, 0), 3)
        info_text = f"Det{{det_idx}}: {{type_names[tl_type]}} ({{det_scores[tl_type]:.2f}})"
        cv2.putText(img_detection, info_text, (x1, y1-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Detecciones inválidas
    for det_idx, det in enumerate(invalid):
        x1, y1, x2, y2 = map(int, det[1:5])
        cv2.rectangle(img_detection, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img_detection, f"Invalid{{det_idx}}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # Guardar imagen de detección en carpeta específica
    out_img1 = os.path.join(detection_dir, frame_name.replace('.jpg','_detection.jpg'))
    cv2.imwrite(out_img1, img_detection)
    
    # ────── IMAGEN 2: ETAPA RECONOCIMIENTO ──────
    img_recognition = image_np.copy()
    
    # Título de la etapa
    cv2.putText(img_recognition, f"STAGE 2: RECOGNITION - {{test_name.upper()}}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Proyecciones (azul claro)
    for x1, y1, x2, y2, pid in bboxes:
        cv2.rectangle(img_recognition, (x1, y1), (x2, y2), (200, 100, 0), 1)
        cv2.putText(img_recognition, f"P{{pid}}", (x1, y1-5),
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
        info_text = f"{{pred_color.upper()}} ({{max_conf:.2f}})"
        cv2.putText(img_recognition, info_text, (x1, y1-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    
    # Guardar imagen de reconocimiento en carpeta específica
    out_img2 = os.path.join(recognition_dir, frame_name.replace('.jpg','_recognition.jpg'))
    cv2.imwrite(out_img2, img_recognition)
    
    # ────── IMAGEN 3: ETAPA FINAL (TRACKING) ──────
    img_final = image_np.copy()
    
    # Título de la etapa
    cv2.putText(img_final, f"STAGE 3: FINAL - {{test_name.upper()}}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Proyecciones con estado revisado
    for x1, y1, x2, y2, pid in bboxes:
        cv2.rectangle(img_final, (x1, y1), (x2, y2), (100, 100, 255), 1)
        
        # Mostrar estado revisado para esta proyección
        rev_color, blink = revised.get(pid, ('none', False))
        proj_text = f"P{{pid}}: {{rev_color.upper()}}"
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
        info_text = f"Det{{det_idx}}>P{{proj_id}}: {{pred_color}}>{{rev_color}}"
        if blink:
            info_text += " *BLINK*"
        cv2.putText(img_final, info_text, (x1, y1-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    # Guardar imagen final en carpeta específica
    out_img3 = os.path.join(final_dir, frame_name.replace('.jpg','_final.jpg'))
    cv2.imwrite(out_img3, img_final)
    
    print(f"✅ {{frame_name}}: {{len(valid)}} valid, {{len(invalid)}} invalid, {{len(bboxes)}} projections → imágenes guardadas")

print(f"\\n📄 RESULTADOS CASO: {test_name.upper()}")
print(f"📊 CSV Files:")
print(f"   📋 0_all_detections.csv: TODAS las detecciones (válidas + inválidas)")
print(f"   🔍 1_detection_results.csv: Solo detecciones válidas del detector")
print(f"   🎨 2_recognition_results.csv: Después de clasificación de color") 
print(f"   🏁 3_final_results.csv: Después de asignación y tracking")
print(f"\\n🖼️  Image Folders:")
print(f"   📁 1_detection/: {{len(entries)}} imágenes mostrando detecciones")
print(f"   📁 2_recognition/: {{len(entries)}} imágenes mostrando clasificación de colores")
print(f"   📁 3_final/: {{len(entries)}} imágenes mostrando tracking final")
print(f"\\n📂 Todo en: {{output_dir}}/")
'''
    
    script_path = test_dir / f"run_test_{test_name}.py"
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"   📝 Script creado: {script_path}")

def main():
    # Configuración
    source_dir = "frames_auto_labeled"
    base_output_dir = "robustness_tests"
    
    # Verificar directorio fuente
    if not os.path.exists(source_dir):
        print(f"❌ Directorio fuente no encontrado: {source_dir}")
        return
    
    # Crear directorio base de salida
    os.makedirs(base_output_dir, exist_ok=True)
    
    print("🧪 CONFIGURADOR DE PRUEBAS DE ROBUSTEZ")
    print("=" * 50)
    print(f"📥 Fuente: {source_dir}")
    print(f"📤 Destino: {base_output_dir}")
    
    # Seleccionar efectos para pruebas
    effects_to_test = [
        'original',        # Sin modificación (baseline)
        'sepia',          # Luz cálida
        'blue_night',     # Condiciones nocturnas  
        'dark',           # Baja iluminación
        'bright',         # Alta iluminación
        'low_contrast',   # Bajo contraste
        'fog_light',      # Niebla ligera
        'rain_light',     # Lluvia ligera
        'noise_light',    # Ruido de cámara
        'jpeg_compression' # Compresión agresiva
    ]
    
    print(f"\\n🎨 Creando {len(effects_to_test)} casos de prueba:")
    for effect in effects_to_test:
        print(f"   • {effect}")
    
    # Crear cada caso de prueba
    created_tests = []
    
    for effect_name in effects_to_test:
        if effect_name not in TEST_EFFECTS:
            print(f"⚠️  Efecto desconocido: {effect_name}, saltando...")
            continue
        
        try:
            test_dir = create_test_case(
                source_dir, 
                effect_name, 
                TEST_EFFECTS[effect_name], 
                base_output_dir
            )
            created_tests.append((effect_name, test_dir))
            
        except Exception as e:
            print(f"❌ Error creando caso {effect_name}: {e}")
    
    # Crear script maestro de ejecución
    create_master_execution_script(created_tests, base_output_dir)
    
    print(f"\\n🎉 ¡Configuración completada!")
    print(f"📁 {len(created_tests)} casos de prueba creados en: {base_output_dir}/")
    print(f"\\n🚀 Para ejecutar cada prueba individualmente:")
    
    for effect_name, test_dir in created_tests:
        script_path = test_dir / f"run_test_{effect_name}.py"
        print(f"   python {script_path}")
    
    print(f"\\n🎯 O ejecutar todas las pruebas:")
    print(f"   python {base_output_dir}/run_all_tests.py")

def create_master_execution_script(created_tests, base_output_dir):
    """Crea un script para ejecutar todas las pruebas automáticamente"""
    
    master_script = f'''import subprocess
import sys
import time
from pathlib import Path

def run_test(test_name, script_path):
    """Ejecuta una prueba individual"""
    print(f"\\n🧪 EJECUTANDO PRUEBA: {{test_name.upper()}}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              capture_output=True, text=True, check=True)
        
        execution_time = time.time() - start_time
        
        print(f"✅ {{test_name}} completada en {{execution_time:.2f}}s")
        return True, execution_time
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {{test_name}}:")
        print(f"   Stdout: {{e.stdout[-500:] if e.stdout else 'N/A'}}")  # Últimas 500 chars
        print(f"   Stderr: {{e.stderr[-500:] if e.stderr else 'N/A'}}")
        return False, 0

def main():
    print("🚀 EJECUTOR MAESTRO DE PRUEBAS DE ROBUSTEZ")
    print("=" * 60)
    
    # Lista de pruebas a ejecutar
    tests = [
{chr(10).join([f"        ('{name}', Path(r'{test_dir}') / 'run_test_{name}.py')," for name, test_dir in created_tests])}
    ]
    
    print(f"🎯 Ejecutando {{len(tests)}} pruebas...")
    
    results = []
    total_start_time = time.time()
    
    for test_name, script_path in tests:
        success, exec_time = run_test(test_name, script_path)
        results.append((test_name, success, exec_time))
    
    total_time = time.time() - total_start_time
    
    # Resumen final
    print(f"\\n📊 RESUMEN DE EJECUCIÓN")
    print("=" * 60)
    
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"✅ Exitosas: {{successful}}")
    print(f"❌ Fallidas: {{failed}}")
    print(f"⏱️  Tiempo total: {{total_time:.2f}}s")
    print(f"⏱️  Tiempo promedio: {{total_time/len(results):.2f}}s/prueba")
    
    print(f"\\n📋 DETALLE POR PRUEBA:")
    for test_name, success, exec_time in results:
        status = "✅" if success else "❌"
        print(f"   {{status}} {{test_name:<15}} {{exec_time:>6.2f}}s")
    
    if failed > 0:
        print(f"\\n⚠️  Revisa los outputs anteriores para detalles de errores")
    
    print(f"\\n📁 Resultados en: {base_output_dir}/[test_name]/output/")

if __name__ == "__main__":
    main()
'''
    
    master_script_path = Path(base_output_dir) / "run_all_tests.py"
    
    with open(master_script_path, 'w', encoding='utf-8') as f:
        f.write(master_script)
    
    print(f"   🎯 Script maestro creado: {master_script_path}")

if __name__ == "__main__":
    main()