import cv2
import numpy as np
import torch
import sys
import os

# Agregar path al módulo tlr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tlr.pipeline import load_pipeline

# Cargar projection boxes
def load_projections(filepath):
    """Cargar bounding boxes de proyección desde archivo CSV"""
    entries = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                frame_name = parts[0]
                # bbox formato: [x1, y1, x2, y2, id]
                bbox = list(map(int, parts[1:6]))  # Lee todos los 5 valores
                entries.setdefault(frame_name, []).append(bbox)
    return entries

# Determinar device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Usando device: {device}")

# Cargar pipeline
print("🔄 Cargando pipeline...")
pipeline = load_pipeline(device)
print("✅ Pipeline cargado\n")

# Directorio base
base_dir = os.path.dirname(__file__)

# Cargar proyecciones
proj_file = os.path.join(base_dir, 'projection_bboxes_master.txt')
if not os.path.exists(proj_file):
    print(f"❌ No se encontró {proj_file}")
    exit(1)

entries = load_projections(proj_file)
print(f"📋 Cargadas proyecciones para {len(entries)} frames\n")

# Probar los primeros 50 frames
test_frames = [f'frame_{i:04d}.jpg' for i in range(50)]

print("="*80)
print("COMPARACIÓN: ORIGINAL vs FILTRADO")
print("="*80)

for frame_name in test_frames:
    # Buscar proyección para este frame
    if frame_name not in entries:
        print(f"⚠️  No hay proyección para {frame_name}, saltando...")
        continue

    proj_bboxes = entries[frame_name]

    # Procesar frame ORIGINAL
    original_path = os.path.join(base_dir, 'input_frames', frame_name)
    if not os.path.exists(original_path):
        print(f"⚠️  No existe {original_path}")
        continue

    img_original_np = cv2.imread(original_path)
    if img_original_np is None:
        print(f"⚠️  No se pudo leer {original_path}")
        continue

    img_original = torch.from_numpy(img_original_np.astype(np.float32)).to(device)
    valid_dets_orig, recs_orig, assigns_orig, invalid_dets_orig, revised_orig = pipeline(
        img_original, proj_bboxes, frame_ts=0.0
    )

    # Procesar frame FILTRADO
    filtered_path = os.path.join(base_dir, 'frames_with_filter', frame_name)
    if not os.path.exists(filtered_path):
        print(f"⚠️  No existe {filtered_path}")
        continue

    img_filtered_np = cv2.imread(filtered_path)
    if img_filtered_np is None:
        print(f"⚠️  No se pudo leer {filtered_path}")
        continue

    img_filtered = torch.from_numpy(img_filtered_np.astype(np.float32)).to(device)
    valid_dets_filt, recs_filt, assigns_filt, invalid_dets_filt, revised_filt = pipeline(
        img_filtered, proj_bboxes, frame_ts=0.0
    )

    # Mostrar resultados
    print(f"\n📸 {frame_name}")
    print("-" * 80)

    # Original
    print("  🔵 ORIGINAL:")
    if len(valid_dets_orig) > 0:
        for i in range(len(valid_dets_orig)):
            det = valid_dets_orig[i]
            rec = recs_orig[i] if i < len(recs_orig) else None

            # Obtener color reconocido
            color_names = ['black', 'red', 'yellow', 'green']
            color_idx = int(torch.argmax(rec)) if rec is not None else 0
            color = color_names[color_idx]

            # Score de la detección (max de type scores)
            score = torch.max(det[5:9]).item()

            print(f"     Luz {i+1}: {color.upper()} (score: {score:.2f})")
    else:
        print("     No se detectó semáforo")

    # Filtrado
    print("  🌅 FILTRADO:")
    if len(valid_dets_filt) > 0:
        for i in range(len(valid_dets_filt)):
            det = valid_dets_filt[i]
            rec = recs_filt[i] if i < len(recs_filt) else None

            # Obtener color reconocido
            color_names = ['black', 'red', 'yellow', 'green']
            color_idx = int(torch.argmax(rec)) if rec is not None else 0
            color = color_names[color_idx]

            # Score de la detección
            score = torch.max(det[5:9]).item()

            # Marcar si cambió la detección
            cambio = ""
            if len(valid_dets_orig) > i:
                rec_orig = recs_orig[i] if i < len(recs_orig) else None
                color_idx_orig = int(torch.argmax(rec_orig)) if rec_orig is not None else 0
                color_orig = color_names[color_idx_orig]
                if color != color_orig:
                    cambio = f" ← CAMBIÓ de {color_orig.upper()}"

            print(f"     Luz {i+1}: {color.upper()} (score: {score:.2f}){cambio}")
    else:
        print("     No se detectó semáforo")

print("\n" + "="*80)
print("FIN DE PRUEBA")
print("="*80)
