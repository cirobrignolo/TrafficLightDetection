import cv2
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 🔧 Fix para importar tlr.pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from tlr.pipeline import load_pipeline

def test_variation(variation_dir, pipeline, output_base_dir):
    """
    Prueba una variación específica de imágenes
    """
    variation_name = Path(variation_dir).name
    print(f"\n🔍 PROBANDO VARIACIÓN: {variation_name.upper()}")
    print("=" * 50)
    
    # Configurar directorios
    bbox_file = os.path.join(variation_dir, 'projection_bboxes_master.txt')
    if not os.path.exists(bbox_file):
        print(f"❌ No encontrado: {bbox_file}")
        return None
    
    output_dir = os.path.join(output_base_dir, f"results_{variation_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Leer proyecciones por frame
    entries = {}
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            frame = parts[0]
            bbox = list(map(int, parts[1:]))
            entries.setdefault(frame, []).append(bbox)
    
    # CSV de resultados para esta variación
    results_csv = os.path.join(output_dir, f'robustness_results_{variation_name}.csv')
    
    # Limpiar archivo anterior
    if os.path.exists(results_csv):
        os.remove(results_csv)
    
    # Escribir cabecera
    with open(results_csv, 'w') as out:
        out.write("variation,frame,valid_detections,invalid_detections,avg_confidence,detected_colors,tracking_changes\\n")
    
    # Contadores para estadísticas
    total_valid = 0
    total_invalid = 0
    confidence_scores = []
    color_changes = 0
    processing_times = []
    
    # Procesar cada frame
    for frame_idx, (frame_name, bboxes) in enumerate(entries.items()):
        frame_path = os.path.join(variation_dir, frame_name)
        image_np = cv2.imread(frame_path)
        if image_np is None:
            print(f"❌ No se pudo cargar {frame_name}")
            continue
        
        # Convertir a tensor y pasar a GPU
        image_tensor = torch.from_numpy(image_np.astype(np.float32)).to('cuda:0')
        frame_ts = time.time()
        
        # Medir tiempo de procesamiento
        start_time = time.time()
        
        # Ejecutar pipeline
        valid, recognitions, assignments, invalid, revised = pipeline(
            image_tensor, bboxes, frame_ts
        )
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Analizar resultados
        num_valid = len(valid)
        num_invalid = len(invalid)
        
        # Calcular confianza promedio
        if len(recognitions) > 0:
            confidences = [torch.max(rec).item() for rec in recognitions]
            avg_confidence = np.mean(confidences)
            confidence_scores.extend(confidences)
        else:
            avg_confidence = 0.0
        
        # Detectar colores encontrados
        detected_colors = []
        tracking_changes = 0
        
        for det_idx in range(len(valid)):
            if det_idx < len(recognitions):
                pred_cls = int(torch.argmax(recognitions[det_idx]))
                pred_color = ['black','red','yellow','green'][pred_cls]
                detected_colors.append(pred_color)
        
        # Contar cambios por tracking
        assign_list = assignments.cpu().tolist()
        assign_map = {det_idx: proj_id for proj_id, det_idx in assign_list}
        
        for det_idx in range(len(valid)):
            if det_idx < len(recognitions):
                pred_cls = int(torch.argmax(recognitions[det_idx]))
                pred_color = ['black','red','yellow','green'][pred_cls]
                
                proj_id = assign_map.get(det_idx, -1)
                rev_color, _ = revised.get(proj_id, (pred_color, False)) if proj_id != -1 else (pred_color, False)
                
                if pred_color != rev_color:
                    tracking_changes += 1
        
        # Guardar en CSV
        colors_str = "|".join(detected_colors) if detected_colors else "none"
        with open(results_csv, 'a') as out:
            out.write(f"{variation_name},{frame_name},{num_valid},{num_invalid},{avg_confidence:.4f},{colors_str},{tracking_changes}\\n")
        
        # Acumular estadísticas
        total_valid += num_valid
        total_invalid += num_invalid
        color_changes += tracking_changes
        
        if frame_idx % 10 == 0:
            print(f"  📊 Frame {frame_idx}: {num_valid} válidas, {num_invalid} inválidas, conf={avg_confidence:.3f}")
    
    # Calcular estadísticas finales
    total_frames = len(entries)
    avg_valid_per_frame = total_valid / total_frames if total_frames > 0 else 0
    avg_invalid_per_frame = total_invalid / total_frames if total_frames > 0 else 0
    avg_confidence_overall = np.mean(confidence_scores) if confidence_scores else 0
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    stats = {
        'variation': variation_name,
        'total_frames': total_frames,
        'total_valid_detections': total_valid,
        'total_invalid_detections': total_invalid,
        'avg_valid_per_frame': avg_valid_per_frame,
        'avg_invalid_per_frame': avg_invalid_per_frame,
        'avg_confidence': avg_confidence_overall,
        'tracking_changes': color_changes,
        'avg_processing_time_ms': avg_processing_time * 1000
    }
    
    print(f"\\n📈 ESTADÍSTICAS {variation_name.upper()}:")
    print(f"   🎯 Frames procesados: {total_frames}")
    print(f"   ✅ Detecciones válidas: {total_valid} (promedio: {avg_valid_per_frame:.2f}/frame)")
    print(f"   ❌ Detecciones inválidas: {total_invalid} (promedio: {avg_invalid_per_frame:.2f}/frame)")
    print(f"   🎯 Confianza promedio: {avg_confidence_overall:.4f}")
    print(f"   🔄 Cambios por tracking: {color_changes}")
    print(f"   ⏱️  Tiempo promedio: {avg_processing_time*1000:.2f}ms/frame")
    
    return stats

def main():
    # Configuración
    variations_base_dir = "frames_variations_test"
    output_base_dir = "robustness_test_results"
    
    # Crear directorio de resultados
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Cargar pipeline
    print("🚀 Cargando pipeline...")
    pipeline = load_pipeline('cuda:0')
    print("✅ Pipeline cargado")
    
    # Encontrar todas las variaciones
    variations_dir = Path(variations_base_dir)
    if not variations_dir.exists():
        print(f"❌ Directorio de variaciones no encontrado: {variations_base_dir}")
        print("💡 Ejecuta primero: python generate_test_variations.py")
        return
    
    variation_dirs = [d for d in variations_dir.iterdir() if d.is_dir()]
    
    if not variation_dirs:
        print(f"❌ No se encontraron variaciones en: {variations_base_dir}")
        return
    
    print(f"\\n🔍 Encontradas {len(variation_dirs)} variaciones:")
    for d in variation_dirs:
        print(f"   📁 {d.name}")
    
    # Probar cada variación
    all_stats = []
    
    for variation_dir in variation_dirs:
        try:
            stats = test_variation(str(variation_dir), pipeline, output_base_dir)
            if stats:
                all_stats.append(stats)
        except Exception as e:
            print(f"❌ Error procesando {variation_dir.name}: {e}")
    
    # Generar resumen comparativo
    summary_file = os.path.join(output_base_dir, 'robustness_summary.csv')
    
    with open(summary_file, 'w') as out:
        out.write("variation,total_frames,total_valid,total_invalid,avg_valid_per_frame,avg_invalid_per_frame,avg_confidence,tracking_changes,avg_processing_time_ms\\n")
        
        for stats in all_stats:
            out.write(f"{stats['variation']},{stats['total_frames']},{stats['total_valid_detections']},{stats['total_invalid_detections']},{stats['avg_valid_per_frame']:.4f},{stats['avg_invalid_per_frame']:.4f},{stats['avg_confidence']:.4f},{stats['tracking_changes']},{stats['avg_processing_time_ms']:.2f}\\n")
    
    print(f"\\n📊 RESUMEN COMPARATIVO:")
    print("=" * 80)
    print(f"{'Variación':<15} {'Frames':<7} {'Válidas':<8} {'Inválidas':<9} {'Confianza':<10} {'Cambios':<8} {'Tiempo':<8}")
    print("-" * 80)
    
    for stats in all_stats:
        print(f"{stats['variation']:<15} {stats['total_frames']:<7} {stats['total_valid_detections']:<8} {stats['total_invalid_detections']:<9} {stats['avg_confidence']:<10.4f} {stats['tracking_changes']:<8} {stats['avg_processing_time_ms']:<8.1f}ms")
    
    print(f"\\n📄 Resumen completo guardado en: {summary_file}")
    print(f"📁 Resultados detallados en: {output_base_dir}/")

if __name__ == "__main__":
    main()