import cv2
import os
import sys

def extract_all_frames_from_video(video_path, output_dir):
    """
    Extrae TODOS los frames de un video para análisis completo
    """
    print(f"🎬 Extrayendo TODOS los frames de: {video_path}")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video {video_path}")
        return False
    
    # Obtener información del video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 Video info:")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duración: {duration:.2f}s")
    print(f"   Extrayendo TODOS los {total_frames} frames")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Guardar CADA frame
        frame_name = f"frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
        
        # Mostrar progreso cada 30 frames
        if frame_count % 30 == 0:
            print(f"   📸 Procesados {frame_count}/{total_frames} frames...")
    
    cap.release()
    print(f"✅ Extraídos {frame_count} frames completos en {output_dir}")
    return True

if __name__ == "__main__":
    # Usar solo un video para este test
    video_path = "inputTesis/Flecha roja 1er export.mp4"
    output_dir = "test_retroalimentacion_bbox/frames_flecha_roja"
    
    if os.path.exists(video_path):
        extract_all_frames_from_video(video_path, output_dir)
    else:
        print(f"❌ Video no encontrado: {video_path}")
    
    print("\n🎯 Próximos pasos:")
    print("1. Revisar los frames extraídos")
    print("2. Seleccionar proyecciones manualmente para el primer frame")
    print("3. Implementar sistema de retroalimentación de bounding boxes")