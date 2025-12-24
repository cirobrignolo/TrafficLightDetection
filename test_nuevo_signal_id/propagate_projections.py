import os
import glob
import cv2

# Crear carpeta para verificar alineación
os.makedirs('projection_verification', exist_ok=True)

# Leer las proyecciones del frame_0000.jpg desde projection_bboxes_master.txt
projections = []
with open('projection_bboxes_master.txt', 'r') as f:
    for line in f:
        if line.strip():
            parts = line.strip().split(',')
            if len(parts) == 6:
                projections.append([int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])])

print(f"📊 Proyecciones base: {len(projections)}")
for i, proj in enumerate(projections):
    print(f"   Proj {i}: [{proj[0]}, {proj[1]}, {proj[2]}, {proj[3]}] ID:{proj[4]}")

# Obtener todos los frames desde input_frames/
frames = sorted(glob.glob('input_frames/frame_*.jpg'))
print(f"🖼️ Total frames encontrados: {len(frames)}")

# Propagar proyecciones y crear imágenes de verificación
with open('projection_bboxes_master.txt', 'a') as f:
    for frame in frames:
        frame_name = os.path.basename(frame)
        
        # Leer imagen para verificación visual
        img = cv2.imread(frame)
        if img is not None:
            # Dibujar proyecciones en la imagen
            for proj in projections:
                x1, y1, x2, y2, box_id = proj
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f'ID:{box_id}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Guardar imagen de verificación
            verification_path = os.path.join('projection_verification', f'{frame_name}_proj.jpg')
            cv2.imwrite(verification_path, img)
        
        # Propagar proyecciones al archivo (skip frame_0000.jpg que ya las tiene)
        if frame_name != 'frame_0000.jpg':
            for proj in projections:
                x1, y1, x2, y2, box_id = proj
                f.write(f"{frame_name},{x1},{y1},{x2},{y2},{box_id}\n")

print(f"✅ Proyecciones propagadas a {len(frames)-1} frames adicionales")
print(f"🖼️ Imágenes de verificación creadas: {len(frames)}")
print("📄 Archivo projection_bboxes_master.txt actualizado")
print("📁 Verificar alineación en: projection_verification/")
print("\n🎯 Próximo paso:")
print("   1. Revisar imágenes en projection_verification/ para ver si las proyecciones están bien alineadas")
print("   2. Si están bien: python3 run_pipeline_debug_stages_fixed.py")
print("   3. Si no están bien: ajustar manualmente algunas proyecciones clave")