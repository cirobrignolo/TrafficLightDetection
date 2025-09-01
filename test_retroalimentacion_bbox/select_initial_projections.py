import cv2
import os
import sys

if len(sys.argv) != 2:
    print("Uso: python3 select_initial_projections.py <frame.jpg>")
    print("Ejemplo: python3 select_initial_projections.py frame_0000.jpg")
    sys.exit(1)

image_path = sys.argv[1]
image_name = os.path.basename(image_path)
# Guardar en nuestra carpeta del test
output_path = "initial_projections.txt"

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"No se pudo abrir la imagen en {image_path}")

print(f"🎯 Seleccionando proyecciones iniciales para: {image_name}")
print("📝 Instrucciones:")
print("   - Haz clic y arrastra para crear un rectángulo")
print("   - Presiona 's' para guardar y continuar")
print("   - Presiona 'r' para resetear todas las selecciones")
print("   - Presiona 'z' para deshacer la última selección")
print("   - Presiona 'q' para salir sin guardar")

clone = image.copy()
boxes = []
id_counter = 0
drawing = False
x_start, y_start = -1, -1

# Escalar imagen si es muy grande
h, w = image.shape[:2]
if w > 1200:
    scale = 1200 / w
else:
    scale = 1

resized = cv2.resize(clone, (0, 0), fx=scale, fy=scale)

def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, drawing, boxes, id_counter, resized

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        image_copy = resized.copy()
        # Dibujar rectángulos existentes
        for box in boxes:
            x1, y1, x2, y2, box_id = box
            cv2.rectangle(image_copy, (int(x1 * scale), int(y1 * scale)), 
                         (int(x2 * scale), int(y2 * scale)), (0, 255, 0), 2)
            cv2.putText(image_copy, f'ID:{box_id}', 
                       (int(x1 * scale), int(y1 * scale) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Dibujar rectángulo actual
        cv2.rectangle(image_copy, (x_start, y_start), (x, y), (255, 0, 0), 2)
        cv2.imshow('Selector de Proyecciones', image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        x1 = int(min(x_start, x_end) / scale)
        y1 = int(min(y_start, y_end) / scale)
        x2 = int(max(x_start, x_end) / scale)
        y2 = int(max(y_start, y_end) / scale)
        
        # Validar que el rectángulo tenga tamaño mínimo
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            boxes.append([x1, y1, x2, y2, id_counter])
            print(f"✅ Proyección {id_counter} agregada: [{x1}, {y1}, {x2}, {y2}]")
            id_counter += 1
        
        # Redibujar imagen con todos los rectángulos
        image_copy = resized.copy()
        for box in boxes:
            x1, y1, x2, y2, box_id = box
            cv2.rectangle(image_copy, (int(x1 * scale), int(y1 * scale)), 
                         (int(x2 * scale), int(y2 * scale)), (0, 255, 0), 2)
            cv2.putText(image_copy, f'ID:{box_id}', 
                       (int(x1 * scale), int(y1 * scale) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Selector de Proyecciones', image_copy)

cv2.namedWindow('Selector de Proyecciones')
cv2.setMouseCallback('Selector de Proyecciones', draw_rectangle)
cv2.imshow('Selector de Proyecciones', resized)

print(f"📊 Dimensiones de imagen: {w}x{h} (escala: {scale:.2f})")

while True:
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):  # Guardar
        if len(boxes) > 0:
            with open(output_path, "w") as f:
                for box in boxes:
                    x1, y1, x2, y2, box_id = box
                    f.write(f"{image_name},{x1},{y1},{x2},{y2},{box_id}\n")
            print(f"💾 Guardadas {len(boxes)} proyecciones en {output_path}")
            break
        else:
            print("⚠️  No hay proyecciones para guardar")
    
    elif key == ord('r'):  # Reset
        boxes = []
        id_counter = 0
        cv2.imshow('Selector de Proyecciones', resized)
        print("🔄 Proyecciones reseteadas")
    
    elif key == ord('z'):  # Undo
        if boxes:
            removed = boxes.pop()
            id_counter = max(0, id_counter - 1)
            image_copy = resized.copy()
            for box in boxes:
                x1, y1, x2, y2, box_id = box
                cv2.rectangle(image_copy, (int(x1 * scale), int(y1 * scale)), 
                             (int(x2 * scale), int(y2 * scale)), (0, 255, 0), 2)
                cv2.putText(image_copy, f'ID:{box_id}', 
                           (int(x1 * scale), int(y1 * scale) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Selector de Proyecciones', image_copy)
            print(f"⏪ Proyección eliminada: {removed}")
        else:
            print("⚠️  No hay proyecciones para eliminar")
    
    elif key == ord('q'):  # Salir sin guardar
        print("❌ Saliendo sin guardar")
        break

cv2.destroyAllWindows()