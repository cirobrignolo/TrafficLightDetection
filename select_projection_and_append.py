import cv2
import os
import sys

if len(sys.argv) != 2:
    print("Uso: python3 select_projection_and_append.py <frame.jpg>")
    sys.exit(1)

image_path = sys.argv[1]
image_name = os.path.basename(image_path)
#output_path = "frames_labeled/projection_bboxes_master.txt"
output_path = "frames_auto_labeled/projection_bboxes_master.txt"

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"No se pudo abrir la imagen en {image_path}")

clone = image.copy()
boxes = []
id_counter = 0
drawing = False
x_start, y_start = -1, -1

scale = 1
resized = cv2.resize(clone, (0, 0), fx=scale, fy=scale)

def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, drawing, boxes, id_counter, resized

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        image_copy = resized.copy()
        cv2.rectangle(image_copy, (x_start, y_start), (x, y), (255, 0, 0), 2)
        cv2.imshow('Selector', image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        x1 = int(min(x_start, x_end) / scale)
        y1 = int(min(y_start, y_end) / scale)
        x2 = int(max(x_start, x_end) / scale)
        y2 = int(max(y_start, y_end) / scale)
        boxes.append([x1, y1, x2, y2, id_counter])
        id_counter += 1
        cv2.rectangle(resized, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (255, 0, 0), 2)
        cv2.imshow('Selector', resized)

cv2.namedWindow('Selector')
cv2.setMouseCallback('Selector', draw_rectangle)

print("🖱️ Dibujá las proyecciones. Cerrá con ESC.")
cv2.imshow('Selector', resized)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or cv2.getWindowProperty('Selector', cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()

# Guardar en master.txt
with open(output_path, "a") as f:
    for b in boxes:
        line = f"{image_name},{b[0]},{b[1]},{b[2]},{b[3]},{b[4]}\n"
        f.write(line)

print(f"\n✅ Cajas agregadas a {output_path}")

