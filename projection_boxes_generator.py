import cv2

image_path = 'inputTesis/IMG_2252.JPEG'
image = cv2.imread(image_path)
clone = image.copy()

boxes = []
id_counter = 0
drawing = False
x_start, y_start = -1, -1

def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, drawing, boxes, id_counter, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        image_copy = clone.copy()
        cv2.rectangle(image_copy, (x_start, y_start), (x, y), (0, 255, 0), 2)
        cv2.imshow('Select Projections', image_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y
        box = [min(x_start, x_end), min(y_start, y_end), max(x_start, x_end), max(y_start, y_end), id_counter]
        boxes.append(box)
        id_counter += 1
        cv2.rectangle(clone, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow('Select Projections', clone)

cv2.namedWindow('Select Projections')
cv2.setMouseCallback('Select Projections', draw_rectangle)

print("🖱️ Dibujá las cajas con el mouse. Cerrá la ventana cuando termines.")
cv2.imshow('Select Projections', image)

# Esperar hasta que el usuario presione ESC o cierre la ventana
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    if cv2.getWindowProperty('Select Projections', cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()


print("\n📦 Proyecciones seleccionadas:")
for b in boxes:
    print(b)

# Guardar en un archivo si querés
with open("projection_bboxes.txt", "w") as f:
    for b in boxes:
        f.write(str(b) + "\n")

print("\n✅ Guardadas en 'projection_bboxes.txt'")
