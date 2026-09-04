from ultralytics import YOLO

model = YOLO("yolov8n.pt")

print("COCO MODEL CLASSES:")
print(model.names)

results = model(
    "test.jpg",
    conf=0.10,
    imgsz=640,
    save=True,
    verbose=True
)

for result in results:

    print("\nDETECTIONS:")

    if result.boxes is None or len(result.boxes) == 0:
        print("NO OBJECTS DETECTED")
        continue

    for box in result.boxes:

        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Class ID: {cls_id} | "
            f"Class: {model.names[cls_id]} | "
            f"Confidence: {confidence:.3f}"
        )