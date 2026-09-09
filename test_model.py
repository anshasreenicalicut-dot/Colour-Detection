from ultralytics import YOLO

model = YOLO("models/best.pt")

print("MODEL CLASSES:")
print(model.names)

results = model(
    "test.jpg",
    conf=0.10,
    verbose=True
)

for result in results:

    print("\nDETECTIONS:")

    if result.boxes is None or len(result.boxes) == 0:
        print("NO OBJECTS DETECTED")
        continue

    for box in result.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        print(
            "Class:",
            cls,
            "Name:",
            model.names[cls],
            "Confidence:",
            conf
        )