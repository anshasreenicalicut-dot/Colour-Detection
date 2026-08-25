from ultralytics import YOLO
import os
import shutil


def main():

    print("=" * 60)
    print("FAST CAR DETECTION TRAINING")
    print("=" * 60)

    # Small YOLO model
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",

        # FAST SETTINGS
        epochs=3,
        imgsz=416,
        batch=4,

        # CPU settings
        workers=2,
        device="cpu",

        # Stop early if model stops improving
        patience=2,

        # Reduce augmentation to speed up training
        mosaic=0.0,
        amp=False,

        project="runs",
        name="car_detection_fast",
        exist_ok=True,

        verbose=True
    )

    # Location of trained model
    trained_model = (
        "runs/car_detection_fast/weights/best.pt"
    )

    # Create models folder
    os.makedirs("models", exist_ok=True)

    if os.path.exists(trained_model):

        shutil.copy(
            trained_model,
            "models/best.pt"
        )

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED!")
        print("=" * 60)
        print("Model saved at:")
        print("models/best.pt")

    else:

        print("\nTraining finished, but best.pt was not found.")


if __name__ == "__main__":
    main()