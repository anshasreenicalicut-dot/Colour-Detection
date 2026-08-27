import cv2
from ultralytics import YOLO

from color_detector import detect_car_color


class TrafficDetector:

    def __init__(self):

        print("Loading YOLOv8 pretrained model...")

        # Use pretrained model for now
        self.model = YOLO("yolov8n.pt")

        print("Model loaded successfully.")

    def process_image(self, image):

        if image is None:
            return None, 0, 0

        output = image.copy()

        car_count = 0
        person_count = 0

        # ==========================================
        # YOLO DETECTION
        # ==========================================

        results = self.model(
            image,
            conf=0.20,
            imgsz=640,
            verbose=False
        )

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # Keep coordinates inside image
                h, w = image.shape[:2]

                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                # ==========================================
                # CAR
                # COCO CLASS = 2
                # ==========================================

                if cls_id == 2:

                    car_crop = image[
                        y1:y2,
                        x1:x2
                    ]

                    # Detect car colour
                    car_color = detect_car_color(
                        car_crop
                    )

                    # Blue car -> RED BOX
                    if car_color == "Blue":

                        box_color = (
                            0,
                            0,
                            255
                        )

                    # Other colour -> BLUE BOX
                    else:

                        box_color = (
                            255,
                            0,
                            0
                        )

                    cv2.rectangle(
                        output,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        3
                    )

                    cv2.putText(
                        output,
                        f"{car_color} Car {confidence:.2f}",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2
                    )

                    car_count += 1

                # ==========================================
                # PERSON
                # COCO CLASS = 0
                # ==========================================

                elif cls_id == 0:

                    box_color = (
                        0,
                        255,
                        0
                    )

                    cv2.rectangle(
                        output,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    cv2.putText(
                        output,
                        f"Person {confidence:.2f}",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2
                    )

                    person_count += 1

                # ==========================================
                # TRAFFIC LIGHT
                # COCO CLASS = 9
                # ==========================================

                elif cls_id == 9:

                    box_color = (
                        0,
                        255,
                        255
                    )

                    cv2.rectangle(
                        output,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    cv2.putText(
                        output,
                        f"Traffic Light {confidence:.2f}",
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2
                    )

        # ==========================================
        # INFORMATION PANEL
        # ==========================================

        cv2.rectangle(
            output,
            (10, 10),
            (330, 100),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output,
            f"Cars: {car_count}",
            (25, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"People: {person_count}",
            (25, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        return output, car_count, person_count


# ==========================================================
# TEST detect.py DIRECTLY
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CAR COLOUR DETECTION TEST")
    print("=" * 60)

    # Load test image
    image = cv2.imread("test.jpg")

    if image is None:

        print("\nERROR: test.jpg was not found!")
        print("Place test.jpg in the project folder.")
        exit()

    print("\nImage loaded successfully.")

    # Create detector
    detector = TrafficDetector()

    print("\nRunning detection...")

    # Process image
    result, car_count, person_count = detector.process_image(
        image
    )

    print("\n" + "=" * 60)
    print(f"Cars detected   : {car_count}")
    print(f"People detected : {person_count}")
    print("=" * 60)

    # Save result
    cv2.imwrite(
        "detection_result.jpg",
        result
    )

    print("\nResult saved as:")
    print("detection_result.jpg")

    # Show result
    cv2.imshow(
        "Car Colour Detection",
        result
    )

    print("\nPress any key on the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()