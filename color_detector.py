import cv2
import numpy as np


def detect_car_color(car_image, return_confidence=False):
    """
    Detect the dominant colour of a car crop.

    Returns:
        Blue, Red, Green, Yellow, Orange, Purple,
        White, Black, Gray, Silver, Other

    Project rule:
        Blue car  -> RED bounding box
        Other car -> BLUE bounding box
    """

    if car_image is None or car_image.size == 0:
        if return_confidence:
            return "Unknown", 0.0
        return "Unknown"

    # ---------------------------------------------------------
    # Resize
    # ---------------------------------------------------------

    image = cv2.resize(
        car_image,
        (160, 160),
        interpolation=cv2.INTER_AREA
    )

    h, w = image.shape[:2]

    # ---------------------------------------------------------
    # Use central car area
    # Reduce road/background influence
    # ---------------------------------------------------------

    x1 = int(w * 0.10)
    x2 = int(w * 0.90)

    y1 = int(h * 0.15)
    y2 = int(h * 0.90)

    image = image[y1:y2, x1:x2]

    if image.size == 0:
        if return_confidence:
            return "Unknown", 0.0
        return "Unknown"

    # ---------------------------------------------------------
    # Convert to HSV
    # ---------------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    total_pixels = H.size

    if total_pixels == 0:
        if return_confidence:
            return "Unknown", 0.0
        return "Unknown"

    # ---------------------------------------------------------
    # Remove extremely dark and overexposed pixels
    # ---------------------------------------------------------

    valid = (
        (V > 25) &
        (V < 250)
    )

    # ---------------------------------------------------------
    # Colour masks
    # ---------------------------------------------------------

    masks = {}

    # BLUE
    masks["Blue"] = (
        (H >= 90) &
        (H <= 135) &
        (S >= 70) &
        (V >= 45)
    )

    # RED
    masks["Red"] = (
        (
            (H <= 10) |
            (H >= 170)
        ) &
        (S >= 70) &
        (V >= 45)
    )

    # GREEN
    masks["Green"] = (
        (H >= 35) &
        (H <= 85) &
        (S >= 60) &
        (V >= 40)
    )

    # YELLOW
    masks["Yellow"] = (
        (H >= 18) &
        (H <= 38) &
        (S >= 70) &
        (V >= 60)
    )

    # ORANGE
    masks["Orange"] = (
        (H >= 5) &
        (H <= 20) &
        (S >= 80) &
        (V >= 60)
    )

    # PURPLE
    masks["Purple"] = (
        (H >= 135) &
        (H <= 170) &
        (S >= 60) &
        (V >= 40)
    )

    # ---------------------------------------------------------
    # Calculate colour percentages
    # ---------------------------------------------------------

    scores = {}

    valid_count = np.sum(valid)

    if valid_count > 0:

        for colour, mask in masks.items():

            mask = mask & valid

            pixels = np.sum(mask)

            scores[colour] = pixels / valid_count

    # ---------------------------------------------------------
    # Check chromatic colours
    # ---------------------------------------------------------

    if scores:

        best_colour = max(
            scores,
            key=scores.get
        )

        best_score = scores[best_colour]

    else:

        best_colour = "Other"
        best_score = 0.0

    # ---------------------------------------------------------
    # Detect white / black / gray / silver
    # ---------------------------------------------------------

    # Low saturation = grayscale/white/black region
    low_saturation = S < 45

    # BLACK
    black_mask = (
        low_saturation &
        (V < 65)
    )

    # WHITE
    white_mask = (
        low_saturation &
        (V >= 170)
    )

    # GRAY / SILVER
    gray_mask = (
        low_saturation &
        (V >= 65) &
        (V < 170)
    )

    black_ratio = np.sum(
        black_mask
    ) / total_pixels

    white_ratio = np.sum(
        white_mask
    ) / total_pixels

    gray_ratio = np.sum(
        gray_mask
    ) / total_pixels

    # ---------------------------------------------------------
    # Determine final colour
    # ---------------------------------------------------------

    # Give chromatic colours priority when they are strong
    if best_score >= 0.12:

        detected_colour = best_colour
        confidence = best_score

    else:

        # Determine achromatic colour
        achromatic_scores = {
            "Black": black_ratio,
            "White": white_ratio,
            "Gray": gray_ratio
        }

        achromatic_colour = max(
            achromatic_scores,
            key=achromatic_scores.get
        )

        achromatic_score = achromatic_scores[
            achromatic_colour
        ]

        if achromatic_score >= 0.30:

            detected_colour = achromatic_colour
            confidence = achromatic_score

        elif best_score >= 0.05:

            detected_colour = best_colour
            confidence = best_score

        else:

            detected_colour = "Other"
            confidence = best_score

    # ---------------------------------------------------------
    # Silver refinement
    # ---------------------------------------------------------

    if detected_colour == "Gray":

        if (
            np.mean(V[low_saturation]) > 115
            if np.any(low_saturation)
            else False
        ):
            detected_colour = "Silver"

    confidence = min(
        confidence,
        1.0
    )

    # ---------------------------------------------------------
    # Return result
    # ---------------------------------------------------------

    if return_confidence:

        return (
            detected_colour,
            confidence
        )

    return detected_colour


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CAR COLOUR DETECTOR TEST")
    print("=" * 60)

    image = cv2.imread("test.jpg")

    if image is None:

        print("ERROR: test.jpg not found!")
        print()
        print(
            "Place test.jpg in the project folder."
        )

        exit()

    print()
    print(
        "WARNING:"
    )
    print(
        "test.jpg should ideally contain ONE CAR."
    )
    print(
        "For a traffic image containing multiple cars,"
    )
    print(
        "use detect.py so YOLO can crop each car first."
    )

    colour, confidence = detect_car_color(
        image,
        return_confidence=True
    )

    print()
    print(
        f"Detected Colour : {colour}"
    )

    print(
        f"Confidence       : {confidence:.2f}"
    )

    print("=" * 60)