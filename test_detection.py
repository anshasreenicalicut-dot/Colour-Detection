import cv2
from detect import TrafficDetector


# Load test image
image = cv2.imread("test.jpg")

if image is None:
    print("ERROR: test.jpg not found.")
    exit()

print("Loading detector...")

detector = TrafficDetector()

print("Running detection...")

result, car_count, person_count = detector.process_image(image)

print("================================")
print(f"Cars detected: {car_count}")
print(f"People detected: {person_count}")
print("================================")

# Display result
cv2.imshow("Car Colour Detection", result)

print("Press any key on the image window to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()