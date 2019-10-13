import math
import cv2
import numpy as np

color_image = cv2.imread('sample-skittles-512.jpg')
gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
# hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
# gray_image = hsv_image[:, :, 0]
gray_image = cv2.medianBlur(gray_image, 5)

# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.5, 25, param1=1, param2=20, minRadius=25, maxRadius=30)
circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 24, param1=1, param2=12, minRadius=12, maxRadius=16)

circles = np.uint16(np.around(circles))
print('Found {} circles.'.format(len(circles[0])))


def average_color(image, c):
    total = [0.0, 0.0, 0.0]
    count = 0
    for x in range(math.floor(c[0] - c[2]), math.ceil(c[0] + c[2] + 1)):
        if x < 0 or x >= len(image):
            continue
        for y in range(math.floor(c[1] - c[2]), math.ceil(c[1] + c[2] + 1)):
            if y < 0 or y >= len(image[x]):
                continue
            if math.hypot(x - c[0], y - c[1]) <= c[2]:
                total += image[x][y]
                count += 1
    if count:
        return [x / count for x in total]
    else:
        return [0.0 for x in total]


def color_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))


used_circles = []
for c in circles[0, :]:
    c_a_c = average_color(color_image, c)
    d_c_a_c = average_color(color_image, [c[0], c[1], 1.25 * c[2]])
    if color_distance(d_c_a_c, c_a_c) > 0.2:
        used_circles.append(c)
for c in used_circles:
    # draw the outer circle
    cv2.circle(color_image, (c[0], c[1]), c[2], (16, 224, 16), 3, lineType=cv2.LINE_AA)
    # draw the center of the circle
    cv2.circle(color_image, (c[0], c[1]), 2, (0, 0, 224), 1, lineType=cv2.LINE_AA)

# cv2.imwrite('skittles-512-unfiltered.jpg', color_image)

cv2.imshow('Detected circles', color_image)
while True:
    key = cv2.waitKey(100)
    if key >= 0:
        break
    if cv2.getWindowProperty('Detected circles', cv2.WND_PROP_VISIBLE) < 1:
        break
cv2.destroyAllWindows()
