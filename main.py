import numpy
import os
import math
import cv2
import json
import datetime


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


def find_circles(color_image):
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
    # hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
    # gray_image = hsv_image[:, :, 0]
    gray_image = cv2.medianBlur(gray_image, 5)

    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 0.5, 25, param1=1, param2=20, minRadius=25, maxRadius=30)
    circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1, 24, param1=1, param2=12, minRadius=12, maxRadius=16)

    circles = numpy.uint16(numpy.around(circles))

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

    return circles[0]


def ensure_path_exists(path):
    os.makedirs(path)


def main():
    with open('data/labels.json') as data_handle:
        labels = json.load(data_handle)
    date_and_time = datetime.datetime.now().isoformat('-')
    output_path = os.path.join('output', date_and_time)
    ensure_path_exists(output_path)
    for label in labels['labels']:
        filename = label + '.jpg'
        image = cv2.imread(os.path.join('data/images', filename))
        image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
        circles = find_circles(image)
        print('Found {} circles in {}.'.format(len(circles), filename))
        cv2.imwrite(os.path.join(output_path, filename), image)
        break


if __name__ == '__main__':
    main()
