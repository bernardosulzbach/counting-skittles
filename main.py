import os
import math
import json
import datetime

import numpy
import cv2

REFERENCE_COLORS = {
    "green": (73, 149, 0),
    "red": (131, 15, 24),
    "purple": (49, 28, 25),
    "yellow": (247, 197, 62),
    "orange": (234, 75, 16)
}


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
        return [0.0 for _ in total]


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


def find_and_write_circles(image, filename, destination_path):
    circles = find_circles(image)
    print('Found {} circles in {}.'.format(len(circles), filename))
    cv2.imwrite(destination_path, image)


def ensure_path_exists(path):
    os.makedirs(path)


class Instance:
    def __init__(self, identifier: str, counts: dict):
        self.id = identifier
        self.counts = counts


def saturation_cut(image, destination_path):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    threshold = numpy.float32(0.4 * 255.0)
    zero = numpy.float32(0.0 * 255.0)
    one = numpy.float32(1.0 * 255.0)
    vectorized = hsv_image.reshape((-1, 3))
    vectorized[:, 1] = numpy.where(vectorized[:, 1] < threshold, zero, vectorized[:, 1])
    vectorized[:, 2] = numpy.where(vectorized[:, 1] == zero, one, vectorized[:, 2])
    hsv_image = vectorized.reshape(hsv_image.shape)
    hsv_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(destination_path, hsv_image)


def segment(image, destination_path):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    threshold = numpy.float32(0.3 * 255.0)
    zero = numpy.float32(0.0 * 255.0)
    one = numpy.float32(1.0 * 255.0)
    vectorized = numpy.float32(hsv_image.reshape((-1, 3)))
    vectorized[:, 1] = numpy.where(vectorized[:, 1] < threshold, zero, vectorized[:, 1])
    vectorized[:, 0] = numpy.where(vectorized[:, 1] == zero, zero, vectorized[:, 0])
    vectorized[:, 2] = numpy.where(vectorized[:, 1] == zero, one, one)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    clusters = 6
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS
    return_value, best_labels, centers = cv2.kmeans(vectorized, clusters, None, criteria, attempts, flags)
    centers = numpy.uint8(centers)
    segmented_image = centers[best_labels.flatten()].reshape(hsv_image.shape)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(destination_path, segmented_image)


def main():
    with open('data/labels.json') as data_handle:
        labels = json.load(data_handle)['labels']
    date_and_time = datetime.datetime.now().isoformat('-')
    output_path = os.path.join('output', date_and_time)
    ensure_path_exists(output_path)
    segmented_path = os.path.join('segmented', date_and_time)
    ensure_path_exists(segmented_path)
    saturation_path = os.path.join('saturation', date_and_time)
    ensure_path_exists(saturation_path)
    instances = []
    for label in labels:
        instances.append(Instance(label, labels[label]))
        filename = label + '.jpg'
        image = cv2.imread(os.path.join('data/images', filename))
        image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
        find_and_write_circles(image.copy(), filename, os.path.join(output_path, filename))
        segment(image.copy(), os.path.join(segmented_path, filename))
        saturation_cut(image.copy(), os.path.join(saturation_path, filename))


if __name__ == '__main__':
    main()
