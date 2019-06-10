import cv2 as cv


class RGBHistogram:
    def __init__(self, bins):
        self.bins = bins

    def describe(self, image):
        histogram = cv.calcHist([image], [0, 1, 2], None,
                                     self.bins, [0, 256, 0, 256, 0, 256])

        histogram = cv.normalize(histogram, histogram)

        return histogram.flatten()
