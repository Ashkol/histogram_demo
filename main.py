import cv2 as cv
import numpy as np
import os, argparse
from rgbhistogram import RGBHistogram

MAX_BEST_FITS_DISPLAY = 12


def load_images(path):
    images = []
    for file in os.listdir(path):
        img = cv.imread(os.path.join(path, file))
        if img is not None:
            images.append(img)
    return images


def chi2_distance(hist_a, hist_b, eps = 1e-10):
    d = 0.5 * np.sum([((a-b) ** 2) / (a + b + eps)
                        for (a, b) in zip(hist_a, hist_b)])
    return d


def display_best_finds(image, best_finds):
    max_y = max([i.shape[0] for i in best_finds])
    display_images = [np.zeros(shape=(max_y, i.shape[1], 3), dtype="uint8") for i in best_finds]
    for i in range(len(best_finds)):
        display_images[i][max_y-best_finds[i].shape[0]:max_y, 0:best_finds[i].shape[1], :] = best_finds[i]
    display_image = cv.hconcat(display_images)
    cv.imshow("Model", image)
    cv.imshow("Best fits (shown from left)", display_image)
    cv.waitKey(0)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", required=True,
                        help="Path to the directory that contains models")
    parser.add_argument("-d", "--dataset", required=True,
                        help="Path to the directory that contains images")
    parser.add_argument("-b", "--bin_size", required=False, default=8,
                        help="Histogram columns number")
    parser.add_argument("-img", "--image", required=False, default="Models/ryu_idle.png",
                        help="Image for which we're looking for similar ones")
    args = parser.parse_args()
    bin_size = args.bin_size

    model_image = cv.imread(args.image)
    images = load_images(args.dataset)

    desc = RGBHistogram([bin_size, bin_size, bin_size])
    image_descriptions = []
    model_description = (model_image, desc.describe(model_image))
    for i in images:
        image_descriptions.append((i, desc.describe(i)))

    results = []
    for im_desc in image_descriptions:
        d = chi2_distance(model_description[1], im_desc[1])
        results.append(d)

    sorted_images = [x[0] for _, x in sorted(zip(results, image_descriptions))]

    display_best_finds(model_description[0], sorted_images[:MAX_BEST_FITS_DISPLAY if len(image_descriptions)>MAX_BEST_FITS_DISPLAY else len (image_descriptions)])


if __name__ == "__main__":
    main()