import math
from os import write
from typing import ByteString
from PIL import Image

import numpy as np


def read_image(path):
    return np.asarray_chkfinite(Image.open(path).convert('L'))

def write_image(image, path):
    img = Image.fromarray(np.array(image), 'L')
    img.save(path)


DATA_DIR = "letter_data"
X_TEST_DIR = "x_test"
X_TRAIN_DIR = "x_train"


TRAIN_IMAGES_FILENAME = DATA_DIR + "/emnist-letters-train-images-idx3-ubyte"
TRAIN_LABELS_FILENAME = DATA_DIR + "/emnist-letters-train-labels-idx1-ubyte"
TEST_IMAGES_FILENAME = DATA_DIR + "/emnist-letters-test-images-idx3-ubyte"
TEST_LABELS_FILENAME = DATA_DIR + "/emnist-letters-test-labels-idx1-ubyte"


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, "big")


def read_images(file_name, n_max_images=None):
    with open(file_name, "rb") as f:
        images = []

        f.read(4)  # magic number, we don't care
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))

        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(file_name, n_max_labels=None):
    with open(file_name, "rb") as f:
        labels = []
        f.read(4)  # magic number, we don't care
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            label = chr(ord('`') + label)
            labels.append(label)

    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def dist(x, y):
    return math.sqrt(
        sum(
            [
                int.__sub__(*map(bytes_to_int, [x_i, y_i])) ** 2 for x_i, y_i in zip(x, y)
            ]
        )
    )


def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, y_train, X_test, k=3):
    y_pred = []

    for test_sample_idx, sample in enumerate(X_test):
        print(test_sample_idx)
        training_distances = get_training_distances_for_test_sample(X_train, sample)
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
            enumerate(training_distances),
            key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    return y_pred


def main():
    X_train = read_images(TRAIN_IMAGES_FILENAME, 30000)
    y_train = read_labels(TRAIN_LABELS_FILENAME, 30000)
    X_test = read_images(TEST_IMAGES_FILENAME, 100)
    y_test = read_labels(TEST_LABELS_FILENAME, 100)
    
    # X_test = [read_image(f"{X_TRAIN_DIR}/3.png")]
    # y_test = [0]
    
    for idx, test_sample in enumerate(X_test):
        write_image(test_sample, f"{X_TEST_DIR}/{idx}.png")

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)
    
    y_pred = knn(X_train, y_train, X_test, 7)
    
    accuracy = sum([
        y_pred_i == y_test_i
        for y_pred_i, y_test_i
        in zip(y_pred, y_test)
    ]) / len(y_test)
    print("Tests: ", y_test)
    print("Predictions:", y_pred)
    print(f"Accuracy: {accuracy*100}%")


if __name__ == "__main__":
    main()
