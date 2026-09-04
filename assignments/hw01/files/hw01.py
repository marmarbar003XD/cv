"""Homework 01: Working with Images.

Complete the TODO functions below. Do not change their names or arguments;
the Gradescope autograder calls these functions directly.

Run locally from the course `cv` Conda environment with:
    python hw01.py my_image.png

The program must print the requested values and save hw01_output.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def load_image(filename: str | Path) -> np.ndarray:
    """Load a color image from disk using OpenCV and return it."""
    # TODO: load the image with OpenCV and return the resulting NumPy array.
    og_img = cv2.imread(str(filename))
    return og_img



def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale float32 with values in [0, 1]."""
    # TODO: convert with OpenCV, then convert to np.float32 in the [0, 1] range.

    # img to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # formats dtype and scales range
    img_g_float = img_gray.astype(np.float32)/ 255.0

    return img_g_float
   


def top_left_3x5(gray_image: np.ndarray) -> np.ndarray:
    """Return the top-left block containing 3 rows and 5 columns."""
    # TODO
    top_left_bck = gray_image[0:3, 0:5]
    return top_left_bck
    


def get_pixel_value(gray_image: np.ndarray, row: int, col: int) -> np.float32:
    """Return the pixel value at zero-based (row, col)."""
    # TODO
    return gray_image[row, col]



def save_grayscale_image(
    gray_image: np.ndarray, output_path: str | Path = "hw01_output.png") -> None:
    """Save the grayscale image using OpenCV."""
    # TODO: save a viewable grayscale image with OpenCV.
    # Hint: cv2.imwrite expects conventional image intensities for a PNG.

    # convert back to norm range and uint8 (the standard) to save
    gray_image = (gray_image * 255).astype(np.uint8)

    cv2.imwrite(output_path, gray_image)


def main() -> None:
    parser = argparse.ArgumentParser(description="HW01: Working with Images")
    parser.add_argument("image", help="Path to your personal headshot image")
    args = parser.parse_args()

    image = load_image(args.image)
    gray = convert_to_grayscale(image)
    block = top_left_3x5(gray)
    pixel = get_pixel_value(gray, 1, 2)

    print("Top-left 3x5 grayscale block:")
    print(block)
    print("\nPixel value at row 1, column 2:")
    print(pixel)

    save_grayscale_image(gray, "hw01_output.png")
    print("\nSaved grayscale image to hw01_output.png")


if __name__ == "__main__":
    main()
