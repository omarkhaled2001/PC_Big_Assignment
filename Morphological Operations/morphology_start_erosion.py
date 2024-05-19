import numpy as np
import cv2
import sys

def erosion_kernel(img, num_iterations=1):
    # Define structuring element for erosion
    kernel = np.ones((3, 3), np.uint8)

    # Perform erosion operation
    eroded_img = cv2.erode(img, kernel, iterations=num_iterations)

    return eroded_img

def dilation_kernel(img, num_iterations=1):
    # Define structuring element for dilation
    kernel = np.ones((3, 3), np.uint8)

    # Perform dilation operation
    dilated_img = cv2.dilate(img, kernel, iterations=num_iterations)

    return dilated_img

def main(img_path, num_iterations):
    # Read input image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Perform erosion operation
    eroded_img = erosion_kernel(img, num_iterations)

    # Perform dilation operation
    dilated_img = dilation_kernel(img, num_iterations)

    # Save output images
    cv2.imwrite("eroded_image.jpg", eroded_img)
    cv2.imwrite("dilated_image.jpg", dilated_img)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <image_path> <num_iterations>")
        sys.exit(1)

    img_path = sys.argv[1]
    num_iterations = int(sys.argv[2])

    main(img_path, num_iterations)
