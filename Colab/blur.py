import cv2
import numpy as np

def create_gaussian_kernel(radius, sigma):
    size = 2 * radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    sum_val = 0.0

    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            exponent = -(x**2 + y**2) / (2 * sigma**2)
            kernel[y + radius, x + radius] = np.exp(exponent)
            sum_val += kernel[y + radius, x + radius]

    kernel /= sum_val
    return kernel

def gaussian_blur(img, kernel_radius, sigma):
    kernel = create_gaussian_kernel(kernel_radius, sigma)
    return cv2.filter2D(img, -1, kernel)

def main(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("Could not open or find the image")
        return

    # Define Gaussian kernel parameters
    kernel_radius = 20  # Increased from 2 to 10
    sigma = 10.0        # Increased from 1.0 to 5.0

    # Apply Gaussian blur
    blurred_img = gaussian_blur(img, kernel_radius, sigma)

    # Save the output image
    output_path = "blurred_image.jpg"
    cv2.imwrite(output_path, blurred_img)
    print("Image blurring completed! Output saved as", output_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python gaussian_blur.py <image-path>")
    else:
        main(sys.argv[1])
