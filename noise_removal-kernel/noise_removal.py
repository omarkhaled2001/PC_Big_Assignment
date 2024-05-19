import cv2
import numpy as np

def unsharp_mask(img, radius=2, amount=1.5, sigma=1.0):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (2*radius+1, 2*radius+1), sigma)

    # Calculate unsharp mask
    mask = img.astype(np.float32) - blurred.astype(np.float32)
    sharpened = img.astype(np.float32) + amount * mask

    # Clip values to [0, 255] range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    return sharpened

def main():
    # Read input image
    img = cv2.imread("input_image.jpg")

    # Apply unsharp masking
    sharpened_img = unsharp_mask(img)

    # Save output image
    cv2.imwrite("sharpened_image.jpg", sharpened_img)

if __name__ == "__main__":
    main()