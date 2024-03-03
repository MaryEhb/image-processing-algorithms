"""Image Dithering Functions"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

def threshold_dithering(img, threshold=128):
    """simplest dithering that applies a threshold value to each pixal"""
    new_img = (img >= threshold) * 255
    return new_img

def floyd(img, threshold=128, alpha=7/16, beta=3/16, gamma=5/16, delta=1/16):
    """Applies Floyd-Steinberg Error Diffusion Dithering to the input image."""

    img_height, img_width = img.shape
    new_img = img.copy()

    for i in range(img_height):
        for j in range(img_width):

            if (new_img[i][j] >= threshold):
                err = new_img[i][j] - 255
                new_img[i][j] = 255
            else:
                err = new_img[i][j]
                new_img[i][j] = 0

            if j < img_width - 1:
                new_img[i][j + 1] +=  err * alpha

            if i < img_height - 1:

                new_img[i + 1][j] += err * gamma

                if j != 0:
                    new_img[i + 1][j - 1] += err * beta

                if j < img_width - 1:
                    new_img[i + 1][j + 1] += err * delta 

    return new_img

def ordered_dithering(img, threshold_frame):
    """Applies ordered dithering to the input image using a given threshold matrix."""

    img_height, img_width = img.shape
    threshold_height, threshold_width = threshold_frame.shape
    new_img = img.copy()

    for i in range(img_height):
        for j in range(img_width):
            i_filter = i % threshold_height
            j_filter = j % threshold_width

            if new_img[i][j] >= threshold_frame[i_filter][j_filter]:
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0
    return new_img

def pattern_dithering(img, n=2):
    matrix_n_2 = [[255, 255, 255, 255], [255, 0, 255, 255], [255, 0, 0, 255], [0, 0, 0, 255], [0, 0, 0, 0]]
    
    img_height, img_width = img.shape
    new_img = img.copy()

    for i in range(0, img_height, n):
        for j in range(0, img_width, n):
            # Calculate the end indices for the block within the image bounds
            end_i = min(i + n, img_height)
            end_j = min(j + n, img_width)

            # Loop within the bounds of the image
            matrix_value = 0
            for x in range(i, end_i):
                for y in range(j, end_j):
                    matrix_value += new_img[x, y]
            
            matrix_value = matrix_value / ((end_i - i) * (end_j - j) * 255)
            
            # Determine the pattern index
            if matrix_value < 0.2:
                k = 0
            elif matrix_value < 0.4:
                k = 1
            elif matrix_value < 0.6:
                k = 2
            elif matrix_value < 0.8:
                k = 3
            else:
                k = 4

            # Loop again within the bounds of the image to apply the pattern
            for x in range(i, end_i):
                for y in range(j, end_j):
                    new_img[x, y] = matrix_n_2[k][x % n * n + y % n]

    return new_img

def cross_dissolve(img1, img2, alpha):
    """Performs cross-dissolve between two input images based on the given alpha value."""
    # Check if the dimensions of the images match
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions for cross-dissolve.")

    img_height, img_width = img1.shape
    new_img = np.zeros((img_height, img_width))

    for i in range(img_height):
        for j in range(img_width):
            new_img[i][j] = (1 - alpha) * img1[i][j] + alpha * img2[i][j] 

    return new_img

def dither_dissolve(img1, img2, alpha):
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions for dither-dissolve.")

    img_height, img_width = img1.shape
    new_img = img1.copy()
    img2_limit = int(alpha * img_width)

    for i in range(img_height):
        for j in range(img2_limit):
            new_img[i][j] = img2[i][j]
    return new_img

if __name__ == '__main__':
    # Load an image
    image_path = 'test.jpg'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    # Choose the algorithm
    algorithm_choice = input("Choose the algorithm:\n1) Floyd-Steinberg\n2) Ordered Dithering\n3) pattern Dithering\n4) Cross-Dissolve\n5) dither_cross_dissolve\n ")

    if algorithm_choice == '1':
        # Apply Floyd-Steinberg dithering
        threshold = float(input('Enter the value of threshold (between 0 and 255): '))
        new_img = floyd(img, threshold, alpha=7/16, beta=3/16, gamma=5/16, delta=1/16)
        title = 'Floyd-Steinberg Dithering'
    elif algorithm_choice == '2':
        # Apply ordered dithering (provide a threshold matrix)
        threshold_frame = np.array([[20, 150], [80, 120]])
        new_img = ordered_dithering(img, threshold_frame)
        title = 'Ordered Dithering'
    elif algorithm_choice == '3':
        new_img = pattern_dithering(img)
        title = 'pattern Dithering'
    elif algorithm_choice == '4':
        # Apply cross-dissolve
        second_image_path = 'test2.jpg'
        img2 = cv2.imread(second_image_path, cv2.IMREAD_GRAYSCALE)
        alpha = float(input("Enter the alpha value for cross-dissolve (between 0 and 1): "))
        new_img = cross_dissolve(img, img2, alpha)
        title = 'Cross-Dissolve'
    elif algorithm_choice == '5':
        second_image_path = 'test2.jpg'
        img2 = cv2.imread(second_image_path, cv2.IMREAD_GRAYSCALE)
        alpha = float(input("Enter the alpha value for cross-dissolve (between 0 and 1): "))
        new_img = dither_dissolve(img, img2, alpha)
        title = 'dither_dissolve'

    # Display the processed image
    plt.subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray')
    plt.title(title)

    # Print the array matrix of the new image
    print("Array matrix of the new image:")
    print(new_img)

    # Show the plots
    plt.show()