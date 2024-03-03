# Image Processing Algorithms

This repository contains a collection of image processing algorithms implemented in Python. The algorithms provided include dithering, color manipulation, and various blur filters.

## Table of Contents
- [Dithering Algorithms](#dithering-algorithms)
- [Color Manipulations](#color-manipulations)
- [Blur Filters](#blur-filters)
- [Usage](#usage)
- [Future Modifications](#future-modifications)

## Dithering Algorithms

The dithering algorithms provided in this repository include:
- **Threshold Dithering**: Applies a threshold value to each pixel to convert the image into black and white.
- **Floyd-Steinberg Error Diffusion Dithering**: Diffuses quantization errors to neighboring pixels.
- **Ordered Dithering**: Applies dithering using a predefined threshold matrix.
- **Pattern Dithering**: Applies dithering using a pattern matrix.
- **Cross-Dissolve**: Blends two images using a specified alpha value.
- **Dither-Dissolve**: Blends two images using dithering and a specified alpha value.

## Color Manipulations

The color manipulation functions provided include:
- **Inverted**: Produces the inverted image of the input.
- **BGR to RGB**: Converts the image from BGR to RGB color space.
- **RGB to Grayscale**: Converts the image to grayscale using the luminosity method.
- **BGR to Grayscale**: Converts the image from BGR to grayscale.
- **HSV to Grayscale**: Converts the image from HSV to grayscale using the Value component.
- **CMY to Grayscale**: Converts the image from cyan, magenta, and yellow (CMY) to grayscale.
- **CMYK to Grayscale**: Converts the image from CMYK to grayscale.
- **YCbCr to Grayscale**: Converts the image from YCbCr to grayscale by extracting the Y component.

## Blur Filters

The blur filters provided include:
- **Average Blur**: Applies an average blur filter to the image.
- **Gaussian Blur**: Applies a Gaussian blur filter to the image.
- **Median Blur**: Applies a median blur filter to the image.

## Usage

To use these algorithms, simply import the respective Python modules and call the appropriate functions with your input images.

```python
from algorithms.dithering_algorithms import *
from algorithms.color_manipulations import *
from algorithms.blur_filters import *

# Load an image
image_path = 'test.jpg'
img = cv2.imread(image_path)

# Apply a dithering algorithm
new_img = floyd(img, threshold=128)

# Display the processed image
plt.imshow(new_img, cmap='gray')
plt.title('Processed Image')
plt.show()

## Future Modifications
This repository is a work in progress, and more modifications and additional algorithms will be added in the future. 
