'''
imge color manipulations functions
Convert RGB imgs to grayscale by formula:
0.299 * Red + 0.587 * Green + 0.114 * Blue
'''
import numpy as np

def inverted(img):
    '''Equivalent to cv2.bitwise_not(img)'''
    return -img

def BGRtoRGB(img):
    # Swap blue and red channels
    RGBimg = img[:, :, ::-1]
    return RGBimg

def RGBtograyscale(img):
    '''RGB to Grayscale
       Eqivalent to cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    '''

    # Extract the height and width of the input image
    img_height, img_width, _ = img.shape

    # Create an empty grayscale image with the same dimensions
    grayscale_img = np.zeros((img_height, img_width))

    # Iterate over each pixel in the image
    for i in range(img_height):
        for j in range(img_width):
            # Convert RGB to grayscale using the weighted average
            grayscale_img[i][j] = 0.299 * img[i][j][0] + 0.587 * img[i][j][1] + 0.114 * img[i][j][2]

    # Return the resulting grayscale image
    return grayscale_img

def BGRtograyscale(img):
        '''Equivalent to cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)'''
        return RGBtograyscale(BGRtoRGB(img))

def HSVtograyscale(img):
    '''HSV to Grayscale
       Equivalent to cv2.cvtColor(img, cv2.COLOR_HSV2GRAY)
    '''
    # Convert HSV image to grayscale using the Value component
    grayscale_img = img[:,:,2]
    return grayscale_img

def CMYtograyscale(img):
    '''cyan, magenta, and yellow (CMY) to Grayscale'''
    # Extract the height and width of the input image
    img_height, img_width, _ = img.shape

    # Create an empty grayscale image with the same dimensions
    grayscale_img = np.zeros((img_height, img_width))

    # Iterate over each pixel in the image
    for i in range(img_height):
        for j in range(img_width):
            # Convert CMY to grayscale using the formula: Grayscale = 1 - max(C, M, Y)
            grayscale_img[i][j] = 1 - max(img[i][j][0], img[i][j][1], img[i][j][2])

    # Return the resulting grayscale image
    return grayscale_img

def CMYKtograyscale(img):

    # Extract the height and width of the input image
    img_height, img_width, _ = img.shape

    # Create an empty grayscale image with the same dimensions
    grayscale_img = np.zeros((img_height, img_width))

    # Iterate over each pixel in the image
    for i in range(img_height):
        for j in range(img_width):
            c, m, y, k = img[i, j, 0], img[i, j, 1], img[i, j, 2], img[i, j, 3]
            
            # Convert CMYK to RGB
            r = (1 - c) * (1 - k)
            g = (1 - m) * (1 - k)
            b = (1 - y) * (1 - k)

            # Convert RGB to grayscale
            grayscale_img[i, j] = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Return the resulting grayscale image
    return grayscale_img

def YCbCrtoGrayscale(img):
    '''Convert YCbCr image to grayscale by extracting the Y component'''
    # Convert YCbCr image to grayscale using the Y component
    grayscale_img = img[:,:,0]
    return grayscale_img