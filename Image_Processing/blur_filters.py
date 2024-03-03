'''blur filters: avrage / gaussian / median'''
import numpy as np

def avg_blur(img, n=3):
    '''return blur image by avraging algorithm'''
    imgHeight, imgWidth = img.shape[:2]
    filter = np.ones((n, n), dtype=float) / (n * n) # Normalized averaging filter

    if len(img.shape) == 2:  # Grayscale image
        blurImg = np.zeros((imgHeight - n + 1, imgWidth - n + 1))
    
        for i in range(n - 1, imgHeight - n + 1):
            for j in range(n - 1, imgWidth - n + 1):
                block = [[img[x][y] for x in range(i - (n - 1) // 2, i + (n - 1) // 2 + 1)] for y in range(j - (n - 1) // 2, j + (n - 1) // 2 + 1)]
                blurImg[i - n + 1, j - n + 1] = np.sum(np.multiply(block, filter))

    elif img.shape[2] and len(img.shape) == 3: #colored image
        num_channels = img.shape[2]
        blurImg = np.zeros((imgHeight - n + 1, imgWidth - n + 1, num_channels), dtype=np.uint8)

        for c in range(num_channels):
            for i in range(imgHeight - n + 1):
                for j in range(imgWidth - n + 1):
                    block =  [[img[x][y][c] for x in range(i - (n - 1) // 2, i + (n - 1) // 2 + 1)] for y in range(j - (n - 1) // 2, j + (n - 1) // 2 + 1)]
                    blurImg[i, j, c] = np.sum(np.multiply(block, filter))
    else:
        raise ValueError("Unsupported image format")

    return blurImg

def gaussian_blur(img):
    '''Equivalent to cv2.GaussianBlur(img,(3,3),0)'''
    pass

def median_blur(img):
    pass