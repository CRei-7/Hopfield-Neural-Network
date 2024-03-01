import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# import the image and extract
def imageGenerator(imageVector):
    cleanImage = np.where(imageVector == 1, 1, -1)  # Binarize input image
    noisyImage = cleanImage + np.random.normal(0, 2, cleanImage.shape)  # Adding Gaussian noise to the clean image
    noisyImage = np.where(noisyImage >= 0, 1, -1)  # Binarize noisy image
    return cleanImage, noisyImage

# Building up the coefficient matrix
def trainer(vector, oldCoefMat):
    vector = vector.flatten()
    n = len(vector)
    coefMat = np.zeros((n, n))

    if np.isscalar(oldCoefMat):
        for i in range(n):
            for j in range(n):
                if j <= i:
                    coefMat[i, j] = vector[i] * vector[j]
                else:
                    coefMat[i, j] = vector[j] * vector[i]

    elif oldCoefMat.shape == coefMat.shape:
        coefMat += np.outer(vector, vector)

    np.fill_diagonal(coefMat, 0)  # Zero out the diagonal
    return coefMat

def prediction(curruptedVec, coefMat):
    curruptedVec = curruptedVec.flatten()
    predictVec = np.sign(np.dot(coefMat, curruptedVec))  # Matrix-vector multiplication and binarization
    return predictVec.reshape((int(np.sqrt(len(predictVec))), int(np.sqrt(len(predictVec)))))

def image_to_np(path):
    image = Image.open(path).convert('L').resize((128, 128))  # Resize image to 128x128
    np_vector = np.asarray(image)
    np_vector = np.where(np_vector < 128, -1, 1)  # Binarize image
    return np_vector


# Import the images
plt.figure(figsize=(15, 10))
plt.subplots_adjust(hspace=0.5)
coefMatrix = 0  # Initialize coefficient matrix
png_files = [file for file in os.listdir('data') if file.endswith('.png')]
for i, png_file in enumerate(png_files, start=1):
    image = image_to_np(os.path.join('data', png_file))
    if i == 1:
        vector, noisyVec = imageGenerator(image)
        coefMatrix = trainer(vector, 0)
    else:
        vector, noisyVec = imageGenerator(image)
        coefMatrix = trainer(vector, coefMatrix)

    predictedVec = prediction(noisyVec, coefMatrix)

    plt.subplot(4, 4, (i-1)*4 + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Imported Picture {i}')
    plt.subplot(4, 4, (i-1)*4 + 2)
    plt.imshow(vector, cmap='gray')
    plt.title(f'Cleaned and Squared Picture {i}')
    plt.subplot(4, 4, (i-1)*4 + 3)
    plt.imshow(noisyVec, cmap='gray')
    plt.title(f'Noisy Picture {i}')
    plt.subplot(4, 4, (i-1)*4 + 4)
    plt.imshow(predictedVec, cmap='gray')
    plt.title(f'Recalled Picture {i}')

plt.savefig('MultiHopfield.png')
plt.clf()
plt.imshow(coefMatrix)
plt.savefig('CoefficientMatrix.png')
plt.title('Coefficient Matrix')
plt.show()