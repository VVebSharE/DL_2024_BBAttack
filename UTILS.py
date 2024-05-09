import torch
from PIL import Image
import os

def loadPertubation(pertPath):
    perturbation = torch.load(pertPath)
    return perturbation

def applyPertubation(image, pertubation):
    return image - pertubation

def toImage(tensor):
    numpy_array = tensor.cpu().numpy()

    # Rescale values to the range [0, 255]
    min_val = numpy_array.min()
    max_val = numpy_array.max()
    numpy_array = (255.0 * (numpy_array - min_val) / (max_val - min_val)).astype('uint8')

    # Convert NumPy array to PIL image
    image = Image.fromarray(numpy_array.transpose(1, 2, 0))  # Assuming tensor shape is (3, height, width)
    # convert to rgb
    image = image.convert("RGB")

    # Display the image
    return image

def saveImages(images_folder,samples,n=100):
    # Saving perturbed images

    if(not os.path.exists(images_folder)):
        os.makedirs(images_folder)

    for i in range(n):
        image = toImage(samples[i])
        image.save(os.path.join(images_folder,f"class1_{i}.png"))