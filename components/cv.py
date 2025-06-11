import streamlit as st
import numpy as np
import cv2, io
from torchvision import transforms
from PIL import Image
from components.utilities import error

prototxt_path = "./models/colorization_deploy_v2.prototxt"
model_path = "./models/colorization_release_v2.caffemodel"
kernel_path = "./models/pts_in_hull.npy"

# prototxt_path = r"D:\Others\colorize-me\models\colorization_deploy_v2.prototxt"
# model_path = r"D:\Others\colorize-me\models\colorization_release_v2.caffemodel"
# kernel_path = r"D:\Others\colorize-me\models\pts_in_hull.npy"


# Reads image and returns as numpy array, and file name
def readImg(image):
    if image.name.count(".") != 1:
        error()
    a, b = image.name.split(".")
    return Image.open(image), (f"{a}_colorized.{b}")


def readNPArray(image):
    im = Image.fromarray(image)
    with io.BytesIO() as f:
        im.save(f, format="PNG")
        data = f.getvalue()
    return data


def resizeImg(image, sizeamt):
    return transforms.Resize(size=sizeamt)(Image.open(image))


def colorize(image):
    # Load the model (CNN) and points in hull
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path).transpose().reshape(2, 313, 1, 1)

    # Get model layers to apply to image
    model.getLayer(model.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    model.getLayer(model.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, dtype="float32")
    ]

    # Convert image numpy array & normalize RGB values between 0-1
    image2 = np.array(image.convert("RGB"))  # Ensure RGB
    normalize = image2.astype("float32") / 255.0

    # Convert RGB LAB (Louminousity L, and colors AB)
    LAB = cv2.cvtColor(normalize, cv2.COLOR_BGR2LAB)

    # Resize image down to 224x224 (models input size)
    rescale = cv2.resize(LAB, (224, 224))

    # Split out L from LAB to get the luminosity and apply colorization to it using the model
    L = cv2.split(rescale)[0]
    L -= 52
    model.setInput(cv2.dnn.blobFromImage(L))
    ab = model.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the output back to the original image size
    ab = cv2.resize(ab, (image2.shape[1], image2.shape[0]))
    L = cv2.split(LAB)[0]

    # Combine the output with the original image (& convert back to RGB)
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)

    # De-normalize pixel values back to between 0-255
    colorized = (255 * colorized).astype("uint8")

    return colorized
