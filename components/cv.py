import streamlit as st
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

prototxt_path = "./models/colorization_deploy_v2.prototxt"
model_path = "./models/colorization_release_v2.caffemodel"
kernel_path = "./models/pts_in_hull.npy"

# prototxt_path = r"D:\Others\colorize-me\models\colorization_deploy_v2.prototxt"
# model_path = r"D:\Others\colorize-me\models\colorization_release_v2.caffemodel"
# kernel_path = r"D:\Others\colorize-me\models\pts_in_hull.npy"


def readImg(image):
    return Image.open(image)


def resizeImg(image, sizeamt):
    return transforms.Resize(size=sizeamt)(Image.open(image))
    # return Image.open(image)


def colorize(image):
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    # st.write(model.getLayerNames())

    # if model.empty():
    #     raise RuntimeError("Model failed to load.")

    # if model.getLayerId("conv8_313_rh") == -1:
    #     raise RuntimeError(
    #         "Layer 'conv8_313_rn' not found in model. Are you using the correct files?"
    #     )
    points = np.load(kernel_path).transpose().reshape(2, 313, 1, 1)
    model.getLayer(model.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    model.getLayer(model.getLayerId("conv8_313_rh")).blobs = [
        np.full([1, 313], 2.606, dtype="float32")
    ]
    # image2 = np.array(image)  # Ensure RGB
    image2 = np.array(image.convert("RGB"))  # Ensure RGB
    # image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV)
    bw_im = image2
    normalize = bw_im.astype("float32") / 255.0
    # Louminousity, and colors (A, B)
    LAB = cv2.cvtColor(normalize, cv2.COLOR_BGR2LAB)
    rescale = cv2.resize(LAB, (224, 224))
    L = cv2.split(rescale)[0]
    L -= 50
    model.setInput(cv2.dnn.blobFromImage(L))
    ab = model.forward()[0, :, :, :].transpose((1, 2, 0))
    # ab = cv2.resize(ab, bw_im.shape[1], bw_im.shape[0])
    ab = cv2.resize(ab, (bw_im.shape[1], bw_im.shape[0]))

    L = cv2.split(LAB)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = (255 * colorized).astype("uint8")
    return colorized
