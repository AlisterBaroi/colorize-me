import streamlit as st
from torchvision import transforms
from PIL import Image


def resizeImg(image, sizeamt):
    return transforms.Resize(size=sizeamt)(Image.open(image))


# import the required libraries
# import torch
# import torchvision.transforms as T
# from PIL import Image

# # read the input image
# img = Image.open("lounge.jpg")

# # compute the size(width, height) of image
# size = img.size
# print("Size of the Original image:", size)

# # define transformt o resize the image with given size
# transform = T.Resize(size=(250, 450))

# # apply the transform on the input image
# img = transform(img)
# print("Size after resize:", img.size)
