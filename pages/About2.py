import streamlit as st

st.set_page_config(
    page_title="About",
    layout="centered",
)

st.title("*White Paper —*")

# Author and GitHub
author_text = """
**:green[Author:]** *Alister Animesh Baroi*  
**:green[GitHub Repo:]** *[https://github.com/AlisterBaroi/colorize-me](https://github.com/AlisterBaroi/colorize-me)*
"""
st.write(author_text)
st.divider()

# 1. Abstract
st.subheader("*:blue[1. Abstract —]*")
# st.write(
#     """
#     This project takes greyscale (or existing colour) images and automatically infers plausible colours using a feed-forward pass through a Convolutional Neural Network (CNN). The underlying model is based on the paper *:green["Colorful Image Colorization"]* as described by Zhang et al [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40). This white paper summarises:

#     1. The problem of single-image colourisation.
#     2. The network architecture and colour-space transformations.
#     3. Implementation details (code structure, dependencies, and Streamlit UI).
#     4. Example results and usage.
#     5. Conclusions and potential extensions.
#     """
# )
st.write(
    """
    This project addresses the challenge of restoring natural colour to greyscale images by leveraging a pre-trained Convolutional Neural Network (CNN) to predict chromatic information from luminance alone. In this work, any input image, whether a purely greyscale (Black & White) image or an existing colour image stripped of its chromatic channels, is converted into the CIELAB (also known as LAB) colour space, and only the lightness channel is fed into the network. The network has been trained to map each pixel's lightness to a plausible distribution over chroma values, enabling it to infer realistic colours that are spatially coherent across objects and scenes. After obtaining the predicted chroma layers, these are upscaled to the original image resolution and recombined with the original lightness, producing a fully colourised/recolourised output. This entire pipeline is wrapped in a lightweight Streamlit web interface, where users can upload an image, immediately view a side-by-side comparison of the original and the colourised result, and download the new image file. The result is a seamless, real-time application that demonstrates how the latest neural-based colourisation techniques can be packaged in Python to create an engaging, user-friendly experience. By combining deep learning inference, efficient colour-space transformations, and a simple web front end, this project serves both as a practical tool and as a portfolio piece showcasing end-to-end AI implementation in Python.
    """
)

# 2. Introduction & Background
st.subheader("*:blue[2. Introduction & Background —]*")

# st.write(
#     """
#     Greyscale images lack chromatic information, which can make them appear flat or less engaging. Restoring colour to a single greyscale image is inherently ambiguous, there are many possible plausible colour assignments for any given pixel. Traditional hand-tinting methods require significant manual effort, and
#     earlier automated approaches (e.g. example-based transfer) often rely on a reference colour image.

#     In 2016, Zhang et al. proposed a fully automatic, end-to-end CNN that, given only the lightness channel (L) in CIELAB space, predicts a probability distribution over possible :green['a'] and :green['b'] colour channels. By learning a broad mapping from :green[***L → (a, b)***], the network can produce realistic, richly coloured outputs with no user intervention [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40).
# """
# )
st.write(
    """
    Colourising greyscale images is a longstanding problem in computer vision, dating back several decades when practitioners attempted to hand-tint black-and-white photographs or develop rule-based algorithms. Modern approaches, however, rely on data-driven methods that learn plausible mappings from luminance to chrominance. In this project, we build upon the foundational work of Zhang et al. who introduced a fully automatic, end-to-end CNN that predicts colour (:green["A"] and :green["B"] channels in LAB space) from lightness (:green["L"] channel) alone [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40). Their method addresses the core ambiguities that arise when inferring colour, where, a single grey value may correspond to many possible real-world hues; and ensuring spatial consistency across complex scenes is non-trivial.

    This application adapts this CNN for rapid, user-friendly deployment via a Streamlit front end. Before describing the implementation, it is helpful to review key concepts and prior art:

    1. **The Ambiguity of Single-Image Colourisation**  
       - A single greyscale pixel could represent foliage, skin, sky or man-made objects.  
       - Traditional exemplar-based techniques transfer colour from a reference image, but require a suitable match and manual intervention.  
       - Zhang et al. (2016) overcame this by predicting a probability distribution over 313 quantised colour bins, thus capturing multiple plausible chroma options rather than committing to a single mapping.

    2. **CIELAB Colour Space and Its Advantages**  
       - Images are converted from RGB to CIELAB (or “LAB”), which separates lightness (L) from chromaticity (a, b).  
       - By restricting the CNN input to the L channel, the model focuses purely on inferring hue and saturation while preserving the original luminance structure.  
       - After inference, predicted a, b maps are merged with the input L channel, and converted back to RGB for display.

    3. **CNN Architecture (Zhang et al., 2016)**  
       - The network is based on a mid-level VGG-style encoder, followed by dilated convolutions to maintain spatial resolution.  
       - The final layer produces a 313-way softmax at each pixel location, representing discrete chroma clusters derived from a large image corpus.  
       - A rebalancing layer (“conv8_313_rh”) enables the model to place higher weight on rarer colours during training, reducing bias toward desaturated outputs.

    4. **Training Data and Pretrained Weights**  
       - Zhang et al. trained their model on millions of images from ImageNet and other large-scale datasets.  
       - We rely on their publicly available prototxt and caffemodel files, along with the “pts_in_hull.npy” cluster centres, which encapsulate the original training distribution (Zhang et al., 2016).

    5. **Software and Frameworks**  
       - **OpenCV (cv2.dnn):** Used to load the Caffe model, perform forward passes, and handle colour-space conversions.  
       - **Streamlit:** Provides a lightweight web framework for rapid prototyping. Users can upload images, view real-time before/after comparisons, and download the colourised output.  
       - **Torch and Torchvision:** Employed in `components/cv.py` for resizing and tensor transformations, though all core inference occurs in OpenCV’s DNN module.

    6. **Motivation for a Web-Based Demo**  
       - While many colourisation demos exist as standalone scripts or notebooks, embedding the pipeline within Streamlit allows immediate, zero-configuration interaction.  
       - This approach showcases an end-to-end AI deployment: from preprocessing and model inference to post-processing and user interaction, all in pure Python.

    By combining these elements (LAB conversions, a pretrained CNN, and a simple web interface) this project demonstrates how complex deep-learning pipelines can be distilled into a real-time, user-friendly application. In the following sections, we delve into the problem statement, detailed methodology, and implementation specifics.
    """
)

# 3. Problem Description
st.subheader("*:blue[3. Problem Description —]*")
st.write(
    """
    **Objective:** Given an input that is either:
    1. A truly greyscale (lab) image  
    2. A colour image (RGB) whose chroma we want to re-infer from its luminosity

    …produce a colourised RGB image that looks natural.  

    **Key challenges include:**
    - **Ambiguity of colour:** A neutral grey pixel could correspond to many real-world colours (e.g. sky, metal,
      foliage).  
    - **Spatial consistency:** Neighbouring pixels must have coherent colours (trees should all be green, skin tones
      consistent, etc.).  
    - **High-resolution inference:** The network takes a 224×224-pixel input; results must be upscaled to
      arbitrary image sizes without visible artefacts.
    """
)

# 4. Methodology / Solution
st.subheader("*:blue[4. Methodology / Solution —]*")
st.write(
    """
    **4.1. Colour Space Conversion (RGB ↔ CIELAB)**  
    - Input: The user’s uploaded file (JPEG/PNG) is read via PIL and converted to an RGB NumPy array.  
    - Normalisation: Pixel values are scaled to [0, 1] (float32).  
    - Conversion: Using OpenCV, RGB → CIELAB (L, a, b). Only the L (lightness) channel is retained for the CNN.

    **4.2. CNN Architecture (Zhang et al., 2016)**  
    - The network was trained on millions of images to predict, for each spatial location, a 313-way quantised
      distribution over “a” and “b” colour bins.  
    - Inference:  
      1. Resize L-channel to 224×224.  
      2. Subtract 50–52 (mean centring) to normalise for network statistics.  
      3. Feed into `cv2.dnn.readNetFromCaffe` (using `colorization_deploy_v2.prototxt` and
         `colorization_release_v2.caffemodel`).  
      4. The two special layers—`class8_ab` (313 bins) and `conv8_313_rh` (re-balance layer)—are manually
         injected with pretrained “points_in_hull.npy” to guide the prediction.  
      5. The raw network output is an (H × W × 2) map of a,b values (float32).  

    **4.3. Post-processing**  
    - Resize the predicted (a, b) map from 224×224 back to the original image’s width × height using
      bilinear interpolation.  
    - Concatenate the upscaled a, b channels with the original L channel.  
    - Convert back from CIELAB → RGB with OpenCV.  
    - De-normalise: Multiply by 255, clip to [0, 255], and cast to uint8.  

    **4.4. Web-app Front End (Streamlit)**  
    - **File Upload:** Users can upload `.jpg`, `.jpeg`, `.png`. The function `readImg()` (in `components/cv.py`)
      checks for exactly one “.” in the filename; otherwise, a Streamlit dialog signals an error.  
    - **Image Comparison:** The original (L or RGB) vs. colourised output is displayed side by side via
      `streamlit_image_comparison`.  
    - **Download:** After inference, users click “Download” to save the result as `<original_name>_colorized.png`.  
    - **Layout:** A single‐column centred layout with a title, description, upload area, and comparison widget.
    """
)

# 5. Implementation Details
st.subheader("*:blue[5. Implementation Details —]*")
st.write(
    """
    **5.1. Project Structure**  
    ```
    colorize-me/
    ├── components/
    │   └── cv.py
    ├── models/
    │   ├── colorization_deploy_v2.prototxt
    │   ├── colorization_release_v2.caffemodel
    │   └── pts_in_hull.npy
    ├── Colorize_Me.py
    ├── About.py
    └── requirements.txt
    ```

    **5.2. `components/cv.py`**  
    - **Imports:** `streamlit` (for error dialogues), `numpy`, `cv2` (OpenCV), `io`, `torchvision.transforms`, `PIL.Image`.  
    - **Constants:**  
      ```python
      prototxt_path = "./models/colorization_deploy_v2.prototxt"
      model_path   = "./models/colorization_release_v2.caffemodel"
      kernel_path  = "./models/pts_in_hull.npy"
      ```  
    - **Functions:**  
      1. `error()`: Displays an upload error if the filename has multiple “.” characters.  
      2. `readImg(image)`: Checks filename, splits name/extension, opens via PIL, returns `(PIL.Image, new_filename)`.  
      3. `readNPArray(image)`: Given a NumPy array, re-encodes to PNG bytes so `st.download_button` can stream it.  
      4. `resizeImg(image, sizeamt)`: Uses `torchvision.transforms.Resize` to create a thumbnail for previews.  
      5. `colorize(image)`:  
         - Loads Caffe model into an OpenCV DNN.  
         - Injects the `pts_in_hull.npy` into the special prototype layers.  
         - Converts the input to CIELAB and isolates L.  
         - Runs a forward pass to get predicted a,b.  
         - Upscales a,b to original resolution, recombines with L, converts back to RGB.  
         - Returns a `uint8` RGB NumPy array.

    **5.3. `Colorize_Me.py` (Streamlit Front End)**  
    - **Page Config:** Title “Colorize Me” with centred layout.  
    - **Title & Description:** Explains the use of the Zhang et al. (2016) feed-forward CNN to colourise B&W or
      recolour existing RGB.  
    - **Uploader Widget:**  
      ```python
      uploaded_file = st.file_uploader("Choose an image to colorize...", type=["jpg", "png", "jpeg"])
      ```  
    - **Main Logic:**  
      ```python
      if uploaded_file:
          readNP, fileName = readImg(uploaded_file)
          c = colorize(readNP)
          # Display original vs. colorized side by side
          image_comparison(
              img1=readNP,
              img2=c,
              label1="Original",
              label2="Colorized",
              width=550,
              starting_position=50,
              show_labels=True,
              make_responsive=True,
              in_memory=True,
          )
          # Download button
          st.download_button("Download", data=readNPArray(c),
                             mime="image/png", file_name=fileName,
                             use_container_width=True)
      ```
    - **Dependencies (from `requirements.txt`):**  
      ```
      numpy
      opencv-python
      pillow
      torchvision
      torch
      streamlit
      streamlit-image-comparison
      ```  
      (Plus other transitive dependencies listed; see full `requirements.txt`.)

    **5.4. Model Files (not shown)**  
    - **`colorization_deploy_v2.prototxt`:** Defines the CNN architecture layers.  
    - **`colorization_release_v2.caffemodel`:** Contains pre-trained weights (Zhang et al., 2016).  
    - **`pts_in_hull.npy`:** A 313×2 array of cluster centres in a,b space (used to “soft-encode” predictions).
    """
)

# 6. Results and Demonstration
st.subheader("*:blue[6. Results & Demonstration —]*")
st.write(
    """
    Below are a few illustrative examples (replace these with your own screenshots or live Streamlit demo links):

    1. **Classic Greyscale Portrait:**  
       - *Input:* A greyscale photograph of a person.  
       - *Output:* Natural skin-tones, subtle shading, and realistic hair colour.  

    2. **Landscape Scene:**  
       - *Input:* A greyscale outdoor scene (trees, sky, water).  
       - *Output:* Rich greens for foliage, blue sky gradient, and water reflections.  

    3. **Recolouring a Colour Photo:**  
       - *Input:* A full-colour image.  
       - *Output:* The model discards original a,b channels, infers fresh colours from L (e.g. slight hue shifts).  

    > *Tip:* If you have a public Streamlit share link or GIFs of the “before/after” widget, embed them below.
    """
)

# 7. Conclusion
st.subheader("*:blue[7. Conclusion —]*")
st.write(
    """
    “Colorize Me” demonstrates how a pretrained CNN (Zhang et al., 2016) can be wrapped in a lightweight
    Streamlit interface to provide real-time single-image colourisation. Key takeaways:

    - **Simplicity:** With just three model files (prototxt, caffemodel, pts_in_hull), we can produce
      high-quality full-colour outputs in under a second (on modern hardware).  
    - **Extensibility:** Future work could integrate user controls (e.g. “scribble” hints), try alternative
      colourisation networks (GAN-based, U-Net), or deploy via Flask/Django for production.  
    - **Education & Portfolio:** This live demo serves both as an illustration of deep learning pipelines (colour
      space conversions, DNN inference) and a portfolio piece showcasing Python + AI expertise.
    """
)

# 8. References
st.subheader("*:blue[8. References —]*")
# st.write(
#     """
#     1. **Zhang, R., Isola, P., & Efros, A. A. (2016).** Colorful Image Colorization. In *European Conference on
#        Computer Vision (ECCV)*, Springer. DOI: [10.1007/978-3-319-46487-9_40](https://doi.org/10.1007/978-3-319-46487-9_40).
#     2. **GitHub Repo (Zhang et al.):**
#        https://github.com/richzhang/colorization (accessed June 2025).
#     3. **OpenCV Documentation:** Colour space conversion (cv2.cvtColor).
#     4. **Streamlit:** https://docs.streamlit.io/ (accessed June 2025).
#     """
# )
st.markdown(
    """    
    ###### :green[[1]] **R. Zhang, P. Isola, and A. A. Efros**. "Colorful Image Colorization", in *Computer Vision* -- ECCV 2016, 2016, pp. 649-666, doi: [10.1007/978-3-319-46487-9_40](https://doi.org/10.1007/978-3-319-46487-9_40).
    
    ###### :green[[2]] richzhang, "GitHub - richzhang/colorization: Automatic colorization using deep neural networks. 'Colorful Image Colorization.' In ECCV, 2016.," GitHub, available: [github.com/richzhang/colorization](https://github.com/richzhang/colorization).
    """,
)

# st.subheader(
#     """
#     [1] **R. Zhang, P. Isola, and A. A. Efros**. "Colorful Image Colorization", in *Computer Vision* -- ECCV 2016, 2016, pp. 649-666, doi: [10.1007/978-3-319-46487-9_40](https://doi.org/10.1007/978-3-319-46487-9_40).
#     """,
#     anchor="ref1",
# )
# st.subheader(
#     """
#     [2] richzhang, "GitHub - richzhang/colorization: Automatic colorization using deep neural networks. 'Colorful Image Colorization.' In ECCV, 2016.," GitHub, available: [github.com/richzhang/colorization](https://github.com/richzhang/colorization).
#     """,
#     anchor="ref2",
# )
# st.subheader(
#     """
#     [2]

#     richzhang, "GitHub - richzhang/colorization: Automatic colorization using deep neural networks. 'Colorful Image Colorization.' In ECCV, 2016.," GitHub, available: [github.com/richzhang/colorization](https://github.com/richzhang/colorization).
#     """,
#     anchor="ref2",
# )
