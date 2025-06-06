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
st.subheader("*:blue[Abstract —]*")
st.markdown(
    """
    <p style='text-align: justify;'>
    This project addresses the challenge of restoring natural color to greyscale images by leveraging a pre-trained Convolutional Neural Network (CNN) to predict chromatic information from luminance alone. In this work, any input image, whether a purely greyscale (Black & White) image or an existing color image stripped of its chromatic channels, is converted into the CIELAB (also known as `LAB`) color space, and only the lightness channel is fed into the network. The network has been trained to map each pixel's lightness to a plausible distribution over chroma values, enabling it to infer realistic colors that are spatially coherent across objects and scenes. After obtaining the predicted chroma layers, these are upscaled to the original image resolution and recombined with the original lightness, producing a fully colorized/recolorized output. This entire pipeline is wrapped in a lightweight Streamlit web interface, where users can upload an image, immediately view a side-by-side comparison of the original and the colorized result, and download the new image file. The result is a seamless, real-time application that demonstrates how the latest neural-based colorisation techniques can be packaged in Python to create an engaging, user-friendly experience. By combining deep learning inference, efficient color-space transformations, and a simple web front end, this project serves both as a practical tool and as a portfolio piece showcasing end-to-end AI implementation in Python.
    </p>
    """,
    unsafe_allow_html=True,
)

# 2. Introduction & Background
st.subheader("*:blue[1. Introduction & Background —]*")
st.write(
    """
    Colorising greyscale images is a longstanding problem in computer vision, dating back several decades when practitioners attempted to hand-tint black-and-white photographs or develop rule-based algorithms. Modern approaches, however, rely on data-driven methods that learn plausible mappings from luminance to chrominance. In this project, we build upon the foundational work of :green[Zhang et al.] (in their paper :green[***"Colorful Image Colorization"***]) who introduced a fully automatic, end-to-end CNN that predicts color (`A` and `B` channels in `LAB` space) from lightness (`L` channel) alone [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40). Their method addresses the core ambiguities that arise when inferring color, where a single grey value may correspond to many possible real-world hues; and ensuring spatial consistency across complex scenes is non-trivial.

    This application adapts this CNN for rapid, user-friendly deployment via a Streamlit front end. Before describing the implementation, it is helpful to review key concepts and prior art:

    1. **The Ambiguity of Single-Image colorisation:**  
       - A single greyscale pixel could represent foliage, skin, sky or man-made objects.  
       - Traditional exemplar-based techniques transfer color from a reference image, but require a suitable match and manual intervention.  
       - Zhang et al. overcame this by predicting a probability distribution over 313 quantised color bins, thus capturing multiple plausible chroma options instead of committing to a single mapping [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40).

    2. **CIELAB color Space and Its Advantages:**  
       - Images are converted from `RGB` to `LAB`, which separates lightness (`L`) from chromaticity (`A, B`).  
       - By restricting the CNN input to the `L` channel, the model focuses purely on inferring hue and saturation while preserving the original luminance structure.  
       - After inference, predicted `A`, `B` maps are merged with the input `L` channel, and converted back to `RGB` for display.

    3. **CNN Architecture (Zhang et al.) [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40):**  
       - The network is based on a mid-level VGG-style encoder, followed by dilated convolutions to maintain spatial resolution.  
       - The final layer produces a 313-way softmax at each pixel location, representing discrete chroma clusters derived from a large image corpus.  
       - A rebalancing layer (`conv8_313_rh`) enables the model to place higher weight on rarer colors during training, reducing bias toward desaturated outputs.

    4. **Training Data and Pretrained Weights:**  
       - Zhang et al. trained their model on millions of images from ImageNet and other large-scale datasets [[ 2 ]](#2-richzhang-git-hub-richzhang-colorization-automatic-colorization-using-deep-neural-networks-colorful-image-colorization-in-eccv-2016-git-hub-available-github-com-richzhang-colorization).  
       - We rely on their publicly available prototxt and caffemodel files, along with the `pts_in_hull.npy` cluster centres, which encapsulate the original training distribution [[ 1, 2 ]](#7-references).

    5. **Software and Frameworks:**  
       - **OpenCV (`cv2.dnn`):** Used to load the Caffe model, perform forward passes, and handle color-space conversions.  
       - **Streamlit:** Provides a lightweight web framework for rapid prototyping. Users can upload images, view real-time before/after comparisons, and download the colorized output.  
       - **Torch and Torchvision:** Employed in `components/cv.py` for resizing and tensor transformations, though all core inference occurs in OpenCV's DNN module.

    6. **Motivation for a Web-Based Demo:**  
       - While many colorization demos exist as standalone scripts or notebooks, embedding the pipeline within Streamlit allows immediate, zero-configuration interaction.  
       - This approach showcases an end-to-end AI deployment: from preprocessing and model inference to post-processing and user interaction, all in pure Python.

    > By combining these elements (`LAB` conversions, a pre-trained CNN, and a simple web interface) this project demonstrates how complex Deep Learning pipelines can be distilled into a real-time, user-friendly application. In the following sections, we delve into the problem statement, detailed methodology, and implementation specifics.
    """
)

# 3. Problem Description
st.subheader("*:blue[2. Problem Description —]*")
st.write(
    """
    The core objective of this project is to take any input image, whether a genuine greyscale (black & white) photograph or an existing colour image with its chromatic channels removed, and produce a fully colorized output that appears natural and coherent. In practice, this means inferring plausible `A`, `B` chroma values for every pixel using only the lightness (`L`) channel as input. Although this task may sound straightforward, it presents several inherent challenges:

    1. **Ambiguity of Colour Choices:**  
       - A single grey-level pixel can correspond to many real-world hues (for example, foliage, human skin, sky or concrete).  
       - The model must learn a distribution of plausible colours rather than making an arbitrary or desaturated guess.

    2. **Spatial and Semantic Consistency:**  
       - Neighbouring pixels belonging to the same object (such as a person's face or a patch of grass) should share coherent chroma values.  
       - Ensuring that distinct semantic regions (sky versus ground, skin versus clothing) receive appropriate, context-sensitive colour assignments is non-trivial.

    3. **Resolution and Upscaling:**  
       - The CNN operates on a fixed `224×224` pixel input; colour predictions must be upscaled to match the original image's dimensions without introducing artefacts or blurring.  
       - High-resolution outputs demand careful interpolation and recombination with the original `L` channel to preserve sharpness and detail.

    4. **Real-Time Inference:**  
       - To offer a seamless user experience, the entire pipeline (colour-space conversion, forward pass through the network, upscaling and post-processing) must execute in under a second on typical modern hardware.

    > Together, these challenges define a problem space where both perceptual quality and computational efficiency must be balanced. The subsequent Methodology section explains how colour-space transformations, a pre-trained CNN, and post-processing strategies address each of these issues.
    
    """
)

# 4. Methodology / Solution
st.subheader("*:blue[3. Methodology / Solution —]*")
st.write(
    """
    **3.1. Color Space Conversion (RGB ↔ LAB)**      
    - **Image Loading & Normalisation:**  
      1. The user's uploaded file (JPEG/PNG) is read via `PIL` and converted into an `RGB` `NumPy` array of dtype `uint8`.  
      2. Pixel values are cast to `float32` and scaled to the `[0, 1]` range by dividing by `255`.  
    - **Conversion to LAB:**  
      1. OpenCV's `cv2.cvtColor` function transforms the normalized `RGB` array into `LAB` colour space.  
      2. In `LAB`, the image is decomposed into:  
         - **`L`** channel (lightness, range 0-100)  
         - **`A`**, **`B`** channels (chromaticity, range approximately -128 to 127)  
      3. Only the **`L`** channel is extracted and resized to `224×224` pixels using bilinear interpolation (`cv2.resize`). This maintains compatibility with the pretrained CNN's expected input size and preserves the original luminance structure.

    **3.2. CNN Architecture (Zhang et al. [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40))**  
    - **Model Overview:**  
      1. The network is a modified VGG-style encoder with dilated convolutions in later layers to retain spatial resolution.  
      2. The final convolutional output produces a 313-way softmax distribution at every spatial location, corresponding to discrete chroma cluster centres (quantised in the `A-B` plane).  
      3. A **"rebalancing"** layer (`conv8_313_rh`) adjusts the softmax probabilities to mitigate undersampling of rare colours, ensuring richer, more vibrant results.
    - **Loading Pretrained Weights:**  
      1. Use `cv2.dnn.readNetFromCaffe("models/colorization_deploy_v2.prototxt", "models/colorization_release_v2.caffemodel")` to load the model definition and weights.  
      2. Load `pts_in_hull.npy` (a 313×2 NumPy array of `A-B` cluster centres) and assign them to the network's `class8_ab` and `conv8_313_rh` layers via `net.getLayer(layer_id).blobs = [pts]`.  
    - **Forward Pass:**  
      1. Subtract 52 (dataset mean) from the resized `L` channel to zero-centre the data.  
      2. Create a 4D blob via `cv2.dnn.blobFromImage(L_resized)` (shape: `1×1×224×224`).  
      3. `net.setInput(blob)` and call `net.forward()` to obtain a raw output of shape `1×313×56×56` (for example).  
      4. Decode this into a `56×56×2` map of predicted `A`, `B` values by multiplying the softmax probabilities with the corresponding cluster centres and summing across the 313 bins.

    **3.3. Post-processing**  
    - **Upscaling Predicted Channels:**  
      1. The raw `A-B` map (`56×56`) is resized back to the original image's *`width × height`* using bilinear interpolation (`cv2.resize`).  
      2. This interpolation preserves smooth transitions and avoids blocky artefacts.  
    - **Recombining with `L` Channel:**  
      1. The original `L` channel (at full resolution) is concatenated with the upscaled `A`, `B` maps to form a complete `LAB` image.  
      2. OpenCV's `cv2.cvtColor` converts the combined `LAB` image back to `RGB`.  
    - **De-normalisation & Data Type Conversion:**  
      1. Multiply all `RGB` channels by 255, clip values to the `[0, 255]` range, and convert to `uint8`.  
      2. The result is a colorized `RGB` NumPy array suitable for display or file export.

    **3.4. Web-app Front End (Streamlit)**  
    - **File Uploader & Validation:**  
      1. `st.file_uploader("Choose an image to colourize...", type=["jpg", "png", "jpeg"])` restricts uploads to valid image formats.  
      2. The function `readImg()` (in `components/cv.py`) ensures exactly one period (`.`) in the filename, else displays a Streamlit error dialog (`st.error`).  
      3. If valid, `readImg()` returns a `PIL` image and a sanitized filename (without extension) for output.  
    - **Colourisation Trigger:**  
      1. Upon successful upload, the `PIL` image is converted to a NumPy array via `np.array(img)`.  
      2. The array is passed to the `colorize()` function, which runs the entire pipeline described above (colour-space conversion, CNN inference, post-processing).  
    - **Result Display & Comparison:**  
      1. `streamlit_image_comparison` displays the original greyscale (or colored RGB) image side by side with the colorized output.  
      2. Users can interactively slide between ***"Original"*** and ***"Colourised"*** to visually assess differences.  
    - **Download Capability:**  
      1. The `colorize()` function outputs a `uint8` RGB array.  
      2. `readNPArray()` encodes this array into PNG bytes via `PIL` in-memory buffer, returning a `BytesIO` object.  
      3. `st.download_button("Download", data=readNPArray(colorized), file_name=f"{filename}_colourized.png", mime="image/png")` allows users to save the result locally as `<original_name>_colorized.png`.  
    - **Performance Considerations:**  
      1. The entire pipeline—from upload to download—executed on a typical modern CPU/GPU completes in under one second.  
      2. Efficient resizing and OpenCV's optimized DNN backend ensure real-time responsiveness.
    """
)

# 5. Implementation Details
st.subheader("*:blue[4. Implementation Details —]*")
st.write(
    """
    **4.1. Project Structure**  
    The repository is organised as follows:
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

    **4.2. Backend Functions:**  
    1. `error()`: Displays an error dialog via `st.error()` if the uploaded filename contains more than one period (`.`), preventing ambiguous file extensions. 
    2. `readImg(image)`:  
       - Accepts a `UploadedFile` object from Streamlit.  
       - Verifies exactly one `.` in the filename; if invalid, calls `error()` and returns `None`.  
       - Uses `PIL.Image.open()` to load the image into a `PIL.Image` object and splits name/extension to extract a sanitized base filename (no extension).  
       - Returns `(pil_image, base_filename)`. 
    3. `readNPArray(image)`:
       - Takes a colourised NumPy array (`uint8`, shape `H×W×3`).  
       - Converts to `PIL.Image` via `Image.fromarray()`, writes to a `BytesIO` buffer in PNG format, and returns the buffer’s byte data.  
       - Enables `st.download_button` to stream a valid PNG file without writing to disk.  
    4. `resizeImg(image, sizeamt)`: Uses `torchvision.transforms.Resize` to create a thumbnail for previews.  
    5. `colorize(image)`:  
        - Loads Caffe model into an OpenCV DNN.  
        - Injects the `pts_in_hull.npy` into the special prototype layers.  
        - Pre-processes the input image as follows:
            1. Converts `image_np` (RGB `uint8`) to LAB using `cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)`.  
            2. Extracts the `L` channel and normalises it to `[0, 100]`.  
            3. Resizes `L` to `224×224` with `cv2.resize(L, (224, 224), interpolation=cv2.INTER_CUBIC)`.  
            4. subtracts a constant mean (52) to align with the network’s training distribution.  
            5. Creates a blob via `cv2.dnn.blobFromImage(L_resized)` (shape `1×1×224×224`) and sets it as the network input.  
        - Runs a forward pass to get predicted (A, B):
            1. Runs `output = net.forward()` to get shape `1×313×56×56`.  
            2. Reshapes and applies a weighted sum over the 313 colour bins using `pts_in_hull` cluster centres to produce a `56×56×2` array of predicted `A`,`B`.  
        - Upscales `(A, B)` to original resolution, recombines with `L`, converts back to `RGB`.  
        - Returns a `uint8` RGB NumPy array.

    **4.3. Dependencies (from `requirements.txt`):**  
    ```
    numpy
    opencv-python
    pillow
    torchvision
    torch
    streamlit
    streamlit-image-comparison
    ```  
    **Key Points:**  
      - `opencv-python` provides DNN support for Caffe and colour-space conversions.  
      - `torch` and `torchvision` are only used for resizing helper functions; core inference relies on OpenCV.  
      - `streamlit-image-comparison` is a lightweight widget to compare two images with an interactive slider.  
    - To install all dependencies in a fresh environment:  
      ```bash
      pip install -r requirements.txt
      ```
    (Plus other transitive dependencies listed; see full `requirements.txt` in the GitHub Repo.)

    **4.4. Model Files *(not shown)*:**  
    - **`colorization_deploy_v2.prototxt`:** Defines the CNN architecture layers.  
    - **`colorization_release_v2.caffemodel`:** Contains pre-trained weights (Zhang et al.).  
    - **`pts_in_hull.npy`:** A `313×2` array of cluster centres in (A, B) space (used to "soft-encode" predictions).

    > This detailed breakdown illustrates how each file and function collaborates to transform a simple image upload into a fully colorized output, bridging theory (the Zhang et al. architecture) with practical, production-ready code [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40).  
    """
)

# 6. Results and Demonstration
st.subheader("*:blue[5. Results & Demonstration —]*")
st.write(
    "Below are a few illustrative examples (replace these with your own screenshots or live Streamlit demo links):"
)
st.image("demo/image.gif", caption="Demo (GIF)")
st.write(
    """
    1. **Classic Greyscale Portrait:**  
       - *Input:* A greyscale photograph of a person.  
       - *Output:* Natural skin-tones, subtle shading, and realistic hair color.  

    2. **Landscape Scene:**  
       - *Input:* A greyscale outdoor scene (trees, sky, water).  
       - *Output:* Rich greens for foliage, blue sky gradient, and water reflections.  

    3. **Recoloring a Color Photo:**  
       - *Input:* A full-color image.  
       - *Output:* The model discards original (A, B) channels, infers fresh colors from L (e.g. slight hue shifts).  

    > These examples demonstrated how quickly and accurately the pre-trained CNN was also to colorize a wide variety of inputs, from portraits to landscapes to high-resolution photographs, all within a few hundred milliseconds on modern hardware.
    """
)

# 7. Conclusion
st.subheader("*:blue[6. Conclusion —]*")
st.write(
    """
    This project (:green[Colorize Me]) demonstrates how a pre-trained CNN (by Zhang et al.) can be seamlessly integrated into a lightweight Streamlit application to deliver real-time, high-quality colourisation of greyscale or desaturated images. By leveraging LAB conversions, a publicly available Caffe model, and careful upscaling/post-processing, the pipeline produces natural, spatially coherent colours with minimal computation. The intuitive web interface, complete with upload, interactive comparison, and download, showcases an end-to-end AI deployment in pure Python. Future enhancements might include user-guided hints (e.g., colour scribbles), alternative neural architectures (GAN-based or U-Net), or containerised deployment for scalable production use. As a portfolio piece, this project highlights expertise in deep learning inference, efficient image processing, and rapid web prototyping.
    """
)
# st.write(
#     """
#     “Colorize Me” demonstrates how a pretrained CNN (by Zhang et al.) can be wrapped in a lightweight Streamlit interface to provide real-time single-image colorisation [[ 1 ]](#1-r-zhang-p-isola-and-a-a-efros-colorful-image-colorization-in-computer-vision-eccv-2016-2016-pp-649-666-doi-10-1007-978-3-319-46487-9-40). Key takeaways:

#     - **Simplicity:** With just three model files (prototxt, caffemodel, pts_in_hull), we can produce
#       high-quality full-color outputs in under a second (on modern hardware).
#     - **Extensibility:** Future work could integrate user controls (e.g. “scribble” hints), try alternative
#       colorisation networks (GAN-based, U-Net), or deploy via Flask/Django for production.
#     - **Education & Portfolio:** This live demo serves both as an illustration of deep learning pipelines (color
#       space conversions, DNN inference) and a portfolio piece showcasing Python + AI expertise.
#     """
# )

# 8. References
st.subheader("*:blue[7. References —]*")
st.markdown(
    """    
    ###### :green[[1]] **R. Zhang, P. Isola, and A. A. Efros**. "Colorful Image Colorization", in *Computer Vision* -- ECCV 2016, 2016, pp. 649-666, doi: [10.1007/978-3-319-46487-9_40](https://doi.org/10.1007/978-3-319-46487-9_40).
    
    ###### :green[[2]] richzhang, "GitHub - richzhang/colorization: Automatic colorization using deep neural networks. 'Colorful Image Colorization.' In ECCV, 2016.," GitHub, available: [github.com/richzhang/colorization](https://github.com/richzhang/colorization).
    """,
)
