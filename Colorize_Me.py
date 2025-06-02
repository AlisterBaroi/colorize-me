import streamlit as st
from streamlit_image_comparison import image_comparison
from components.cv import resizeImg, readImg, colorize, readNPArray

st.set_page_config(
    page_title="Colorize Me",
    # page_icon=":page_facing_up:",
    layout="centered",
    # initial_sidebar_state="collapsed",
    # menu_items=None
)

# with st.container(border=True, height=350):
with st.container(border=False):
    # st.header("E-Commerce Template", anchor=False)
    # col1, col2, col3 = st.columns([1, 1, 1])
    # with col2:
    st.title("*Colorize Me —*")
    st.markdown(
        "This is an AI/ML project that takes colorless/grayscale (B&W) images and colorizes them using the :green[*Feed-Forward Pass CNN*] model introduced in [*Colorful Image Colorization*](https://doi.org/10.1007/978-3-319-46487-9_40) *(Zhang et al. 2016)*. It also works on colored images; essentially extracting the louminousity from them to be passed to the CNN for recoloring."
    )
    st.divider()
    uploaded_file = st.file_uploader(
        "Choose an image to colorize...", type=["jpg", "png", "jpeg"]
    )
    row0 = st.columns([1, 8, 1], gap="small", border=False)
    if uploaded_file is not None:
        # with row0[0]:
        #     a = resizeImg(uploaded_file, 360)
        #     st.image(a, caption="Original Preview", width=None)
        # with row0[1]:
        #     # call colorize function here
        #     a = resizeImg(uploaded_file, 360)
        #     st.image(a, caption="Colorized Preview", width=None)
        #     pass
        # row0[0].write()
        readNP, fileName = readImg(uploaded_file)
        c = colorize(readNP)
        with row0[1]:
            a = resizeImg(uploaded_file, 360)
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
        # row0[2].write()

        row1 = st.columns([1, 1], gap="small")
        # row1[0].button("Colorize", type="primary", use_container_width=True)
        row0[1].download_button(
            # st.download_button(
            "Download",
            data=readNPArray(c),
            mime="image/png",
            file_name=fileName,
            # type="tertiary",
            use_container_width=True,
        )
    # st.write(
    #     "Welcome to our e-commerce store template. This template is designed to showcase a simple e-commerce store with a product catalog"
    # )
