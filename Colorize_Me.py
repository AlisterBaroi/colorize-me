import streamlit as st
from streamlit_image_comparison import image_comparison
from components.cv import resizeImg, readImg, colorize, readNPArray
from components.utilities import welcome

st.set_page_config(
    page_title="Colorize Me",
    # page_icon=":page_facing_up:",
    layout="centered",
    # initial_sidebar_state="collapsed",
    # menu_items=None
)

# Initialise a session state variable to control the welcome dialogue display
if "dialogue_shown" not in st.session_state:
    st.session_state.dialogue_shown = False

# Show welcome dialogue only once
if not st.session_state.dialogue_shown:
    welcome()
    st.session_state.dialogue_shown = True

# with st.container(border=True, height=350):
with st.container(border=False):
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
