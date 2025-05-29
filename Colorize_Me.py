import streamlit as st
from streamlit_image_comparison import image_comparison
from components.cv import resizeImg, readImg, colorize

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
        c = colorize(readImg(uploaded_file))
        with row0[1]:
            a = resizeImg(uploaded_file, 360)
            image_comparison(
                img1=readImg(uploaded_file),
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
        row1[0].button("Colorize", type="primary", use_container_width=True)
        row1[1].button("Download", type="secondary", use_container_width=True)
    # st.write(
    #     "Welcome to our e-commerce store template. This template is designed to showcase a simple e-commerce store with a product catalog"
    # )
