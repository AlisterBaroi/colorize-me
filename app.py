import streamlit as st
from components.cv import resizeImg

st.set_page_config(
    page_title="Colorize Me",
    # page_icon=":page_facing_up:",
    layout="wide",
    # initial_sidebar_state="collapsed",
    # menu_items=None
)

# with st.container(border=True, height=350):
with st.container(border=True):
    # st.header("E-Commerce Template", anchor=False)
    uploaded_file = st.file_uploader(
        "Choose an image to colorize...", type=["jpg", "png", "jpeg"]
    )
    row0 = st.columns([1, 1], gap="small")
    row1 = st.columns([1, 1], gap="small")
    if uploaded_file is not None:
        with row0[0]:
            a = resizeImg(uploaded_file, 360)
            st.image(a, caption="Original Preview", width=None)
        with row0[1]:
            # call colorize function here
            a = resizeImg(uploaded_file, 360)
            st.image(a, caption="Colorized Preview", width=None)
            pass

        row1[0].button("Colorize", type="primary", use_container_width=True)
        row1[1].button("Download", type="secondary", use_container_width=True)
    # st.write(
    #     "Welcome to our e-commerce store template. This template is designed to showcase a simple e-commerce store with a product catalog"
    # )
