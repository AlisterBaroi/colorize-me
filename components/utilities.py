import streamlit as st


@st.dialog("Upload Error")
def error():
    st.error("**Filename Error:** Contains multiple ' . ' in the filename")
    # if st.button("Retry"):
    # st.rerun()


@st.dialog("Colorize Me", width="small")
def welcome():
    st.image("demo/image.png", use_container_width=True)
    st.write(
        "Colorize Black-&-White images. Input a B&W image and let the AI colorize it for you!"
    )
