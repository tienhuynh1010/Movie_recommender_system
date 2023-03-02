
import streamlit as st
import base64
from path import Path

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('istockphoto-1301889759-612x612.jpeg')





st.markdown(f'<h1 style="color:#000000;font-size:70px;font-family:Arial; font-weight: bold, text-align:center"># Welcome to Tiki! 👋</h1>', unsafe_allow_html=True)
st.sidebar.success("Select an option above.")

st.write("\n"*10)

text = """Tiki is an e-commerce company that specializes in the end-to-end supply chain and partnering with brands. 
Tiki.vn features more than 300,000 products in 12 categories of electronics, lifestyle, and books!"""

st.markdown(f'<p style="color:#8E44AD;font-size:20px;font-family:Arial;">{text}</p>', unsafe_allow_html=True)
st.write("\n"*10)
st.markdown(f'<p style="color:#8E44AD;font-size:30px;font-family:Arial">**👈 Select a demo from the sidebar** to see what product we can offer!</p>', unsafe_allow_html=True)

st.write("\n"*30)

st.markdown(f'<p style="color:#8E44AD;font-size:60px;font-family:Arial">### Want to learn more?</p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:#8E44AD;font-size:30px;font-family:Arial">- Check out <a href="https://tiki.vn/">Tiki.vn</a></p>', unsafe_allow_html=True)
st.markdown(f'<p style="color:#8E44AD;font-size:30px;font-family:Arial">- Ask a question in our <a href="https://tiki.vn/lien-he/gui-yeu-cau">community forum</a></p>', unsafe_allow_html=True)

# ### Want to learn more?
# - Check out [Tiki.vn](https://tiki.vn/)
# - Ask a question in our [community forums](https://tiki.vn/lien-he/gui-yeu-cau)
# ### Be a gift or life amenities, Tiki always beside you !



