import pickle
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import os
import base64
from pathlib import Path
from rembg import remove
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(layout="wide", page_title="Model Demo", page_icon="🖥️")
st.sidebar.header("Model Demo")

st.markdown("""
<style>
.small-font {
    font-size:30px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.medium-font {
    font-size:40px !important;
}
</style>
""", unsafe_allow_html=True)

# video_html = """
# 		<style>

# 		#myVideo {
# 		  position: fixed;
# 		  right: 0;
# 		  bottom: 0;
# 		  min-width: 100%; 
# 		  min-height: 100%;
# 		}

# 		.content {
# 		  position: fixed;
# 		  bottom: 0;
# 		  background: rgba(0, 0, 0, 0.5);
# 		  color: #f1f1f1;
# 		  width: 100%;
# 		  padding: 20px;
# 		}

# 		</style>	
# 		<video autoplay muted loop id="myVideo">
# 		  <source src="https://static.streamlit.io/examples/star.mp4">
# 		  Your browser does not support HTML5 video.
# 		</video>
#         """

# st.markdown(video_html, unsafe_allow_html=True)

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
set_background('istockphoto-1208169038-612x612.jpeg')




# def add_bg_from_url():
#     st.markdown(
#          f"""
#          <style>
#          .stApp {{
#              background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#          unsafe_allow_html=True
#      )

# add_bg_from_url() 

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.markdown("<p style='text-align: center;'>"+img_to_html('logo.png')+"</p>", unsafe_allow_html=True)




# st.title("Tiki Recommendation")

#Load dataframe and similarity matrix
# Read dataframe
with open('product_dict.pkl', 'rb') as file:  
    f = pickle.load(file)
    f = pd.DataFrame(f)
# Read similarity matrix
# with open('similarity.pkl', 'rb') as file:  
#     similarity_mat = pickle.load(file)

f["desc_token"]=f["cleaned_desc"].apply(lambda x: word_tokenize(x, format="text"))
tf = TfidfVectorizer(analyzer='word', min_df=0, stop_words="english")
tfidf_matrix = tf.fit_transform(f['desc_token'])

similarity_mat = cosine_similarity(tfidf_matrix, tfidf_matrix)


product = st.selectbox(label='Search here', options=f['name'].values, index=len(f) - 1)
st.header("\n")

# recommendation by getting value from similarity matrix
product = f[f.name == product]
product_index = product.index[len(product) - 1]
distances = similarity_mat[product_index]
mlist = sorted(list(enumerate(distances)), reverse=True, key=lambda item: item[1])[1:6]

mlist = dict(mlist)

for key in mlist:
    mlist[key] *= f.iloc[key].rating
product_list = dict(sorted(mlist.items(), reverse=True, key=lambda item: item[1]))

result = []
for i in product_list:
    result.append([f.iloc[i]['item_id'], f.iloc[i]['name']])




col_in0, col_in1 = st.columns(2)

with col_in0:
    response = requests.get(f.iloc[product_index]['image'])
    img = Image.open(BytesIO(response.content))
    fixed = remove(img)
    st.image(fixed, width=600)

with col_in1:
    name = f.iloc[product_index]['name']
    st.markdown(f'<h1 style="color:#000000;font-size:50px;font-family:Arial;font-weight: bold">{name}</h1>', unsafe_allow_html=True)
    
    st.write("\n")
    rating = f.iloc[product_index]["rating"]
    if rating == 0:
        st.write("No Review Yet")
    elif 1<=rating<2:
        
        st.markdown(f'<p class="medium-font">{"⭐"}</p>', unsafe_allow_html=True)
    elif 2<=rating<3:
        
        st.markdown(f'<p class="medium-font">{"⭐⭐"}</p>', unsafe_allow_html=True)
    elif 3<=rating<4:
        
        st.markdown(f'<p class="medium-font">{"⭐⭐⭐"}</p>', unsafe_allow_html=True)
    elif 4<=rating<5:
        
        st.markdown(f'<p class="medium-font">{"⭐⭐⭐⭐"}</p>', unsafe_allow_html=True)
    else:
        
        st.markdown(f'<p class="medium-font">{"⭐⭐⭐⭐⭐"}</p>', unsafe_allow_html=True)
      

    price = f.iloc[product_index]['price']
    # st.markdown(f'<p class="big-font">{price}</p>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="color:#FF0000;font-size:50px;">{price}</h1>', unsafe_allow_html=True)
    
    st.write("\n")
    brand = f.iloc[product_index]["brand"]
    st.markdown(f'<p class="small-font"; style="color:#000000; font-family: Arial">Brand: {brand}</p>', unsafe_allow_html=True)

    st.write("\n")
    st.markdown(f'<p class="medium-font"; style="color:#000000; font-family: Arial; font-weight:bold">Description</p>', unsafe_allow_html=True)
    st.write("\n")
    overview = f.iloc[product_index]["trunc_des"]
    st.markdown(f'<p class="small-font"; style="color:#000000; font-family: Arial">{overview}</p>', unsafe_allow_html=True)
 

   


st.title("\n")
# recommendations show
st.subheader("Recommendations for you")
st.subheader("\n")

    
# columns
col0, col1, col2, col3, col4 = st.columns(5)

with col0:
    st.markdown(f"[![Foo]({f.loc[f['name']==result[0][1]]['image'].item()})]({f.loc[f['name']==result[0][1]]['link'].item()})")
    product_name = f.loc[f['name']==result[0][1]]['name'].item()
    st.markdown(f'<p style="color:#000000; font-family: Arial, font-size:25px, font-weight:bold">{product_name}</p>', unsafe_allow_html=True)
with col1:
    st.markdown(f"[![Foo]({f.loc[f['name']==result[1][1]]['image'].item()})]({f.loc[f['name']==result[1][1]]['link'].item()})")
    product_name = f.loc[f['name']==result[1][1]]['name'].item()
    st.markdown(f'<p style="color:#000000; font-family: Arial, font-size:25px, font-weight:bold">{product_name}</p>', unsafe_allow_html=True)

with col2:
    st.markdown(f"[![Foo]({f.loc[f['name']==result[2][1]]['image'].item()})]({f.loc[f['name']==result[2][1]]['link'].item()})")
    product_name = f.loc[f['name']==result[2][1]]['name'].item()
    st.markdown(f'<p style="color:#000000; font-family: Arial, font-size:25px, font-weight:bold">{product_name}</p>', unsafe_allow_html=True)


with col3:
    st.markdown(f"[![Foo]({f.loc[f['name']==result[3][1]]['image'].item()})]({f.loc[f['name']==result[3][1]]['link'].item()})")
    product_name = f.loc[f['name']==result[3][1]]['name'].item()
    st.markdown(f'<p style="color:#000000; font-family: Arial, font-size:25px, font-weight:bold">{product_name}</p>', unsafe_allow_html=True)

with col4:
    st.markdown(f"[![Foo]({f.loc[f['name']==result[4][1]]['image'].item()})]({f.loc[f['name']==result[4][1]]['link'].item()})")
    product_name = f.loc[f['name']==result[4][1]]['name'].item()
    st.markdown(f'<p style="color:#000000; font-family: Arial, font-size:25px, font-weight:bold">{product_name}</p>', unsafe_allow_html=True)