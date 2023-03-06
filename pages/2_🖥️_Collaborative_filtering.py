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
from sklearn.neighbors import NearestNeighbors


st.set_page_config(layout="wide", page_title="Collaborative_filtering", page_icon="🖥️")
st.sidebar.header("Collaborative_filtering Demo")

#Load dataframe
with open('df.pkl', 'rb') as file:  
    df = pickle.load(file)
    df = pd.DataFrame(df)

# Create user-item matrix
matrix = df.pivot_table(index='item_id', columns='customer_id', values='rating')
matrix = matrix.fillna(0)

matrix1 = matrix.copy()

user = st.selectbox(label='Search Customer_ID here', options=df['customer_id'].values, index=len(df) - 1)
st.header("\n")
num_recommendation = st.slider('Number of products recommendation', 1, 5)
num_neighbors = 5



# List of items the selected user has bought

bought = []
for m in matrix[matrix[user] > 0][user].index.tolist():
    bought.append(m)


num_bought = []
if len(bought)<=3:
   num_bought=bought
else:
   num_bought=bought[:3]

st.subheader(f"User-{user} has bought below items:")
st.write("\n"*4)


if len(num_bought)==1:
    col1 = st.columns(1)
    st.write(df[df['item_id']==num_bought[0]]['name'].unique().item())
    response = requests.get(df[df['item_id']==num_bought[0]]['image'].unique().item())
    img = Image.open(BytesIO(response.content))
    st.image(img, width=400)
   
elif len(num_bought)==2:
   col1, col2 = st.columns(2)
   with col1:
    st.write(df[df['item_id']==num_bought[0]]['name'].unique().item())
    response = requests.get(df[df['item_id']==num_bought[0]]['image'].unique().item())
    img = Image.open(BytesIO(response.content))
    st.image(img)
   with col2:
    st.write(df[df['item_id']==num_bought[1]]['name'].unique().item())
    response = requests.get(df[df['item_id']==num_bought[1]]['image'].unique().item())
    img = Image.open(BytesIO(response.content))
    st.image(img)

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(df[df['item_id']==num_bought[0]]['name'].unique().item())
        response = requests.get(df[df['item_id']==num_bought[0]]['image'].unique().item())
        img = Image.open(BytesIO(response.content))
        st.image(img)
    with col2:
        st.write(df[df['item_id']==num_bought[1]]['name'].unique().item())
        response = requests.get(df[df['item_id']==num_bought[1]]['image'].unique().item())
        img = Image.open(BytesIO(response.content))
        st.image(img)
    with col3:
        st.write(df[df['item_id']==num_bought[2]]['name'].unique().item())
        response = requests.get(df[df['item_id']==num_bought[2]]['image'].unique().item())
        img = Image.open(BytesIO(response.content))
        st.image(img)

st.write("\n"*10)


#calculate distance and similarity scores
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(matrix.values)
distances, indices = knn.kneighbors(matrix.values, n_neighbors=num_neighbors)

user_index = matrix.columns.tolist().index(user)

for m,t in list(enumerate(matrix.index)):
    if matrix.iloc[m, user_index] == 0:
      sim_items = indices[m].tolist()
      item_distances = distances[m].tolist()
      
      if m in sim_items:
        id_item = sim_items.index(m)
        sim_items.remove(m)
        item_distances.pop(id_item) 

      else:
        sim_items = sim_items[:n_neighbors-1]
        item_distances = item_distances[:n_neighbors-1]
           
      item_similarity = [1-x for x in item_distances]
      item_similarity_copy = item_similarity.copy()
      nominator = 0

      for s in range(0, len(item_similarity)):
        if matrix.iloc[sim_items[s], user_index] == 0:
          if len(item_similarity_copy) == (num_neighbors - 1):
            item_similarity_copy.pop(s)
          
          else:
            item_similarity_copy.pop(s-(len(item_similarity)-len(item_similarity_copy)))
            
        else:
          nominator = nominator + item_similarity[s]*matrix.iloc[sim_items[s],user_index]
          
      if len(item_similarity_copy) > 0:
        if sum(item_similarity_copy) > 0:
          predicted_r = nominator/sum(item_similarity_copy)
        
        else:
          predicted_r = 0

      else:
        predicted_r = 0
        
      matrix1.iloc[m,user_index] = predicted_r

# recommend
recommended_item = []

for m in matrix[matrix[user] == 0].index.tolist():
    
    index_df = matrix.index.tolist().index(m)
    predicted_rating = matrix1.iloc[index_df, matrix1.columns.tolist().index(user)]
    recommended_item.append((m, predicted_rating))

sorted_rm = sorted(recommended_item, key=lambda x:x[1], reverse=True)
  
  
result = []
for recommended_item in sorted_rm[:num_recommendation]:
   result.append([recommended_item[0], recommended_item[1]])

if st.button('Recommend'):


   if num_recommendation == 1:
        col1 = st.columns(1)
        st.write(df[df['item_id']==result[0][0]]['name'].unique().item())
        response = requests.get(df[df['item_id']==result[0][0]]['image'].unique().item())
        img = Image.open(BytesIO(response.content))
        st.image(img, width=400)

   elif num_recommendation == 2:
        col1, col2 = st.columns(2)
        with col1:
           st.write(df[df['item_id']==result[0][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[0][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        with col2:
           st.write(df[df['item_id']==result[1][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[1][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
   elif num_recommendation == 3:
        col1, col2, col3 = st.columns(3)
        with col1:
           st.write(df[df['item_id']==result[0][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[0][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        with col2:
           st.write(df[df['item_id']==result[1][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[1][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        with col3:
           st.write(df[df['item_id']==result[2][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[2][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
   elif num_recommendation == 4:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
           st.write(df[df['item_id']==result[0][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[0][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        with col2:
           st.write(df[df['item_id']==result[1][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[1][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        with col3:
           st.write(df[df['item_id']==result[2][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[2][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        with col4:
           st.write(df[df['item_id']==result[3][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[3][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)

   else:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
           st.write(df[df['item_id']==result[0][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[0][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        
        with col2:
           st.write(df[df['item_id']==result[1][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[1][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        
        with col3:
           st.write(df[df['item_id']==result[2][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[2][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        
        with col4:
           st.write(df[df['item_id']==result[3][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[3][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
    
        with col5:
           st.write(df[df['item_id']==result[4][0]]['name'].unique().item())
           response = requests.get(df[df['item_id']==result[4][0]]['image'].unique().item())
           img = Image.open(BytesIO(response.content))
           st.image(img)
        


      



