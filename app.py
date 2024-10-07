import streamlit as st
st.title("Image Classification with Pretrained Model")
st.header("Flower Classification Example")
st.text("Upload a Flower Image for flower classification as SunRose , Rose, Diasy")
from flower_classifier import flower_classification

uploaded_file = st.file_uploader("Choose a Flower image ...", type="jpg")
if uploaded_file is not None:
   image = Image.open(uploaded_file)
   st.image(image, caption='Flower.', use_column_width=True)
   st.write("")
   st.write("Classifying...")
   label = teachable_machine_classification(image, 'hyper_tuned_model.h5')
   if label == 'daisy':
  	st.write("The Flower is  daisy")
   elif label ==  'dandelion':
  	st.write("The Flower is  daisy")
   elif label ==  'rose':
  	st.write("The Flower is  rose")
   elif label == 'sunflower':
  	st.write("The Flower is  Sunflower")
   else:
  	st.write("The Flower is  tulip")