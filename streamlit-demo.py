import os
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


def main():
    set_standard_things()
    df_labels = create_dataframe()
    img_label = df_labels.breed
    img_label = pd.get_dummies(df_labels.breed)
    model = load_transfer_model()
    
    st.write('We are working with the following data.')
    df_labels.head(3)
    
    st.write('Show the top-10 breed that has the most images')
    gr_labels = df_labels.groupby("breed").count()
    gr_labels = gr_labels.rename(columns = {"id" : "count"})
    gr_labels = gr_labels.sort_values("count", ascending=False)
    
    st.table(gr_labels[:5])
    
    st.write('Show dataset distribution')
    st.bar_chart(df_labels.breed.unique()[:10], use_container_width=True)
    
    image, pred, tag = get_prediction_of_image(model, img_label)
    st.image(image, width=300, caption=f'Class: {tag} and probability prediction: {pred}')
    
    
    # Select a file (for localhost)
    # if st.checkbox('Select a file in current directory'):
    #     folder_path = '.'
    # if st.checkbox('Change directory'):
    #     folder_path = st.text_input('Enter folder path', '.')
    #     filename = file_selector(folder_path=folder_path)
    #     st.write('You selected `%s`' % filename)
        
    #     # Read in image file
    #     image = tf.io.read_file(filename)
    #     # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    #     image = tf.image.decode_jpeg(image, channels=3)
    #     # Convert the colour channel values from 0-225 values to 0-1 values
    #     image = tf.image.convert_image_dtype(image, tf.float32)
    #     # Resize the image to our desired size (224, 244)
    #     image = tf.image.resize(image, size=(224, 224))
    #     new_image = np.expand_dims(image, axis=0)

    #     yhat = model.predict(new_image)
    #     label = img_label.columns[np.argmax(yhat)]
        
    #     label
    #     'Probability prediction: ', np.max(yhat[0])
        
    #     st.image(filename)
    
    
    

def get_prediction_of_image(model, img_label):
    uploaded_file = st.file_uploader("Choose an image...", type=("jpg","png","jpeg"))
    if uploaded_file is not None:
        # Read in image file
        image = Image.open(uploaded_file)
        # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
        img_array = np.array(image)
        # Convert the colour channel values from 0-225 values to 0-1 values
        i = tf.image.convert_image_dtype(img_array, tf.float32)
        # Resize the image to our desired size (224, 244)
        i = tf.image.resize(i, size=(224, 224))
        # make it the correct dimmensions
        new_image = np.expand_dims(i, axis=0)
        
        yhat = model.predict(new_image)
        label = img_label.columns[np.argmax(yhat)]
        return image, np.max(yhat[0]), label
    

@st.cache(allow_output_mutation=True)
def load_transfer_model():
    model_weights = 'saved_model/model.h5'
    model_json = 'saved_model/model.json'
    with open(model_json) as json_file:
        loaded_model = model_from_json(json_file.read(), custom_objects={'KerasLayer':hub.KerasLayer})
    loaded_model.load_weights(model_weights)
    loaded_model.summary()  # included to make it visible when model is reloaded
    #session = K.get_session()
    return loaded_model
   
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
    
def create_dataframe() -> pd.DataFrame:
    return pd.read_csv("labels.csv")
    
def set_standard_things():
    st.title('Dog breed classification')
    seed = 42

if __name__ == '__main__':
    main()