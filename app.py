import os
import numpy as np
import streamlit as st
import cv2
import pickle
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Function to extract features using ResNet50
def extract_features(img_path, model):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {img_path}")
            return None
        
        img = cv2.resize(img, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        features = model.predict(img)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {img_path}: {e}")
        return None

# Function to load the ResNet50 model
def load_resnet50_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    return model

# Function to train KMeans and save models
def train_model(folder_path, max_images=4000):
    # Load ResNet50 model
    model = load_resnet50_model()

    # Extract features for all images in the folder (limit to max_images)
    feature_list = []
    image_paths = []
    
    image_count = 0
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path) and filename.endswith(('jpg', 'jpeg', 'png')) and image_count < max_images:
            features = extract_features(img_path, model)
            if features is not None:
                feature_list.append(features)
                image_paths.append(img_path)
                image_count += 1
        if image_count >= max_images:
            break
    
    if len(feature_list) == 0:
        print("No images with valid features found in the folder.")
        return None, None
    
    feature_list = np.array(feature_list)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=min(len(feature_list), 10), random_state=42)
    kmeans.fit(feature_list)
    
    # Save the model and image paths
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
    
    with open('image_paths.pkl', 'wb') as f:
        pickle.dump(image_paths, f)

    return kmeans, image_paths

# Function to find similar images
def find_similar_images(kmeans, image_paths, test_img_path, num_similar_images=5):
    # Load ResNet50 model
    model = load_resnet50_model()

    # Extract features for the test image
    test_img_features = extract_features(test_img_path, model)
    if test_img_features is None:
        print(f"Error extracting features for test image: {test_img_path}")
        return []
    
    # Find the cluster of the test image
    test_img_cluster = kmeans.predict([test_img_features])[0]
    
    # Get all images from the same cluster as the test image
    cluster_indices = np.where(kmeans.labels_ == test_img_cluster)[0]
    
    # Find the closest images to the test image in the same cluster
    distances = np.linalg.norm(feature_list[cluster_indices] - test_img_features, axis=1)
    
    # Get the top 'num_similar_images' closest images
    closest_indices = cluster_indices[np.argsort(distances)[:num_similar_images]]
    
    return closest_indices

# Streamlit App
def main():
    # Title and Subtitle
    st.title("THE VOGUE STORE")
    st.subheader("Style that Speaks, Comfort that Lasts")
    
    # Owner's name at the bottom
    st.markdown("<br><br>Created by Abdul Qayyum", unsafe_allow_html=True)
    
    # Dataset path input (for training the model)
    dataset_path = r"C:\Users\YourUserName\Desktop\similarity\dataset"

    
    if dataset_path:
        # Train the model on the provided dataset
        if os.path.exists(dataset_path):
            st.write("Training the model... This may take a while, please be patient.")
            kmeans, image_paths = train_model(dataset_path)
            
            if kmeans and image_paths:
                st.success("Model trained successfully!")
                st.write("You can now upload a test image for similarity search.")
            else:
                st.error("Error training the model. Please check your dataset.")
        else:
            st.error("Invalid dataset path. Please check the path and try again.")
    
    # Upload button for the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Load the pre-trained model and image paths if already trained
    if os.path.exists('kmeans_model.pkl') and os.path.exists('image_paths.pkl'):
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        
        with open('image_paths.pkl', 'rb') as f:
            image_paths = pickle.load(f)
    else:
        st.warning("Model has not been trained yet. Please train the model with a dataset first.")
        return
    
    # Display similar images when the user uploads an image
    if uploaded_file is not None:
        test_img_path = uploaded_file.name
        with open(test_img_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Find similar images
        closest_indices = find_similar_images(kmeans, image_paths, test_img_path, num_similar_images=5)
        
        # Display the result
        if len(closest_indices) > 0:
            st.image(test_img_path, caption="Test Image", use_column_width=True)
            st.subheader("5 Similar Images:")
            
            for idx in closest_indices:
                img = cv2.imread(image_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img, caption=f"Similar Image {idx+1}", use_column_width=True)

if __name__ == "__main__":
    main()
