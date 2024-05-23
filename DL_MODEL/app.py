# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import pandas as pd
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Load the trained model
# model = tf.keras.models.load_model("D:/CROP_PREDECTION/crop_disease_model.h5")

# # Load the CSV file containing disease information
# disease_info_df = pd.read_csv("D:/CROP_PREDECTION/DISEASE_DETECTION.csv")

# # Define function to make predictions
# def predict_crop_disease(image_path, train_generator):
#     img = load_img(image_path, target_size=(224, 224))  # Resize image to match model input size
#     img_array = img_to_array(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match batch size
#     predictions = model.predict(img_array)
#     predicted_class_index = np.argmax(predictions[0])
#     class_labels = train_generator.class_indices
#     predicted_class_label = [k for k, v in class_labels.items() if v == predicted_class_index][0]
#     return predicted_class_label

# # Define function to get recommendations
# def get_recommendations(predicted_disease):
    
#     disease_row = disease_info_df[disease_info_df['Disease Name'] == predicted_disease]
#     if len(disease_row) == 0:
#         return "No recommendations found for the predicted disease."

#     symptoms = disease_row['Symptoms'].values[0]
#     reason = disease_row['Reason'].values[0]
#     prevention_measures = disease_row['Prevention Measures'].values[0]
#     fertilizers_used = disease_row['Fertilizers Used'].values[0]

#     recommendations = f"Disease: {predicted_disease}\n\n"
#     recommendations += f"Symptoms: {symptoms}\n\n"
#     recommendations += f"Reason: {reason}\n\n"
#     recommendations += f"Prevention Measures: {prevention_measures}\n\n"
#     recommendations += f"Fertilizers Used: {fertilizers_used}\n\n"

#     return recommendations
#     # Implement this function according to your needs
#     # Example: Fetch recommendations from a CSV file based on the predicted disease
#     return "Recommendations for " + predicted_disease

# # Streamlit web application
# def main():
#     st.title("Crop Disease Prediction and Recommendation System")
#     st.write("Upload an image of the crop to predict the disease and get recommendations.")

#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Load data generator for class indices
#         train_datagen = ImageDataGenerator(rescale=1./255)
#         train_generator = train_datagen.flow_from_directory(
#             directory='path/to/dataset',  # Specify the path to your dataset directory
#             target_size=(224, 224),
#             batch_size=1,
#             class_mode='categorical',
#             shuffle=False
#         )

#         image = load_img(uploaded_file, target_size=(224, 224))  # Resize image to match model input size
#         st.image(image, caption="Uploaded Image", use_column_width=True)

#         predicted_disease = predict_crop_disease(uploaded_file, train_generator)
#         recommendations = get_recommendations(predicted_disease)

#         st.write("Predicted disease:", predicted_disease)
#         st.write("Recommendations:")
#         st.write(recommendations)

# if __name__ == "__main__":
#     main()










# import streamlit as st
# from PIL import Image
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# # Load the trained model
# model = load_model('D:/CROP_PREDECTION/crop_disease_model.h5')

# # Define labels
# labels = ["Corn (Maize) - Cercospora Leaf Spot & Gray Leaf Spot",
# "Corn (Maize) - Common Rust",
# "Corn (Maize) - Northern Leaf Blight",
# "Corn (Maize) - Healthy",
# "Grape - Black Rot",
# "Grape - Esca (Black Measles)",
# "Grape - Leaf Blight (Isariopsis Leaf Spot)",
# "Grape - Healthy",
# "Potato - Early Blight",
# "Potato - Late Blight",
# "Potato - Healthy",
# "Tomato - Bacterial Spot",
# "Tomato - Early Blight",
# "Tomato - Late Blight",
# "Tomato - Leaf Mold",
# "Tomato - Septoria Leaf Spot",
# "Tomato - Target Spot",
# "Tomato - Spider Mites Two-Spotted Spider Mite",
# "Tomato - Tomato Yellow Leaf Curl Virus",
# "Tomato - Tomato Mosaic Virus",
# "Tomato - Healthy",
# "Wheat - Brown Rust",
# "Wheat - Healthy",
# "Wheat - Yellow Rust",
# "Cotton - Anthracnose",
# "Rice - Bacterial Blight",
# "Rice - Brown Spot",
# "Rice - Common Rust",
# "Rice - Flag Smut",
# "Rice - Gray Leaf Spot",
# "Maize - Healthy",
# "Wheat - Healthy",
# "Cotton - Healthy",
# "Maize - Leaf Curl",
# "Wheat - Leaf Smut",
# "Sugarcane - Mosaic",
# "Sugarcane - Red Rot",
# "Sugarcane - Red Rust",
# "Rice - Blast",
# "Sugarcane - Healthy",
# "Rice - Tungro",
# "Wheat - Brown Leaf Rust",
# "Wheat - Stem Fly",
# "Wheat - Aphid",
# "Wheat - Black Rust",
# "Wheat - Leaf Blight",
# "Wheat - Powdery Mildew",
# "Wheat - Scab",
# "Wheat - Yellow Rust",
# "Wheat - Wilt",
# "Banana - Black Sigatoka Disease",
# "Banana - Bract Mosaic Virus Disease",
# "Banana - Healthy Leaf",
# "Banana - Insect Pest Disease",
# "Banana - Moko Disease",
# "Banana - Panama Disease",
# "Banana - Yellow Sigatoka Disease"]

# # Define function to preprocess image
# def preprocess_image(image):
#     image = image.resize((224, 224))  # Assuming ResNet50 input shape
#     image = img_to_array(image) / 255.0  # Normalize pixel values
#     image = np.expand_dims(image, axis=0)
#     return image

# # Streamlit app
# def main():
#     st.title("Crop Disease Prediction")

#     # Upload image
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image.', use_column_width=True)

#         # Preprocess image
#         processed_image = preprocess_image(image)

#         # Predict
#         prediction = model.predict(processed_image)
#         predicted_class_index = np.argmax(prediction)
#         predicted_class = labels[predicted_class_index]

#         st.write(f"Predicted class: {predicted_class}")

# if __name__ == "__main__":
#     main()











import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model('D:/CROP_PREDECTION/crop_disease_model.h5')

# Load the CSV file containing disease information
disease_info_df = pd.read_csv("D:/CROP_PREDECTION/DISEASE_DETECTION.csv")

# Define labels
labels = ['Anthracnose on Cotton', 
          'Bacterial leaf blight', 'Banana Black Sigatoka Disease', 'Banana Bract Mosaic Virus Disease', 'Banana Healthy Leaf', 'Banana Insect Pest Disease', 'Banana Moko Disease', 'Banana Panama Disease', 'Banana Yellow Sigatoka Disease', 'Becterial Blight in Rice', 'Brown spot', 'Brownspot', 'Common_Rust', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Flag Smut', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Gray_Leaf_Spot', 'Healthy cotton', 'Healthy Maize', 'Healthy Wheat', 'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast', 'Sugarcane Healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tungro', 'Wheat aphid', 'Wheat black rust', 'Wheat Brown leaf Rust', 'Wheat leaf blight', 'Wheat powdery mildew', 'Wheat scab', 'Wheat Stem fly', 'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust', 'Wilt', 'Wheat___Healthy']



# Define function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Assuming ResNet50 input shape
    image = img_to_array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)
    return image

# Function to get recommendations
def get_recommendations(predicted_disease):
    disease_row = disease_info_df[disease_info_df['Disease Name'] == predicted_disease]
    if len(disease_row) == 0:
        return "No recommendations found for the predicted disease."

    symptoms = disease_row['Symptoms'].values[0]
    reason = disease_row['Reason'].values[0]
    prevention_measures = disease_row['Prevention Measures'].values[0]
    fertilizers_used = disease_row['Fertilizers Used'].values[0]

    recommendations = f"Disease: {predicted_disease}\n\n"
    recommendations += f"Symptoms: {symptoms}\n\n"
    recommendations += f"Reason: {reason}\n\n"
    recommendations += f"Prevention Measures: {prevention_measures}\n\n"
    recommendations += f"Fertilizers Used: {fertilizers_used}\n\n"

    return recommendations

# Streamlit app
def main():
    st.title("Crop Disease Prediction and Recommendations")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Preprocess image
        processed_image = preprocess_image(image)

        # Predict
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        predicted_class = labels[predicted_class_index]

        st.write(f"Predicted class: {predicted_class}")

        # Get recommendations
        recommendations = get_recommendations(predicted_class)
        st.write("Recommendations:")
        st.write(recommendations)

if __name__ == "__main__":
    main()
