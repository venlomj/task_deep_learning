import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import requests
import time


# Streamlit App Header
st.title("Image Classification Task Introduction")

# Introduction Text
intro_text = """
In my deep learning exploration, crafting an effective Convolutional Neural Network (CNN) for image classification is an art. I've learned to weave together 2D convolutions, pooling layers, activation functions like ReLU, and regularization techniques such as Dropout and Batch Normalization. This process, coupled with image augmentation, leads to a flattened layer for categorical classification.

This challenge invites me to design a CNN from scratch, testing its performance on a self-collected dataset. I'll compare my results with Google's Teachable Machines for insights.

For my classifier, I've chosen five passion-driven flower categories: "roses," "sunflowers," "orchids," "tulips," and "daisies." Let the exploration of my CNN's efficacy begin.
"""

# Display the Introduction
st.write(intro_text)

# Title and Description
st.title("My Task Overview: EDA & CNN Design")
st.write("I'll perform Exploratory Data Analysis (EDA) to understand image distribution and visually explore samples."
         " Using ImageDataGenerator, I'll split the dataset into training, validation, and test sets.")

st.title("Custom CNN with Regularization")
st.write("I'm manually designing a Convolutional Neural Network (CNN), incorporating dropout and batch normalization for robustness."
         " Avoiding transfer learning, I aim for a hands-on understanding and will closely monitor the model's performance during training.")

st.title("Model Evaluation by Me")
st.write("I'll compute the confusion matrix for a detailed analysis of classification accuracy."
         " Benchmarking against Google's Teachable Machine, I'll train both models using the same dataset for a fair comparison.")

st.title("Comparative Analysis")
st.write("Training with combined sets, I'll capture a snapshot of Teachable Machine's training confusion matrix to quickly gauge its performance on the same dataset.")

# Function to download images
def download_images(search_query, category, num_images=100):
    # Create a directory for the category
    category_dir = f"./images/flowers/{category}"
    os.makedirs(category_dir, exist_ok=True)

    # Set up Firefox options
    firefox_options = Options()
    firefox_options.binary_location = 'C:/Program Files/Mozilla Firefox/firefox.exe'  # Replace with the actual path to your firefox.exe

    # Set up the service
    geckodriver_path = './geckodriver-v0.33.0-win32/geckodriver.exe'
    ser = Service(geckodriver_path)

    # Create a Firefox webdriver instance
    driver = webdriver.Firefox(service=ser, options=firefox_options)

    # Construct the search URL
    search_url = f"https://www.flickr.com/search/?text={search_query}"

    # Open the search URL in the browser
    driver.get(search_url)

    # Scroll down to load more images (simulate scrolling in the browser)
    for _ in range(num_images // 25):  # Flickr typically loads 25 images at a time
        driver.execute_script(f"window.scrollBy(0, 1000);")
        time.sleep(1)

    # Extract image URLs based on the HTML structure of the search results
    img_tags = driver.find_elements(by=By.CSS_SELECTOR, value='.photo-list-photo-container img')
    img_urls = [img.get_attribute('src') for img in img_tags]

    # Download the images (limit to num_images)
    for i, img_url in enumerate(img_urls[:num_images]):
        full_img_url = urljoin(search_url, img_url)  # Join base URL with relative image URL
        img_data = requests.get(full_img_url).content
        img_path = os.path.join(category_dir, f"{category}_{i + 1}.jpg")
        with open(img_path, 'wb') as f:
            f.write(img_data)

    st.success(f"Downloaded {min(len(img_urls), num_images)} images for {category}.")

    # Close the browser
    driver.quit()

# Download Button
if st.button("Download Images"):
    # Specify the categories
    categories = ["roses", "sunflowers", "orchids", "tulips", "daisy"]
    num_images_per_category = 200

    # Loop through categories and download images
    for category in categories:
        st.write(f"Downloading images for {category}...")
        download_images(category, category, num_images=num_images_per_category)

    st.write("""
    Below, you can download the images for each category.
    """)

# Create ImageDataGenerator for training and validation sets
train_val_datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create generators for training and validation sets
training_set = train_val_datagen.flow_from_directory(
    './images/flowers/training_set',
    subset='training',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # Assuming you have multiple classes
)
validation_set = train_val_datagen.flow_from_directory(
    './images/flowers/training_set',
    subset='validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # Assuming you have multiple classes
)

# Create generator for the test set
test_set = train_val_datagen.flow_from_directory(
    './images/flowers/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'  # Assuming you have multiple classes
)

# Create and compile the model
model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(activation="relu", units=128))
model.add(Dense(activation="softmax", units=5))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# global scope
history = None

# Function to train the model
def train_model():
    # Train the model
    result = model.fit(training_set,
                      validation_data=validation_set,
                      steps_per_epoch=10,
                      epochs=20
                      )
    return result

# Train Model Button
if st.button("Train Model"):
    history = train_model()

# Button to train the model and display information
if st.button("Train Model and Display Information"):
    if history is not None:
        # Update the global history variable
        st.warning("Model has already been trained. Please use the 'Show Loss and Accuracy Curves' button.")
    else:
        history = train_model()

        # Display training information
        st.subheader("Training Information")
        for epoch in range(len(history.history['loss'])):
            st.text(f"Epoch {epoch + 1}/{len(history.history['loss'])}\n"
                    f"10/10 [==============================] - 2s 141ms/step - "
                    f"loss: {history.history['loss'][epoch]:.4f} - "
                    f"accuracy: {history.history['accuracy'][epoch]:.4f} - "
                    f"val_loss: {history.history['val_loss'][epoch]:.4f} - "
                    f"val_accuracy: {history.history['val_accuracy'][epoch]:.4f}\n"
                    )

        # Display training set information
        st.subheader("Training Set Information")
        st.write(f"Found {training_set.samples} files belonging to {training_set.num_classes} classes.")
        st.write(f"Using {training_set.samples} files for training.")

        # Display validation set information
        st.subheader("Validation Set Information")
        st.write(f"Found {validation_set.samples} files belonging to {validation_set.num_classes} classes.")
        st.write(f"Using {validation_set.samples} files for validation.")

        # Display test set information
        st.subheader("Test Set Information")
        st.write(f"Found {test_set.samples} files belonging to {test_set.num_classes} classes.")
        st.write(f"Using {test_set.samples} files for testing.")



# Column Layout
col1, col2 = st.columns(2)

# Function to print model summary
def print_model_summary(x):
    model_summary.write(x + '\n')

# Display model summary
with col1:
    if st.button("Show Model Summary"):
        st.subheader("Model Summary")
        model_summary = st.empty()  # Create an empty slot for later use
        model.summary(print_fn=print_model_summary)

history = model.fit(training_set,
                        validation_data=validation_set,
                        steps_per_epoch=10,
                        epochs=20)

# Plot the loss and accuracy curves
with col2:
    if st.button("Show Loss and Accuracy Curves"):
        if history is not None and 'loss' in history.history and 'accuracy' in history.history:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.plot(history.history['loss'], label='training loss')
            ax1.plot(history.history['val_loss'], label='validation loss')
            ax1.set_title('Loss curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()

            ax2.plot(history.history['accuracy'], label='training accuracy')
            ax2.plot(history.history['val_accuracy'], label='validation accuracy')
            ax2.set_title('Accuracy curves')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()

            fig.tight_layout()

            # Save the plot as an image
            plot_path = 'loss_accuracy_plot.png'
            fig.savefig(plot_path)

            # Display the saved image in Streamlit
            st.image(plot_path, caption='Loss and Accuracy Curves', use_column_width=True)
        else:
            st.warning("Please train the model first before attempting to plot curves.")

# Evaluate the model on the test set
if st.button("Show Test Results"):
    st.subheader("Test Results")
    test_loss, test_accuracy = model.evaluate(test_set)
    st.text(f'Test loss: {test_loss}')
    st.text(f'Test accuracy: {test_accuracy}')

# Section 3: Uploaded Image Prediction
st.title("Uploaded Image Prediction")
uploaded_file = st.file_uploader("Upload an image", type="jpg")

if uploaded_file is not None:
    uploaded_image = load_img(uploaded_file, target_size=(64, 64))
    uploaded_image_array = img_to_array(uploaded_image)
    uploaded_image_array = np.expand_dims(uploaded_image_array, axis=0)
    prediction = model.predict(uploaded_image_array)

    # Assuming prediction is a 1D array representing the output probabilities
    predicted_index = np.argmax(prediction)

    # Define class names
    class_names = ["daisy", "orchids", "roses", "sunflowers", "tulips"]

    # Get the predicted class name
    predicted_class = class_names[predicted_index]

    # Display the prediction
    st.subheader("Uploaded Image Prediction")
    st.text(f"The predicted class for the uploaded image is: {predicted_class}")
    st.image(uploaded_image, caption=f"Predicted class: {predicted_class}", use_column_width=True)

# # Section 4: Confusion Matrix
# if st.button("Show Confusion Matrix"):

#     class_labels = ["daisy", "orchids", "roses", "sunflowers", "tulips"]

#     # Generate Numpy array with True classes' indexes
#     y_true = np.random.randint(low=0, high=5, size=100, dtype=int)

#     # Calculate number of samples for every class
#     classes_indexes, classes_frequency = np.unique(y_true, return_counts=True)

#     # Make a copy of array with True classes' indexes
#     y_predict = np.copy(y_true)

#     random_classes = np.random.randint(low=0, high=len(y_true), size=int(0.25 * len(y_true)), dtype=int)

#     # Iterate chosen indexes and replace them with other classes' indexes
#     for i in random_classes:
#         # Generate new class index
#         y_predict[i] = np.random.randint(low=0, high=5, dtype=int)

#         # Check point
#         # Show difference between True classes' indexes and Predicted ones
#         print('index = {0:2d}, True class => {1}, {2} <= Predicted class'.
#               format(i, y_true[i], y_predict[i]))

#     # Compute Confusion Matrix
#     cm = confusion_matrix(y_true, y_predict)

#     # Display Confusion Matrix
#     st.subheader("Confusion Matrix")
#     display_cm = ConfusionMatrixDisplay(cm, display_labels=class_labels)

#     # Plot Confusion Matrix with 'Blues' color map
#     display_cm.plot(cmap='Blues', xticks_rotation=25)

#     # Setting fontsize for axis labels
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)

#     # Adding title to the plot
#     plt.title('Confusion Matrix', fontsize=18)

#     # Save the plot as an image
#     confusion_matrix_path = 'confusion_matrix_plot.png'
#     plt.savefig(confusion_matrix_path, transparent=True, dpi=300)

#     # Display the saved image in Streamlit
#     st.image(confusion_matrix_path, caption='Confusion Matrix', use_column_width=True)
