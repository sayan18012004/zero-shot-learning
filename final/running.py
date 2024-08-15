import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import gensim.downloader as gdownloader
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import gensim.models as models

# Load the model
model = tf.keras.models.load_model(r'E:\Projects\Image-classification-zero-shot-learning-master\final\zsl_model.keras', custom_objects={'CosineSimilarity': tf.keras.metrics.CosineSimilarity()})

# Load FastText vectors
# fast_text_vectors = gdownloader.load("fasttext-wiki-news-subwords-300")
fast_text_vectors = models.KeyedVectors.load(r"E:\Projects\Image-classification-zero-shot-learning-master\final\fasttext-wiki-news-subwords-300.model")

# Fine labels (same as in training.py)
fine_labels = [
    'apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 
    'lamp', 'leopard', 'lion', 'lizard', 'lobster', 'man', 
    'motorcycle', 'mountain', 'mouse', 'mushroom', 'orange', 'orchid', 'otter', 
    'pear', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 
    'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
    'turtle', 'wardrobe', 'whale', 'wolf', 'woman', 'worm'
]

# Create vector representations for fine labels
fine_labels_vecs = np.array([fast_text_vectors[label] for label in fine_labels])

# Define a function to preprocess the input image
def preprocess_image(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# Define a function to get the closest label
def get_closest_label(vector: np.ndarray) -> str:
    similarities = cosine_similarity([vector], fine_labels_vecs)
    closest_idx = np.argmax(similarities)
    return fine_labels[closest_idx]

# Load and preprocess an example image
img_path = r'E:\Projects\Image-classification-zero-shot-learning-master\demo_images\petunia.jpg'  # replace with your image path
preprocessed_img = preprocess_image(img_path)

# Make prediction
pred_vector = model.predict(preprocessed_img)
predicted_label = get_closest_label(pred_vector[0])

print(f"Predicted Label: {predicted_label}")
