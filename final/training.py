import numpy as np
import tensorflow as tf
import gensim as gs
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as gdownloader
from typing import List
import pandas as pd

# Load CIFAR-100 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

# Load FastText vectors
fast_text_vectors: gs.models.keyedvectors = gdownloader.load("fasttext-wiki-news-subwords-300")

fine_labels = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 
    'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 
    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 
    'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 
    'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

def word_list_to_avg_vector(lst: List[str]) -> np.ndarray:
    vec = np.array([0.0] * 300)
    for word in lst:
        vec += fast_text_vectors.word_vec(word)
    return vec / len(lst)

fine_labels_words = list(map(tf.keras.preprocessing.text.text_to_word_sequence, fine_labels))
fine_labels_vecs = np.asarray([word_list_to_avg_vector(words) for words in fine_labels_words])

y_train_vecs = np.asarray([fine_labels_vecs[label] for label in y_train]).reshape((50000, 300))
y_test_vecs = np.asarray([fine_labels_vecs[label] for label in y_test]).reshape((10000, 300))

# Preprocess the images
input_train = tf.keras.applications.vgg19.preprocess_input(x_train)
input_test = tf.keras.applications.vgg19.preprocess_input(x_test)

# Create the model
base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3), pooling='max')
base_model.trainable = False

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(32,32,3)),
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    base_model,
    tf.keras.layers.Dense(448, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(384, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(300)
])

model.compile(optimizer='adam', loss=tf.keras.losses.CosineSimilarity(), metrics=[tf.keras.metrics.CosineSimilarity()])

# Train the model
model.fit(input_train, y_train_vecs, validation_data=(input_test, y_test_vecs), epochs=1, batch_size=32)

# Save the model
model.save('zsl_model.keras')
