import numpy as np
import tensorflow as tf
import matplotlib as mpl
import pandas as pd
import gensim as gs
import sklearn.metrics.pairwise as skpairwise
import gensim.downloader as gdownloader
from matplotlib import pyplot as plt
from typing import List, Dict
from tensorflow.keras.models import load_model

# Load the vectors from the file
fast_text_vectors = gs.models.keyedvectors.KeyedVectors.load(
    "fast_text_vectors.kv")

# Load the model
model = load_model(r"E:\Projects\zero-shot-learning\final\model.keras")

images_paths: List[str] = [
    r'E:\Projects\zero-shot-learning\demo_images\tansy.jpg'
]

for img_path in images_paths:
    img: np.ndarray = np.asarray(tf.keras.preprocessing.image.load_img(
        img_path))

    # get prediction vector
    prediction: np.ndarray = model.predict(np.expand_dims(
        tf.keras.applications.vgg19.preprocess_input(tf.image.resize(
            img, (32, 32))), axis=0))

    # get top-n labels by cosine similarity
    most_similar: List[str] = fast_text_vectors.similar_by_vector(
        prediction[0], topn=5)

    # display image
    plt.figure(figsize=(2,2))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # print the predictions for image
    print(f"Prediction for image: {', '.join([x[0] for x in most_similar])}")
