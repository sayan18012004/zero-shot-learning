import os
import argparse
import numpy as np
import tensorflow as tf
import gensim as gs
import gensim.downloader as gdownloader
from typing import List

script_dir: str = os.path.dirname(os.path.realpath(__file__))


def file_path(path: str) -> str:
    """
    Validate if the provided path is a file.

    Args:
        path (str): The path to check.

    Raises:
        argparse.ArgumentTypeError: If the path is not a file.

    Returns:
        str: The path if it is a valid file.
    """
    if os.path.isfile(path):
        return path
    raise argparse.ArgumentTypeError(f"{path} is not a valid file")


def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a TensorFlow model from the specified path.

    Args:
        model_path (str): Path to the model file.

    Returns:
        tf.keras.Model: Loaded TensorFlow model.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def load_image(img_path: str) -> np.ndarray:
    """
    Load and preprocess an image from the specified path.

    Args:
        img_path (str): Path to the image file.

    Returns:
        np.ndarray: Preprocessed image.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")
    img = tf.keras.preprocessing.image.load_img(img_path)
    img_array = np.asarray(img)
    return img_array


def main(image_path: str, model_path: str):
    try:
        # Download and load FastText vectors
        fast_text_vectors: gs.models.keyedvectors = gdownloader.load(
            "fasttext-wiki-news-subwords-300")

        # Load model
        model: tf.keras.Model = load_model(model_path)

        # Load image
        img: np.ndarray = load_image(image_path)

        # Get prediction vector
        img_resized = tf.image.resize(img, (32, 32))
        img_preprocessed = tf.keras.applications.vgg19.preprocess_input(
            img_resized)
        prediction: np.ndarray = model.predict(
            np.expand_dims(img_preprocessed, axis=0))

        # Get top-n labels by cosine similarity
        most_similar: List[str] = fast_text_vectors.similar_by_vector(
            prediction[0], topn=5)

        # Print the predictions for the image
        print(f"Prediction for image: {', '.join([x[0] for x in most_similar])}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # Create parser for command-line arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Image Classifier Arguments')
    parser.add_argument('image', type=file_path, help='Path to the image')
    parser.add_argument('model', type=file_path, help='Path to the model file')
    args = parser.parse_args()

    # Run main function with provided arguments
    main(args.image, args.model)
