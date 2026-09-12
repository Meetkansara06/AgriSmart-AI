import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / 'agrismart_model.keras'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'


def _load_resources():
	if not MODEL_PATH.is_file():
		raise FileNotFoundError(f'Model file not found: {MODEL_PATH}')
	if not CLASS_INDICES_PATH.is_file():
		raise FileNotFoundError(f'Class mapping file not found: {CLASS_INDICES_PATH}')

	try:
		loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
	except (OSError, ValueError) as error:
		raise RuntimeError(f'Could not load model file: {MODEL_PATH}') from error

	try:
		with CLASS_INDICES_PATH.open('r', encoding='utf-8') as file:
			loaded_class_indices = json.load(file)
	except (OSError, json.JSONDecodeError) as error:
		raise RuntimeError(f'Could not read class mapping file: {CLASS_INDICES_PATH}') from error

	return loaded_model, {
		int(index): class_name
		for class_name, index in loaded_class_indices.items()
	}


model, index_to_class = _load_resources()


def predict_with_confidence(image_path):
	"""Return the predicted class label and confidence for one image."""
	image_path = Path(image_path)
	if not image_path.is_file():
		raise FileNotFoundError(f'Image file not found: {image_path}')

	try:
		image = tf.keras.utils.load_img(
			image_path,
			target_size=(224, 224),
			color_mode='rgb',
		)
	except (OSError, ValueError) as error:
		raise ValueError(f'Could not read image file: {image_path}') from error

	image_array = np.asarray(tf.keras.utils.img_to_array(image), dtype=np.float32)
	image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
	image_batch = tf.expand_dims(image_array, axis=0)

	probabilities = model.predict(image_batch, verbose=0)[0]
	predicted_index = int(tf.argmax(probabilities).numpy())
	try:
		class_label = index_to_class[predicted_index]
	except KeyError as error:
		raise RuntimeError(f'No class name found for prediction index {predicted_index}.') from error
	confidence = float(probabilities[predicted_index])
	return class_label, confidence


def predict(image_path):
	"""Return the predicted class label for one image."""
	class_label, _ = predict_with_confidence(image_path)
	return class_label


def main():
	parser = argparse.ArgumentParser(description='Predict a plant disease from a leaf image.')
	parser.add_argument('--image', required=True, type=Path, help='Path to the leaf image.')
	args = parser.parse_args()

	try:
		class_label, confidence = predict_with_confidence(args.image)
	except (FileNotFoundError, RuntimeError, ValueError) as error:
		parser.error(str(error))

	print(f'Predicted disease: {class_label}')
	print(f'Confidence: {confidence:.2%}')


if __name__ == '__main__':
	main()
