import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
MODEL_PATH = MODEL_DIR / 'agrismart_model.keras'
CLASS_INDICES_PATH = MODEL_DIR / 'class_indices.json'


def evaluate(data_dir, output_dir):
	"""Evaluate the saved model on an unseen directory-structured dataset."""
	with CLASS_INDICES_PATH.open('r', encoding='utf-8') as file:
		class_indices = json.load(file)

	class_names = [
		name for name, index in sorted(class_indices.items(), key=lambda item: item[1])
	]
	data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
		preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
	)
	dataset = data_generator.flow_from_directory(
		str(data_dir),
		target_size=(224, 224),
		batch_size=32,
		classes=class_names,
		shuffle=False,
	)
	model = tf.keras.models.load_model(MODEL_PATH, compile=False)

	probabilities = model.predict(dataset, verbose=1)
	predicted_labels = probabilities.argmax(axis=1)
	true_labels = dataset.classes
	labels = list(range(len(class_names)))
	accuracy = float((predicted_labels == true_labels).mean())
	macro_f1 = float(f1_score(true_labels, predicted_labels, average='macro'))
	report = classification_report(
		true_labels,
		predicted_labels,
		labels=labels,
		target_names=class_names,
		zero_division=0,
	)
	matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / 'classification_report.txt').write_text(
		f'Test accuracy: {accuracy:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}',
		encoding='utf-8',
	)
	(output_dir / 'evaluation_metrics.json').write_text(
		json.dumps({'accuracy': accuracy, 'macro_f1': macro_f1}, indent=4),
		encoding='utf-8',
	)

	plt.figure(figsize=(16, 14))
	sns.heatmap(
		matrix,
		annot=True,
		fmt='d',
		cmap='Blues',
		xticklabels=class_names,
		yticklabels=class_names,
	)
	plt.xlabel('Predicted label')
	plt.ylabel('True label')
	plt.title('AgriSmart AI Evaluation Confusion Matrix')
	plt.xticks(rotation=90)
	plt.yticks(rotation=0)
	plt.tight_layout()
	plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
	plt.close()

	print(f'Accuracy: {accuracy:.4f}')
	print(f'Macro-F1: {macro_f1:.4f}')
	print(f'Reports saved to: {output_dir}')


def main():
	parser = argparse.ArgumentParser(
		description='Evaluate AgriSmart AI on an unseen directory-structured dataset.'
	)
	parser.add_argument(
		'--data-dir',
		required=True,
		type=Path,
		help='Dataset directory containing one subdirectory per class.',
	)
	parser.add_argument(
		'--output-dir',
		type=Path,
		default=PROJECT_ROOT / 'report' / 'external_evaluation',
		help='Directory for evaluation reports and the confusion matrix.',
	)
	args = parser.parse_args()
	evaluate(args.data_dir, args.output_dir)


if __name__ == '__main__':
	main()
