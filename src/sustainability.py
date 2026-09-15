"""Offline sustainability scoring for AgriSmart AI.

The Sustainability Score is an explainable 0-100 measure based on water
efficiency, crop health, and fertilizer/resource use. The inputs are actual
water used in liters, a positive reference maximum water amount, a boolean
indicating whether disease was detected, and fertilizer/resource use on a
0-100 scale. The components are weighted 40% water efficiency, 40% crop
health, and 20% resource efficiency.

Badges use these non-overlapping thresholds: scores above 85 are
"Platinum", scores above 70 are "Green Guardian", scores below 50 are
"At Risk", and all remaining scores are "Needs Improvement".

The public function returns a dictionary containing the final score, badge,
each component score, and one general actionable suggestion. This module is
offline and uses no machine learning, datasets, databases, or external APIs.
"""

import math
from numbers import Real


def sustainability_score(
	water_used_liters: float,
	max_water: float,
	disease_detected: bool,
	fertilizer_level: float,
) -> dict:
	"""Calculate an explainable sustainability score from 0 to 100.

	Args:
		water_used_liters: Actual water used by the crop or farm in liters.
		max_water: Positive reference water amount in liters.
		disease_detected: True when disease is detected, otherwise False.
		fertilizer_level: Fertilizer/resource use normalized from 0 to 100.

	Returns:
		A dictionary with ``score``, ``badge``, ``water_score``,
		``health_score``, ``resource_score``, and ``suggestion`` keys.

	Raises:
		ValueError: If any input is outside its required range or has an
			invalid type.
	"""
	if (
		isinstance(water_used_liters, bool)
		or not isinstance(water_used_liters, Real)
		or not math.isfinite(water_used_liters)
		or water_used_liters < 0
	):
		raise ValueError('water_used_liters must be greater than or equal to 0.')
	if (
		isinstance(max_water, bool)
		or not isinstance(max_water, Real)
		or not math.isfinite(max_water)
		or max_water <= 0
	):
		raise ValueError('max_water must be greater than 0.')
	if not isinstance(disease_detected, bool):
		raise ValueError('disease_detected must be a boolean.')
	if (
		isinstance(fertilizer_level, bool)
		or not isinstance(fertilizer_level, Real)
		or not math.isfinite(fertilizer_level)
		or not 0 <= fertilizer_level <= 100
	):
		raise ValueError('fertilizer_level must be between 0 and 100.')

	water_efficiency = max(0.0, 1.0 - (water_used_liters / max_water))
	water_score = water_efficiency * 100.0
	health_score = 0.0 if disease_detected else 100.0
	resource_score = 100.0 - fertilizer_level

	score = (
		water_score * 0.40
		+ health_score * 0.40
		+ resource_score * 0.20
	)
	score = min(100.0, max(0.0, score))

	if score > 85:
		badge = 'Platinum'
	elif score > 70:
		badge = 'Green Guardian'
	elif score < 50:
		badge = 'At Risk'
	else:
		badge = 'Needs Improvement'

	component_scores = {
		'water_score': water_score,
		'health_score': health_score,
		'resource_score': resource_score,
	}
	weakest_component = min(component_scores, key=component_scores.get)
	suggestions = {
		'water_score': 'Consider reducing unnecessary irrigation/water use.',
		'health_score': 'Monitor crop health and address detected disease.',
		'resource_score': 'Consider reducing unnecessary fertilizer/resource use.',
	}

	return {
		'score': round(score, 2),
		'badge': badge,
		'water_score': round(water_score, 2),
		'health_score': round(health_score, 2),
		'resource_score': round(resource_score, 2),
		'suggestion': suggestions[weakest_component],
	}


if __name__ == '__main__':
	examples = {
		'healthy and efficient': (200, 1000, False, 20),
		'diseased crop': (500, 1000, True, 30),
		'very high water usage': (1500, 1000, False, 20),
		'very high fertilizer use': (500, 1000, False, 100),
		'balanced example': (500, 1000, False, 50),
		'boundary values': (0, 1000, True, 0),
	}

	for name, inputs in examples.items():
		print(f'{name}: {sustainability_score(*inputs)}')

	invalid_inputs = [
		(-1, 1000, False, 20),
		(500, 0, False, 20),
		(500, 1000, 'no', 20),
		(500, 1000, False, 101),
	]
	for inputs in invalid_inputs:
		try:
			sustainability_score(*inputs)
		except ValueError as error:
			print(f'invalid input {inputs}: {error}')
		else:
			raise AssertionError(f'Expected ValueError for inputs: {inputs}')

	for inputs in examples.values():
		result = sustainability_score(*inputs)
		assert 0 <= result['score'] <= 100

	print('All sustainability score checks passed.')
