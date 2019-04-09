import matplotlib.pyplot as plt
import numpy as np
from typing import Any


def visualise(
		results_list: Any,
		border: int = None,
		color: str = 'blue',
		title: str = None,
		show_std: bool = True,
		legend: bool = False,
		labels: list = None,
		clipping: int = None
) -> None:
	"""
	Visualise mean reward plot

	:param results: list of lists of reward lists (or 3D np.ndarray)
	:param border: right border of plot (if None - min(length of all arrays))
	:param color: color of plot (green, blue, red, cyan, magenta, yellow, black, purple, orange, etc.)
	:return: nothing, shows pyplot image
	"""

	plt.rcParams['axes.facecolor'] = 'w'
	plt.figure()
	plt.grid(True)
	for i, results in enumerate(results_list):
		if isinstance(results, list):
			if isinstance(results[0], list) or isinstance(results[0], tuple):
				results = [np.array(i) for i in results]

			# cut all to min length of all
			min_l = min([len(i) for i in results])
			results = [i[:min_l] for i in results]

			results = np.stack(results)
		assert isinstance(results, np.ndarray) and results.ndim == 2,\
			"results should be list of lists, list of arrays or 2D np.ndarray"

		if border is not None and border < results.shape[-1]:
			results = results[:, :border]

		mean_y = np.mean(results, axis=0)
		std_y = np.std(results, axis=0)
		stderr_y = std_y / np.sqrt(results.shape[0])

		xs = np.arange(len(mean_y))

		plt.plot(xs, mean_y, label=labels[i] if labels else None)

		diff = mean_y - stderr_y
		if clipping is not None:
			diff = np.clip(diff, clipping, None)
		plt.fill_between(xs, diff, mean_y + stderr_y, color=color, alpha=.2)
		if show_std: plt.fill_between(xs, np.clip(mean_y - std_y, 0, None), mean_y + std_y, color=color, alpha=.2)
	if title:
		plt.title(title)
	if legend:
		plt.legend()
	plt.show()

