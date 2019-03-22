import matplotlib.pyplot as plt
import numpy as np
from typing import Union, List


def visualise(
		results: Union[List[Union[list, tuple, np.ndarray]], np.ndarray],
		border: int = None,
		color: str = 'blue',
) -> None:
	"""
	Visualise mean reward plot

	:param results: list of reward lists (or 2D np.ndarray)
	:param border: right border of plot (if None - min(length of all arrays))
	:param color: color of plot (green, blue, red, cyan, magenta, yellow, black, purple, orange, etc.)
	:return: nothing, shows pyplot image
	"""

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
	stderr_y = std_y / np.sqrt(results.shape(0))

	xs = np.arange(len(mean_y))

	plt.figure()
	plt.plot(xs, mean_y)
	plt.fill_between(xs, np.clip(mean_y - stderr_y, 0, None), mean_y + stderr_y, color=color, alpha=.4)
	plt.fill_between(xs, np.clip(mean_y - std_y, 0, None), mean_y + std_y, color=color, alpha=.2)
	plt.show()

