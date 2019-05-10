import gym
from gym.core import ObservationWrapper
import numpy as np


class PreprocessNothing(ObservationWrapper):
	"""For environments that do not need to be preprocessed"""
	def observation(self, obs):
		return obs


class PreprocessMountainCar(ObservationWrapper):
	def __init__(self, env):
		super(PreprocessMountainCar, self).__init__(env)

		self.epsmin = 0.000001
		self.epsmax = 1 - self.epsmin

		self.low = np.array([self.epsmin, self.epsmin])
		self.high = np.array([self.epsmax, self.epsmax])

		self.observation_space = gym.spaces.Box(self.low, self.high, dtype=np.float32)


	def observation(self, observation):
		position, velocity = observation

		# -1.2 <= position <= 0.6, let's normalize to 0..1
		position += 0.3
		position /= 2
		position += 0.5
		position = np.clip(position, self.epsmin, self.epsmax)

		# -0.07 <= speed <= 0.07
		velocity *= 7
		velocity += 0.5
		velocity = np.clip(velocity, self.epsmin, self.epsmax)
		return position, velocity


class PreprocessBreakout(ObservationWrapper):
	@staticmethod
	def _to_grayscale(img):
		"""There's a theory that grayscale with emphasized red channel works better"""
		return np.dot(img[..., :3], [0.8, 0.1, 0.1])

	@staticmethod
	def _crop(img, margins = (31, 8, 8, 15)):
		"""Remove unnecessary parts of image"""
		return img[margins[0]:-margins[-1], margins[1]:-margins[2]]

	@staticmethod
	def _resize(img):
		""" Simple downsampling to (82, 72)"""
		return img[::2, ::2]

	@staticmethod
	def _to_float(img):
		"""More memory to the god of the memory"""
		return np.asarray(img, dtype=np.float64) / 255.0

	def observation(self, img):
		img = self._to_grayscale(img)
		img = self._crop(img)
		img = self._resize(img)
		img = self._to_float(img)
		return img

	def __init__(self, env):
		super(PreprocessBreakout, self).__init__(env)

		self.img_size = (82, 72)  # Eee, magic constants (see crop + downsampling)
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=(*self.img_size, 1), dtype=np.float64)


