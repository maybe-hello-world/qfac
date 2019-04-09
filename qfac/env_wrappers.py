import gym
from gym.core import ObservationWrapper
import numpy as np


class PreprocessNothing(ObservationWrapper):
	"""For environments that do not need to be preprocessed"""
	def observation(self, img):
		return img


class PreprocessBreakout(ObservationWrapper):
	"""There's a theory that grayscale with emphsized red channel works better"""
	def observation(self, img):
		return np.dot(img[..., :3], [0.8, 0.1, 0.1])

	def __init__(self, env):
		super(PreprocessBreakout, self).__init__(env)
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 1), dtype=np.uint8)
