# gridsearch test

from qfac import gridsearch_creator

_common_opts = {
	"OVERALL_STEPS": [600000]
}

_kfac_opts = {
	'learning_rate': [1e-4, 5e-4, 1e-3],
	'cov_ema_decay': [0.95],
	'momentum': [0.9],
	'damping': [0.01, 0.05]
}

_adam_opts = {
	'learning_rate': [1e-3],
	'beta1': [0.5, 0.9],
	'beta2': [0.99, 0.999],
	'epsilon': [1e-08]
}

_rmsprop_opts = {
	'learning_rate': [1e-4, 1e-3],
	'decay': [0.95, 0.99],
	'momentum': [0.95, 0.99],
}

gridsearch_creator.create_grid(
	_optimizer="adam",
	filename="task",
	n_hosts=4,
	n_gpus=2,
	_gym_env=['Breakout-v0'],
	_optimizer_opts=_adam_opts,
	_common_opts=_common_opts,
	_seed=[i for i in range(20)]
)
