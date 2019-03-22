# gridsearch test

from qfac import gridsearch_creator

_common_opts = {
	"OVERALL_STEPS": [70000]
}

_kfac_opts = {
	'learning_rate': [1e-4, 1e-3, 3e-3],
	'cov_ema_decay': [0.80, 0.9, 0.95],
	'momentum': [0.3, 0.5, 0.7],
	'damping': [3e-1, 3e-2, 3e-3]
}

_adam_opts = {
	'learning_rate': [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
}

_rmsprop_opts = {
	'learning_rate': [1e-4, 25e-5, 5e-4, 1e-3],
	'decay': [0.9, 0.95, 0.99],
	'momentum': [0.9, 0.95, 0.95],
}

gridsearch_creator.create_grid(
	_optimizer="rmsprop",
	filename="task",
	n_hosts=4,
	n_gpus=2,
	_optimizer_opts=_rmsprop_opts,
	_common_opts=_common_opts,
	_seed=[1, 2, 3, 4]
)
