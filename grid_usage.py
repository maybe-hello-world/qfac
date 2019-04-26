# gridsearch test

from qfac import gridsearch_creator

_common_opts = {
	"OVERALL_STEPS": [6 * 100000],					# overall number of learning steps
	"UPDATE_TARGET_EVERY": [1000],			# how often to transfer weights from online to target
	"WARMUP_TIME": [10000],					# steps to play before training for buffer filling
	"LINEAR_SCHEDULE_LENGTH": [50000],		# steps before exploration value decreasing to min_value
	"REPLAY_BUFFER_SIZE": [100000],			# size of replay buffer
}

_kfac_opts = {
	'learning_rate': [1e-4],
	'cov_ema_decay': [0.95, 0.99],
	'momentum': [0.9, 0.95],
	'damping': [0.03]
}

_adam_opts = {
	'learning_rate': [1e-3, 1e-4],
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
	_gym_env=['BreakoutDeterministic-v4'],
	_optimizer_opts=_adam_opts,
	_common_opts=_common_opts,
	_seed=[i for i in range(6)]
)
