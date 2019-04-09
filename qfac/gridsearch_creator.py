from itertools import product
import pickle

_optimizer_orig = "kfac"  # "adam" or "kfac"
_gym_env_orig = "CartPole-v1"  # any valid OpenAI discrete Gym environment

_common_opts_orig = {
	"OVERALL_STEPS": 150000,					# overall number of learning steps
	"UPDATE_TARGET_EVERY": 1000,			# how often to transfer weights from online to target
	"REPLAY_BUFFER_SAMPLE_SIZE": 100,		# sample size for off-policy training
	"WARMUP_TIME": 1000,					# steps to play before training for buffer filling
	"LINEAR_SCHEDULE_LENGTH": 10000,		# steps before exploration value decreasing to min_value
	"REPLAY_BUFFER_SIZE": 30000,			# size of replay buffer
}

_kfac_opts_orig = {
	'learning_rate': 3e-3,
	'cov_update_every': 1,
	'invert_every': 100,
	'cov_ema_decay': 0.99,
	'momentum': 0.6,
	'damping': 1e-2
}

_adam_opts_orig = {
	'learning_rate': 5e-4
}

_rmsprop_opts_orig = {
	'learning_rate': 25e-5,
	'decay': 0.99,
	'momentum': 0.95,
	'epsilon': 1e-10
}

_opt_options_choice = {
	"kfac": _kfac_opts_orig,
	"adam": _adam_opts_orig,
	"rmsprop": _rmsprop_opts_orig
}

_seed_orig = 1


def create_grid(
		filename: str,
		n_hosts: int, n_gpus: int,
		_optimizer: str = None,
		_gym_env: list = None,
		_common_opts: dict = None,
		_optimizer_opts: dict = None,
		_seed: list = None,
		_out_folder: str = "tasks/"
) -> None:
	"""
	Create grid and save params to files for SLURM.

	If any optional param is None then default value will be used (see module source for details)

	:param filename: filename template for saving
	:param n_hosts: number of SLURM hosts
	:param n_gpus: number of GPUs on each host
	:param _optimizer: chosen optimizers
	:param _gym_env: list of OpenAI Gym discrete environments
	:param _common_opts: dictionary of common options, every option can be a list of values,
							missing options will be initialized with default values
	:param _optimizer_opts: dictionary of optimizer options, every option can be a list of values,
							missing options will be initialized with default values
	:param _seed: list of seeds
	:param _out_folder: where to save tasks for computing
	:return: None
	"""

	def parse_dict(_dict: dict, _original: dict):
		if _dict is None:
			return [_original]

		cart_product_list = [dict(zip(_dict, v)) for v in product(*_dict.values())]
		answer = [_original.copy() for _ in cart_product_list]
		for i, _cart in enumerate(cart_product_list):
			answer[i].update(_cart)
		return answer

	_optimizer = _optimizer or "kfac"
	_gym_env_list = _gym_env or [_gym_env_orig]
	_seed_list = _seed or [_seed_orig]

	# get values for common options
	_common_opts_list = parse_dict(_common_opts, _common_opts_orig)

	# get optimizer and parse it's options
	_optimizer_opts_orig_list = _opt_options_choice[_optimizer]
	_optimizer_opts_list = parse_dict(_optimizer_opts, _optimizer_opts_orig_list)

	# get cartesian product of all lists
	product_list = list(product(
		[_optimizer],
		_gym_env_list, _common_opts_list,
		_optimizer_opts_list, _seed_list
	))

	# split to chunks
	n_chunks = n_hosts * n_gpus
	product_list = [product_list[i::n_chunks] for i in range(n_chunks)]

	# write to filesystem
	for i in range(n_hosts):
		for j in range(n_gpus):
			with open(_out_folder + filename + f"{i}_{j}.pkl", "wb") as f:
				pickle.dump(product_list[i * n_gpus + j], f)

	return
