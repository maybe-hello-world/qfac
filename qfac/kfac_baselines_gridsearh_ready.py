import tensorflow as tf
import baselines.common.tf_util as u
import gym
import numpy as np

from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput
from baselines.common.schedules import LinearSchedule

import kfac

"""
Examples

_optimizer = "kfac"  # "adam" or "kfac"
_gym_env = "CartPole-v1" # any valid OpenAI discrete Gym environment

_common_opts = {
	"OVERALL_STEPS": 30000,					# overall number of learning steps
	"UPDATE_TARGET_EVERY": 1000,			# how often to transfer weights from online to target
	"REPLAY_BUFFER_SAMPLE_SIZE": 100,		# sample size for off-policy training
	"WARMUP_TIME": 1000,					# steps to play before training for buffer filling
	"LINEAR_SCHEDULE_LENGTH": 10000,		# steps before exploration value decreasing to min_value
	"REPLAY_BUFFER_SIZE": 50000,			# size of replay buffer
}

# see KFAC article
_kfac_opts = {
	'learning_rate': 3e-3,
	'cov_update_every': 1,
	'invert_every': 100,
	'cov_ema_decay': 0.99,
	'momentum': 0.6,
	'damping': 1e-2
}

_adam_opts = {
	'learning_rate': 5e-4
}


"""


def learn_cycle(
		_optimizer: str = "kfac",
		_gym_env: str = "CartPole-v1",
		_common_opts: dict = None,
		_optimizer_opts: dict = None,
		_seed: int = 1
):
	tg = tf.Graph()

	if not _optimizer:
		raise ValueError("Optimizer not specified")

	if not _gym_env:
		raise ValueError("Gym environment not specified")

	if _common_opts is None:
		raise ValueError("Common options are not specified, see example in docstring.")

	if _optimizer_opts is None:
		raise ValueError("Optimizer options are not specified, see example in docstring.")

	# seeds
	np.random.seed(_seed)
	tf.random.set_random_seed(_seed)

	# define optimizator for kfac
	def get_kfac_optimizer(_opts, _layer_collection):
		return kfac.PeriodicInvCovUpdateKfacOpt(
			**_opts,
			layer_collection=_layer_collection,
			var_list=_layer_collection.registered_variables,
		)

	# model with kfac-registration
	def model(inpt, num_actions, scope, lc, reuse=False, register=False):
		"""This model takes as input an observation and returns values of all actions."""
		with tf.variable_scope(scope, reuse=reuse):
			layer1 = tf.layers.Dense(64, name="Dense1", activation=None)
			preact1 = layer1(inpt)
			params1 = layer1.kernel, layer1.bias
			if register:
				lc.register_fully_connected(
					params=params1,
					inputs=inpt,
					outputs=preact1,
					reuse=None
				)
			act1 = tf.nn.tanh(preact1, name="Act1")

			layer2 = tf.layers.Dense(16, name="Dense2", activation=None)
			preact2 = layer2(act1)
			params2 = layer2.kernel, layer2.bias
			if register:
				lc.register_fully_connected(
					params=params2,
					inputs=act1,
					outputs=preact2,
					reuse=None
				)
			act2 = tf.nn.tanh(preact2, name="Act2")

			layer3 = tf.layers.Dense(num_actions, name="Dense3", activation=None)
			preact3 = layer3(act2)
			params3 = layer3.kernel, layer3.bias
			if register:
				lc.register_fully_connected(
					params=params3,
					inputs=act2,
					outputs=preact3,
					reuse=None
				)
			act3 = preact3  # linear
			return act3

	# build actuator - model for decision taking
	# as is from baselines
	def build_act(make_obs_ph, q_func, num_actions, lc, scope="deepq", reuse=None):
		with tf.variable_scope(scope, reuse=reuse):
			observations_ph = make_obs_ph("observation")
			stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
			update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

			eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

			q_values = q_func(observations_ph.get(), num_actions, scope="q_func", lc=lc)
			deterministic_actions = tf.argmax(q_values, axis=1)

			batch_size = tf.shape(observations_ph.get())[0]
			random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
			chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
			stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

			output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
			update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))
			_act = u.function(
				inputs=[observations_ph, stochastic_ph, update_eps_ph],
				outputs=output_actions,
				givens={update_eps_ph: -1.0, stochastic_ph: True},
				updates=[update_eps_expr])

			def act_(ob, stochastic=True, update_eps=-1):
				return _act(ob, stochastic, update_eps)
			return act_

	# build train graph
	# modified for k-fac
	def build_train(make_obs_ph, q_func, num_actions, optimizer, lc, gamma=1.0, double_q=True, scope="deepq", reuse=None):
		act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, reuse=reuse, lc=lc)

		with tf.variable_scope(scope, reuse=reuse):
			# set up placeholders
			obs_t_input = make_obs_ph("obs_t")
			act_t_ph = tf.placeholder(tf.int32, [None], name="action")
			rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
			obs_tp1_input = make_obs_ph("obs_tp1")
			done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
			importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

			# q network evaluation
			# reuse parameters from act
			q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", reuse=True, register=True, lc=lc)
			q_func_vars = tf.get_collection(
				tf.GraphKeys.GLOBAL_VARIABLES,
				scope=tf.get_variable_scope().name + "/q_func"
			)

			# target q network evalution
			q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func", lc=lc)
			target_q_func_vars = tf.get_collection(
				tf.GraphKeys.GLOBAL_VARIABLES,
				scope=tf.get_variable_scope().name + "/target_q_func"
			)

			# q scores for actions which we know were selected in the given state.
			q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

			# compute estimate of best possible value starting from state at t + 1
			if double_q:
				q_tp1_using_online_net = q_t
				q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
				q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
			else:
				q_tp1_best = tf.reduce_max(q_tp1, 1)
			q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

			# compute RHS of bellman equation
			q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

			# compute the error (potentially clipped)
			td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
			errors = u.huber_loss(td_error)
			weighted_error = tf.reduce_mean(importance_weights_ph * errors)

			if optimizer == "kfac":
				lc.register_squared_error_loss(errors, seed=_seed)
				opt = get_kfac_optimizer(_optimizer_opts, lc)
			elif optimizer == "adam":
				opt = tf.train.AdamOptimizer(**_optimizer_opts)
			elif optimizer == "rmsprop":
				opt = tf.train.RMSPropOptimizer(**_optimizer_opts)
			else:
				raise ValueError("Unknown optimizer")
			optimize_expr = opt.minimize(weighted_error, var_list=q_func_vars)

			# update_target_fn will be called periodically to copy Q network to target Q network
			update_target_expr = []
			for var, var_target in zip(
					sorted(q_func_vars, key=lambda v: v.name),
					sorted(target_q_func_vars, key=lambda v: v.name)
			):
				update_target_expr.append(var_target.assign(var))
			update_target_expr = tf.group(*update_target_expr)

			# Create callable functions
			_train = u.function(
				inputs=[
					obs_t_input,
					act_t_ph,
					rew_t_ph,
					obs_tp1_input,
					done_mask_ph,
					importance_weights_ph
				],
				outputs=td_error,
				updates=[optimize_expr]
			)
			_update_target = u.function([], [], updates=[update_target_expr])

			q_values = u.function([obs_t_input], q_t)

			return act_f, _train, _update_target, {'q_values': q_values}

	# create environment
	env = gym.make(_gym_env)
	env.seed(_seed)
	obs = env.reset()
	episode_rewards = [0.0]

	# create layer collection for kfac
	lc_ = kfac.LayerCollection(graph=tg)

	# create session
	with tf.Session(graph=tg):

		# Create all the functions necessary to train the model
		act, train, update_target, debug = build_train(
			make_obs_ph=lambda name: ObservationInput(env.observation_space, name=name),
			q_func=model,
			num_actions=env.action_space.n,
			optimizer=_optimizer,
			lc=lc_
		)

		# Create the replay buffer
		replay_buffer = ReplayBuffer(_common_opts['REPLAY_BUFFER_SIZE'])

		# Create the schedule for exploration starting from 1 (every action is random) down to
		# 0.02 (98% of actions are selected according to values predicted by the model).
		exploration = LinearSchedule(schedule_timesteps=_common_opts['LINEAR_SCHEDULE_LENGTH'], initial_p=1.0, final_p=0.02)

		# init params
		u.initialize()

		# copy params to target network
		update_target()

		# leaaaaaaaaarn
		for t in range(_common_opts['OVERALL_STEPS']):
			# Take action and update exploration to the newest value
			action = act(obs[None], update_eps=exploration.value(t))[0]
			new_obs, rew, done, _ = env.step(action)

			# Store transition in the replay buffer.
			replay_buffer.add(obs, action, rew, new_obs, float(done))
			obs = new_obs

			episode_rewards[-1] += rew
			if done:
				obs = env.reset()
				episode_rewards.append(0)

			# Minimize the error in Bellman's equation on a batch sampled from replay buffer.
			if t > _common_opts['WARMUP_TIME']:
				obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(_common_opts['REPLAY_BUFFER_SAMPLE_SIZE'])
				train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))

			# Update target network periodically.
			if t % _common_opts['UPDATE_TARGET_EVERY'] == 0:
				update_target()

	# get hash for combining results with different seed
	_hcommon_opts = hash(frozenset(_common_opts))  # it's valid since it's dictionary of strings and numbers only
	_hoptimizer_opts = hash(frozenset(_optimizer_opts))  # similar
	_attempt_hash = hash((_optimizer, _gym_env, _hcommon_opts, _hoptimizer_opts))
	return _attempt_hash, (_optimizer, _gym_env, _common_opts, _optimizer_opts), tuple(episode_rewards)


if __name__ == '__main__':
	import pickle
	import os
	import datetime

	filename = os.getenv("L_FILENAME")
	if filename is None:
		raise ValueError("L_FILENAME env variable is empty.")
	if not os.path.exists(filename):
		raise ValueError(f"L_FILENAME doesn't exist. L_FILENAME value: {filename}")
	with open(filename, "rb") as f:
		configs = pickle.load(f)
	
	print(f"{filename} successfully loaded, gridsearch starting...")

	results = []
	for i, conf_tuple in enumerate(configs):
		print(f"{i+1}/{len(configs)} started, time: {datetime.datetime.now()}")
		attempt_hash, opts, rews = learn_cycle(*conf_tuple)
		results.append((attempt_hash, opts, rews))
		print(f"{i+1}/{len(configs)} finished, time: {datetime.datetime.now()}")

	with open(filename + ".results", "wb") as f:
		pickle.dump(results, f)
