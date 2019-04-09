import pickle
import os
import visualiser
from collections import defaultdict
import hashlib

RESULT_FOLDER = ["results_kfac_long", "results_adam_long", "results_rmsprop_long"]
all_res = []
for folder in RESULT_FOLDER:
	for file in os.listdir(folder):
		with open(folder + "/" + file, "rb") as f:
			all_res.append(pickle.load(f))

all_res = [j for i in all_res for j in i]

results = defaultdict(list)
for i in all_res:
	results[frozenset(sorted(i[1][3].items()))].append(i[-1])

good_results = [
frozenset({('epsilon', 1e-10), ('momentum', 0.95), ('learning_rate', 0.0001), ('decay', 0.95)}),
frozenset({('epsilon', 1e-10), ('momentum', 0.95), ('learning_rate', 0.0001), ('decay', 0.99)}),
frozenset({('beta2', 0.99), ('learning_rate', 0.001), ('beta1', 0.9), ('epsilon', 1e-08)}),
frozenset({('beta2', 0.999), ('learning_rate', 0.001), ('beta1', 0.5), ('epsilon', 1e-08)}),
frozenset({('beta2', 0.99), ('learning_rate', 0.001), ('beta1', 0.5), ('epsilon', 1e-08)}),
frozenset({('cov_update_every', 1), ('momentum', 0.9), ('invert_every', 100), ('learning_rate', 0.001), ('damping', 0.01), ('cov_ema_decay', 0.95)})
]


visualiser.visualise(
    [results[i] for i in good_results],
    color=['green', 'blue', 'orange'],
    show_std=False,
    legend=True,
    labels=["rmsprop-v1", "rmsprop-v2", "adam-v1", "adam-v2", "adam-v3", "k-fac-v1"],
	border=None
)