import pickle
import os
import visualiser
from collections import defaultdict
import hashlib

RESULT_FOLDER = ["results_test_cnn"]
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
]


visualiser.visualise(
    [results[i] for i in good_results],
    color=['green', 'blue', 'orange'],
    show_std=False,
    legend=True,
    labels=["rmsprop-v1", "rmsprop-v2", "adam-v1", "adam-v2", "adam-v3", "k-fac-v1"],
	border=None
)