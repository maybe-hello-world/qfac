import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

RESULT_FOLDER = ["results_test_cnn"]

all_res = []
for folder in RESULT_FOLDER:
	for file in os.listdir(folder):
		with open(folder + "/" + file, "rb") as f:
			all_res.append(pickle.load(f))


# parse 2 level lists
uncovered_list = [j for i in all_res for j in i]

# sanity check
overall_count = sum(len(x) for x in all_res)
print(overall_count == len(uncovered_list))


# get 250-epoch club
#uncovered_list = [i for i in uncovered_list if np.max(i[1][:250]) > 490]
fig = plt.figure()
for i, a in enumerate(uncovered_list):
	plt.plot(a[-1], label=str(i))

plt.legend()
plt.show()