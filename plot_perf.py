import numpy as np
import matplotlib.pyplot as plt
jumps= 5
num_iters = int(1.5e4)
length_of_trial = 50
learning_rate=2e-3
tag = f"two_l{int(length_of_trial)}_j{jumps}_n{num_iters}"
ctag = tag + "_r0"
perf_vec = np.genfromtxt(f"./data/student_{ctag}.csv", delimiter=",")
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(perf_vec[:,0], label="NN", alpha=1)
ax1.plot(perf_vec[:,1], label="BnB", alpha=0.8, linewidth=0.8)
ax2.plot(perf_vec[:,2], alpha = 0.5, color='g', linewidth=0.8)
ax1.grid()
ax1.legend()
ax1.set_xlabel("Timeslots")
ax1.set_ylabel("Relative error")
ax2.set_ylabel("Number of nodes", color='g')
plt.savefig(f"./plots/student_sgd_{ctag}.png")

decay_vec = np.genfromtxt(f"./data/decay_{ctag}.csv", delimiter=",")
plt.figure()
plt.plot(decay_vec[:, 0], decay_vec[:, 1], label = "Prediction")
plt.plot(decay_vec[:, 0], decay_vec[:, 2], label = "Present Truth")
plt.axhline(decay_vec[0,2], label="Actual truth")
plt.xlabel("Iteration")
plt.ylabel("Number of nodes")
plt.legend()
plt.grid()
plt.title(f"Deterioration of performance with training (SGD, lr={learning_rate:.1e})")
plt.savefig(f"./plots/decay_sgd_{ctag}.png")