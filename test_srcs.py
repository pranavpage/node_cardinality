import numpy as np
import pydtmc
import matplotlib.pyplot as plt 
import os 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
# generate the transition matrix
max_num_nodes = 2**(8)
q = 0.2
p = (1-q)/2
jumps= 5
min_active_nodes = 10
num_runs = 2
num_iters = 5000
split = 0.9
length_of_trial = 50
tag = f"two_l{int(length_of_trial)}_j{jumps}_n{num_iters}"
states = [str(i) for i in range(min_active_nodes, max_num_nodes)]
# eps = 0.1
# length_of_trial = int(65/((1-0.04**eps)**2))
print(f"Length of trial is l={length_of_trial:.2f}")

def gen_transition_matrix(n, p, q):
    # generate TPM for n states
    tpm = []
    for i in range(n):
        row = [0]*(n)
        if(i==0):
            row[0] = q
            row[1] = 1-q 
        elif(i==n-1):
            row[n-2] = 1-q
            row[n-1] = q 
        else:
            row[i-1] = 1-p-q
            row[i] = q
            row[i+1] = p
        tpm.append(row)
    return tpm

def sim_balls_and_bins(n, p_participate, l, seed = 0):
    # simulate a balls-and-bins trial of length l, with probability p_participate with n nodes
    # with p_participate, nodes take part in balls-and-bins trial 
    # choose uniformly from l slots
    np.random.seed(seed)
    prob = np.random.rand(n)
    trial_list = [[] for i in range(l)]
    participating_nodes = [i for i in range(n) if prob[i]<p_participate]
    # print(f"Participating nodes : {participating_nodes}")
    choices = np.random.randint(low=0, high=l, size=len(participating_nodes))
    for i in range(len(participating_nodes)):
        trial_list[choices[i]].append(participating_nodes[i])
    trial_arr = np.array([len(elem) for elem in trial_list])
    return trial_arr
def est_balls_and_bins(trial_arr,p_participate):
    # estimates the number of nodes using number of empty slots
    l = len(trial_arr)
    z = np.sum((trial_arr==0))
    # print(f"Number of empty slots = {z}")
    return np.log(z/l)/(np.log(1-p_participate/l))
def geometric_hash(ID, l):
    # l bit nmber ID
    str = format(ID, f'0{l}b')
    ret_hash = -1
    for i in range(l):
        if(str[l-1-i]=='0'):
            ret_hash = i
            break
    if(ret_hash==-1):
        ret_hash = l-1
    return ret_hash
def sim_lottery_frame(n, l, seed = 0):
    # l = log2(max_num_nodes)
    np.random.seed(seed)
    ID_list = np.random.choice(2**l, n, replace=False)
    slot_list = [geometric_hash(i, l) for i in ID_list]
    trial_arr = [slot_list.count(i) for i in range(l)]
    return trial_arr
def est_lottery_frame(trial_arr):
    # R = position of rightmost zero in bitmap
    l = len(trial_arr)
    R = l-1
    for i in range(l):
        if(trial_arr[i]==0):
            R = i
            break
    return int(1.2897*(2**(R)))
def normalize_feature_vec(bm, nhat, bnb_estimate):
    bm = np.array(bm)/(max_num_nodes/len(bm))
    feature_vec = np.array(list(bm)+[nhat/max_num_nodes, bnb_estimate/max_num_nodes])
    # feature_vec = np.array(list(bm))
    return feature_vec
def run_sim(mc, num_iters, l, ID_bits, model, split,tag, seed = 0):
    # runs a full length simulation of evolving node cardinalities
    # first run LoF to get started 
    # LoF with l = log2(max_num_nodes) slots
    perf = np.zeros((1,3))
    curr_state = 120
    steps = mc.simulate(num_iters, curr_state, seed=seed)
    for i in range(num_iters):
        n_truth = int(steps[i])
        n_truth_prev = int(steps[i-1])
        if(i==0):
            # Run Lottery Frame
            # print("Lottery Frame")
            lof_bm = sim_lottery_frame(n_truth, ID_bits)
            nhat = est_lottery_frame(lof_bm)
            # print(f"nhat={nhat}")
        else:
            # use NN for rough estimate of previous slot 
            nhat = n_prediction[0][0] # predicted by NN in previous slot
            # nhat = n_prev by NN
        if(nhat>0): 
            p_participate = min(1, 1.6*l/nhat)
        else:
            p_participate=1
        # Run balls-and-bins for current slot
        bnb_bm = sim_balls_and_bins(n_truth, p_participate, l, seed=i)
        bnb_estimate = est_balls_and_bins(bnb_bm, p_participate)
        # train NN with bnb_bm, nhat, l, n_bnb to predict n_truth
        feature_vec = normalize_feature_vec(bnb_bm, nhat, bnb_estimate)
        X = feature_vec
        y = np.array([n_truth/max_num_nodes])
        X = X.reshape(1, len(X))
        y = y.reshape(1, len(y))
        # print(feature_vec)
        if(i==500):
            print(f"Feature vec for n={n_truth} is {X}, target is {y}")
            X_500 = X
            y_500 = y
        if(i%100==0 and i>500):
            n_500 = model.predict(X_500, verbose=-1)
            print(f"at {i}, Prediction = {n_500[0][0]:.2f}, Actual = {y_500}")
        n_prediction = (model.predict(X, verbose=-1))*max_num_nodes
        if(n_truth):
            perf[0,0] = (n_prediction-n_truth)/n_truth
            perf[0,1] = (bnb_estimate-n_truth)/n_truth
        else:
            perf[0,0] = (n_prediction-n_truth)
            perf[0,1] = (bnb_estimate-n_truth)
        perf[0,2] = n_truth
        perf[0,:] = [str(elem) for elem in perf[0,:]]
        f = open(f"./data/sim_{tag}.csv", 'a')
        np.savetxt(f, perf, delimiter=',')
        # f.write("\n")
        f.close()
        print(f"Step {i} Actual : {n_truth}, Predicted : {n_prediction[0][0]:.2f}", end='\r')
        if(i<num_iters*split):                
            model.fit(X,y,verbose=-1)
    model.save(f"./models/model_{tag}")
    return 0

tpm = gen_transition_matrix(max_num_nodes - min_active_nodes, p, q)
tpm = np.linalg.matrix_power(tpm, jumps)
mc = pydtmc.MarkovChain(tpm, states)
curr_state = str(int(max_num_nodes/2))
feature_vec_length = length_of_trial+2
model = Sequential()
model.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
model.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
model.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='SGD')
model.summary()


split_iter = int(num_iters*split)

for i in range(num_runs):
    ctag = tag+f"_r{i}"
    if(os.path.exists(f"./data/sim_{ctag}.csv")):
        os.remove(f"./data/sim_{ctag}.csv")
    if(i):
        model = load_model(f"./models/model_{tag}_r{i-1}")
        print(f"Loaded ./models/model_{tag}_r{i-1}")
    run_sim(mc, num_iters, length_of_trial, 8, model, split, ctag, i)
    ret = np.genfromtxt(f"./data/sim_{ctag}.csv", delimiter=',')
    print(ret.shape)
    plt.figure()
    avg_error = np.mean((ret[split_iter:,:2]), axis=0)
    std_error = np.std((ret[split_iter:,:2]), axis=0)
    plt.plot(ret[:,0], label=f"NN, mean={avg_error[0]:.1g}, stddev={std_error[0]:.1g}", linewidth=0.6)
    plt.plot(ret[:,1], label=f"BnB, mean={avg_error[1]:.1g}, stddev={std_error[1]:.1g}", alpha=0.9, linewidth=0.6)
    plt.axvline(split_iter, label="Split", linestyle='dashed', color='red')
    plt.legend()
    plt.grid()
    # plt.ylim(-0.5, 0.5)
    plt.xlabel("Timeslots")
    plt.ylabel("Relative error")
    plt.savefig(f"./plots/sim_{ctag}.png")
    plt.figure()
    plt.grid()
    plt.plot(ret[:,2])
    plt.xlabel("Timeslots")
    plt.ylabel("Number of active nodes")
    plt.savefig(f"./plots/truth_{ctag}.png")