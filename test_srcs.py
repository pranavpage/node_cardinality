import numpy as np
import pydtmc
import matplotlib.pyplot as plt 
import os 
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Layer, Lambda, InputLayer
from tensorflow.keras.metrics import mean_squared_error as mse 
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
# generate the transition matrix

def split_data(data, feature_vec_length=feature_vec_length):
    x = data
    x_student = x[:, :feature_vec_length]
    temp = tf.reshape(x[:, feature_vec_length], [-1, 1])
    x_teacher = tf.concat([x[:, :feature_vec_length-1], temp], 1)
    print(f"Student={x_student}, Teacher={x_teacher}")
    return x_student, x_teacher
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
class Distiller(Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=1,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data
        x_student, x_teacher = split_data(x)
        # Forward pass of teacher
        teacher_predictions = self.teacher(x_teacher, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x_student, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = self.distillation_loss_fn(teacher_predictions,student_predictions)
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data
        x_student, x_teacher = split_data(x)    
        # Compute predictions
        y_prediction = self.student(x_student, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results
    
    def predict_step(self, data):
        x_student, x_teacher = split_data(data)
        return ({"student_prediction": self.student(x_student, training=False)})

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
    if(z):
        return np.log(z/l)/(np.log(1-p_participate/l))
    else:
        print("z=0")
        return max_num_nodes
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
def student_info(feature_vec):
    # returns a student input vec
    bm = feature_vec[:, :-2]
    bm *= max_num_nodes/(bm.shape[1])
    bm = tf.where(bm>1, 1.0, bm/2)
    bm = tf.concat([bm, feature_vec[:, -2:]], -1)
    return bm
def run_sim(mc, num_iters, l, ID_bits, model, tag, split, seed = 0):
    # runs a full length simulation of evolving node cardinalities
    # first run LoF to get started 
    # LoF with l = log2(max_num_nodes) slots
    perf = np.zeros(3)
    decay_perf = np.zeros(3)
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
            # nhat = n_truth_prev # by NN
        if(nhat>0): 
            p_participate = min(1, 1.6*l/nhat)
        else:
            p_participate=1
        # Run balls-and-bins for current slot
        bnb_bm = sim_balls_and_bins(n_truth, p_participate, l, seed=i)
        bnb_estimate = est_balls_and_bins(bnb_bm, p_participate)
        # train NN with bnb_bm, nhat, l, n_bnb to predict n_truth
        feature_vec = normalize_feature_vec(bnb_bm, nhat, bnb_estimate)
        feature_vec = np.append(feature_vec, n_truth_prev/max_num_nodes)
        data_vec = np.append(feature_vec, n_truth/max_num_nodes)
        feature_vec = np.reshape(feature_vec, (1, feature_vec_length+1))
        decay_name = f"./data/decay_{tag}.csv"
        if(i==200):
            test_feature_vec = feature_vec
            n_test_truth = n_truth
        if(i>=200 and i%100 == 0):
            test_prediction = model.predict(test_feature_vec, verbose=-1)
            n_test_prediction = max_num_nodes*test_prediction["student_prediction"]
            decay_perf[0]=i
            decay_perf[1]=n_test_prediction
            decay_perf[2]=n_truth
            print(f"i={i}, NN={n_test_prediction[0][0]}, actual = {n_test_truth} \n")
            with open(decay_name, 'a') as de:
                writer=csv.writer(de)
                writer.writerow(decay_perf)
        dict_prediction = model.predict(feature_vec, verbose=-1)
        n_prediction = max_num_nodes*dict_prediction["student_prediction"]
        # n_prediction = max_num_nodes*model.predict(feature_vec, verbose=-1)
        perf[0] = (n_prediction[0][0] - n_truth)/n_truth
        perf[1] = (bnb_estimate - n_truth)/n_truth
        perf[2] = n_truth
        print(f"Step={i} : Actual={n_truth}, Predicted={n_prediction[0][0]:.2f}, BnB estimate = {bnb_estimate:.2f}", end='\r')
        fname = f"./data/student_{tag}.csv"
        with open(fname, 'a') as f:
            writer=csv.writer(f)
            writer.writerow(perf)
        dname = f"./data/train_{tag}.csv"
        with open(dname, 'a') as d:
            writer=csv.writer(d)
            writer.writerow(data_vec)
        
        if(i<split*num_iters):
            y = np.array(n_truth/max_num_nodes)
            y = np.reshape(y, (1,1))
            model.fit(feature_vec, y, epochs=1, verbose=1)
    return 0
if __name__== "__main__":
    max_num_nodes = 2**(8)
    q = 0.2
    p = (1-q)/2
    jumps= 5
    min_active_nodes = 10
    num_runs = 1
    num_iters = int(1e3)
    split = 0.9
    length_of_trial = 50
    tag = f"two_l{int(length_of_trial)}_j{jumps}_n{num_iters}"
    states = [str(i) for i in range(min_active_nodes, max_num_nodes)]
    # eps = 0.1
    # length_of_trial = int(65/((1-0.04**eps)**2))
    print(f"Length of trial is l={length_of_trial:.2f}")
    print(f"Epsilon for trial = {np.log(1 - ((65/length_of_trial)**0.5))/np.log(0.04):.3f}")
    tpm = gen_transition_matrix(max_num_nodes - min_active_nodes, p, q)
    tpm = np.linalg.matrix_power(tpm, jumps)
    mc = pydtmc.MarkovChain(tpm, states)
    curr_state = str(int(max_num_nodes/2))
    feature_vec_length = length_of_trial+2
    ctag = tag + "_r0"
    if(os.path.exists(f"./data/student_{ctag}.csv")):
        os.remove(f"./data/student_{ctag}.csv")
    if(os.path.exists(f"./data/train_{ctag}.csv")):
        os.remove(f"./data/train_{ctag}.csv")
    if(os.path.exists(f"./data/decay_{ctag}.csv")):
        os.remove(f"./data/decay_{ctag}.csv")

    teacher = load_model(f"./models/model_two_l50_j5_n50000_r0")
    print(f"Loaded ./models/model_two_l50_j5_n50000_r0 as teacher")
    for layer in teacher.layers:
        layer.trainable = False

    student = Sequential()
    student.add(InputLayer(input_shape=(feature_vec_length, )))
    student.add(Lambda(student_info, output_shape = (feature_vec_length, )))
    student.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
    student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    student.add(Dense(1, activation='linear'))
    learning_rate = 2e-3
    momentum = 0
    opt = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum = 0
    )
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=opt,    
        student_loss_fn=MeanSquaredError(),
        distillation_loss_fn=MeanSquaredError(),
        alpha=0.1,
        temperature=1,
    )
    run_sim(mc, num_iters, length_of_trial, 8, distiller, ctag,split, 3)
    print(f" data in ./data/student_{ctag}.csv")

    ## Plotting 
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