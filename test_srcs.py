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
from config import *
# generate the transition matrix

def split_data(data, feature_vec_length = feature_vec_length):
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
def mc_timeseries(mc, num_iters, curr_state, num_anomalies, seed):
    anomaly_start_times = np.random.choice(num_iters, num_anomalies, replace=False, seed=seed)
    curr_time = 0
    simulated_timeseries = []
    for i in range(num_anomalies):
        # start 
        a_time = anomaly_start_times[i]
        timeseries = mc.simulate(a_time - curr_time, curr_state, seed)
        anomaly_sequence = np.linspace(int(timeseries[-1]), max_num_nodes, 5)
        anomaly_sequence = [int(i) for i in anomaly_sequence]
        simulated_timeseries+=anomaly_sequence
        curr_time = a_time
        curr_state = max_num_nodes
    # anomalies done
    
    return 0
def run_sim(mc, num_iters, l, ID_bits, model, tag, split, curr_state=curr_state, seed = 0,\
    fit_after_train=False, store_train=True, track_decay=False, add_n_truth_prev=False, is_teacher=True):
    # runs a full length simulation of evolving node cardinalities
    # first run LoF to get started 
    # LoF with l = log2(max_num_nodes) slots
    perf = np.zeros(3)
    decay_perf = np.zeros(3)
    steps = mc.simulate(num_iters, curr_state, seed=seed)
    feature_vec_length = l+2+(add_n_truth_prev)
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
        if(add_n_truth_prev):
            feature_vec = np.append(feature_vec, n_truth_prev/max_num_nodes)
        data_vec = np.append(feature_vec, n_truth/max_num_nodes)
        feature_vec = np.reshape(feature_vec, (1, feature_vec_length))
        decay_name = f"./data/decay_{tag}.csv"
        if(track_decay):
            if(i==200):
                test_feature_vec = feature_vec
                n_test_truth = n_truth
            if(i>=200 and i%100 == 0):
                test_prediction = model.predict(test_feature_vec, verbose=-1)
                if(is_teacher):
                    n_test_prediction = max_num_nodes*test_prediction["prediction"]
                else:
                    n_test_prediction = max_num_nodes*test_prediction["student_prediction"]
                decay_perf[0]=i
                decay_perf[1]=n_test_prediction
                decay_perf[2]=n_truth
                print(f"i={i}, NN={n_test_prediction[0][0]}, actual = {n_test_truth} \n")
                with open(decay_name, 'a') as de:
                    writer=csv.writer(de)
                    writer.writerow(decay_perf)
        dict_prediction = model.predict(feature_vec, verbose=-1)
        if(is_teacher):
            n_prediction = max_num_nodes*dict_prediction
        else:
            n_prediction = max_num_nodes*dict_prediction["student_prediction"]
        # n_prediction = max_num_nodes*model.predict(feature_vec, verbose=-1)
        perf[0] = (n_prediction[0][0] - n_truth)/n_truth
        perf[1] = (bnb_estimate - n_truth)/n_truth
        perf[2] = n_truth
        print(f"Step={i} : Actual={n_truth}, Predicted={n_prediction[0][0]:.2f}, BnB estimate = {bnb_estimate:.2f}", end='\r')
        fname = f"./data/perf_{tag}.csv"
        with open(fname, 'a') as f:
            writer=csv.writer(f)
            writer.writerow(perf)
        if(store_train):
            dname = f"./data/train_{tag}.csv"
            with open(dname, 'a') as d:
                writer=csv.writer(d)
                writer.writerow(data_vec)
        
        if(fit_after_train):
            if(i<split*num_iters):
                y = np.array(n_truth/max_num_nodes)
                y = np.reshape(y, (1,1))
                model.fit(feature_vec, y, epochs=1, verbose=-1)
    return model
def gen_teacher_data_run_sim(mc, num_iters, l, jumps, ID_bits, tag, split, feature_vec_length = feature_vec_length, seed = 0):
    # runs naive teacher model to generate data
    teacher = Sequential()
    teacher.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
    teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    teacher.add(Dense(1, activation='linear'))
    teacher.compile(loss='mean_squared_error', optimizer='adam')
    ctag = f"teacher_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    # run sim for naive teacher model with training after each prediction
    train_data_fname = f"./data/train_{ctag}.csv"
    if(not os.path.isfile(train_data_fname)):
        model = run_sim(mc, num_iters, l, ID_bits, teacher, ctag, split, fit_after_train=True)
    train_data = np.genfromtxt(train_data_fname, delimiter=',')
    print(train_data.shape)
    return teacher
def train_teacher_offline(num_iters, l, jumps, tag, test_train_split=0.9, epochs=500, batch_size=64):
    fname = f"./data/train_teacher_{tag}_l{int(l)}_j{jumps}_n{num_iters}.csv"
    data = np.genfromtxt(fname, delimiter=",")
    np.random.shuffle(data)
    num_samples = data.shape[0]
    split_sample = int(split*num_samples)
    X = data[:split_sample, :-1]
    y = data[:split_sample, -1]
    X_test = data[split_sample:, :-1]
    y_test = data[split_sample:, -1]
    y_test = np.reshape(y_test, (num_samples-split_sample, 1))
    feature_vec_length = X.shape[1]
    print(X.shape, y.shape)
    print(X_test.shape, y_test.shape)
    
    teacher = Sequential()
    teacher.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
    teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    teacher.add(Dense(1, activation='linear'))
    teacher.compile(loss='mean_squared_error', optimizer='adam')
    
    ctag = f"{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    history = teacher.fit(X,y, validation_data=(X_test, y_test), epochs = epochs, batch_size = batch_size, shuffle=True)
    teacher_model_fname = f"./models/teacher_{ctag}"
    if(not os.path.isdir("./models/")):
        os.makedirs("./models/")
    teacher.save(teacher_model_fname)
    return history, teacher
def gen_student_data_given_teacher_run_sim(teacher, mc, num_iters, l, jumps, ID_bits, tag, split, alpha=0.1, feature_vec_length = feature_vec_length, seed = 1):
    # given teacher, train the student model 
    student = Sequential()
    student.add(InputLayer(input_shape=(feature_vec_length, )))
    student.add(Lambda(student_info, output_shape = (feature_vec_length, )))
    student.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
    student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    student.add(Dense(1, activation='linear'))
    
    ctag = f"{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    teacher_fname = f"./models/teacher_{ctag}"
    teacher = load_model(teacher_fname)
    for layer in teacher.layers:
        layer.trainable = False
    
    distiller = Distiller(student=student, teacher=teacher)
    learning_rate = 1e-3
    momentum = 0
    opt = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
    )
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=opt,    
        student_loss_fn=MeanSquaredError(),
        distillation_loss_fn=MeanSquaredError(),
        alpha=alpha,
        temperature=1,
    )
    
    return 0
def evaluate_student_teacher_run_sim():
    return 0 
if __name__== "__main__":
    exp_name = "reform_teacher"
    tag = f"{exp_name}_l{int(length_of_trial)}_j{jumps}_n{num_iters}"
    states = [str(i) for i in range(min_active_nodes, max_num_nodes)]
    # eps = 0.1
    # length_of_trial = int(65/((1-0.04**eps)**2))
    print(f"Length of trial is l={length_of_trial:.2f}")
    print(f"Epsilon for trial = {np.log(1 - ((65/length_of_trial)**0.5))/np.log(0.04):.3f}")
    tpm = gen_transition_matrix(max_num_nodes - min_active_nodes, p, q)
    tpm = np.linalg.matrix_power(tpm, jumps)
    mc = pydtmc.MarkovChain(tpm, states)
    ctag = tag
    if(os.path.exists(f"./data/perf_{ctag}.csv")):
        os.remove(f"./data/perf_{ctag}.csv")
    # if(os.path.exists(f"./data/train_{ctag}.csv")):
        # os.remove(f"./data/train_{ctag}.csv")
    if(os.path.exists(f"./data/decay_{ctag}.csv")):
        os.remove(f"./data/decay_{ctag}.csv")

    # teacher = load_model(f"./models/model_two_l50_j5_n50000_r0")
    # print(f"Loaded ./models/model_two_l50_j5_n50000_r0 as teacher")
    # for layer in teacher.layers:
    #     layer.trainable = False
    # student = load_model(f"./models/student")
    # print(f"Loaded ./models/student as student")
    # for layer in student.layers:
    #     layer.trainable = False
    # student = Sequential()
    # student.add(InputLayer(input_shape=(feature_vec_length, )))
    # student.add(Lambda(student_info, output_shape = (feature_vec_length, )))
    # student.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
    # student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    # student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
    # student.add(Dense(1, activation='linear'))
    # learning_rate = 2e-3
    # momentum = 0
    # opt = tf.keras.optimizers.Adam(
    #     learning_rate=learning_rate,
    # )
    # distiller = Distiller(student=student, teacher=teacher)
    # distiller.compile(
    #     optimizer=opt,    
    #     student_loss_fn=MeanSquaredError(),
    #     distillation_loss_fn=MeanSquaredError(),
    #     alpha=0.1,
    #     temperature=1,
    # )
    # teacher = gen_teacher_data_run_sim(mc, num_iters, length_of_trial,jumps, ID_bits, exp_name, 0.9)
    history, teacher = train_teacher_offline(num_iters, length_of_trial, jumps, exp_name, epochs=5000)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.ylim(0, 1e-3)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # print(f" data in ./data/student_{ctag}.csv")

    # ## Plotting 
    # perf_vec = np.genfromtxt(f"./data/student_{ctag}.csv", delimiter=",")
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()

    # ax1.plot(perf_vec[:,0], label="NN", alpha=1)
    # ax1.plot(perf_vec[:,1], label="BnB", alpha=0.8, linewidth=0.8)
    # ax2.plot(perf_vec[:,2], alpha = 0.5, color='g', linewidth=0.8)
    # ax1.grid()
    # ax1.legend()
    # ax1.set_xlabel("Timeslots")
    # ax1.set_ylabel("Relative error")
    # ax2.set_ylabel("Number of nodes", color='g')
    # plt.savefig(f"./plots/student_adam_offline_{ctag}.png")

    # decay_vec = np.genfromtxt(f"./data/decay_{ctag}.csv", delimiter=",")
    # plt.figure()
    # plt.plot(decay_vec[:, 0], decay_vec[:, 1], label = "Prediction")
    # plt.plot(decay_vec[:, 0], decay_vec[:, 2], label = "Present Truth")
    # plt.axhline(decay_vec[0,2], label="Actual truth")
    # plt.xlabel("Iteration")
    # plt.ylabel("Number of nodes")
    # plt.legend()
    # plt.grid()
    # plt.title(f"Deterioration of performance with training (SGD, lr={learning_rate:.1e})")
    # plt.savefig(f"./plots/decay_adam_offline_{ctag}.png")