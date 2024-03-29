# Reform the homogenous code as per the updates in the het code
import numpy as np
import pydtmc
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pydtmc, csv, os
import tensorflow as tf
from homo_config import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Layer, Lambda, InputLayer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Model
# generate the transition matrix

def sim_balls_and_bins(n, p_participate, l):
    # simulate a balls-and-bins trial of length l, with probability p_participate with n nodes
    # with p_participate, nodes take part in balls-and-bins trial 
    # choose uniformly from l slots
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
        return n_max
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
def sim_lottery_frame(n, l):
    # l = log2(max_num_nodes)
    # np.random.seed(seed)
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
def srcs(n, l=l, num_lof=num_lof):
    # conduct num_lof Lottery Frames, then balls-and-bins
    lof_est_arr = []
    # print(f"True value = {n:d}")
    for i in range(num_lof):
        trial_arr = sim_lottery_frame(n, ID_bits)
        est_lof = est_lottery_frame(trial_arr)
        # print(f"Actual = {n}, LoF estimate = {est_lof}")
        lof_est_arr+=[est_lof]
    lof_est_arr = np.array(lof_est_arr)
    n_lof_est = np.mean(lof_est_arr)
    # print(f"Average LoF estimate after {num_lof} trials = {n_lof_est:.2f}")
    p_participate = min(1, 1.6*l/n_lof_est)
    trial_arr = sim_balls_and_bins(n, p_participate, l)
    srcs_estimate = est_balls_and_bins(trial_arr, p_participate)
    # print(f"SRCs estimate with {l:d} slots = {srcs_estimate:.2f}")
    return srcs_estimate

def bb_aware(n, prev_estimate, l=l):
    # instead of conducting LoF, we feed previous estimate to BB 
    p_participate = min(1, 1.6*l/prev_estimate)
    trial_arr = sim_balls_and_bins(n, p_participate, l)
    bb_aware_estimate = est_balls_and_bins(trial_arr, p_participate)
    return bb_aware_estimate

def split_data(data, student_len=student_len, teacher_len=teacher_len):
    x_student = data[:, :student_len]
    x_teacher = data[:, student_len:]
    return x_student, x_teacher

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

def get_steps(num_iters = num_iters, l = l, n_max = n_max, n_min = n_min,  jumps = jumps, q=q, seed=0):
    # simulates T types of nodes with 
    states = [str(i) for i in range(n_min, n_max)]
    p=(1-q)/2
    print(f"Length of trial is l={l:.2f}")
    # print(f"Epsilon for trial = {np.log(1 - ((65/l)**0.5))/np.log(0.04):.3f}")
    tpm = gen_transition_matrix(n_max - n_min, p, q)
    tpm = np.linalg.matrix_power(tpm, jumps)
    mc = pydtmc.MarkovChain(tpm, states)
    curr_state = str(int(n_max/2))
    steps = mc.simulate(num_iters, curr_state, seed=seed)
    steps = [int(elem) for elem in steps]
    return np.array(steps)

def bb_trial_arr_to_onehot(trial_arr):
    # trial arr is converted to onehot encoded bitp
    integer_encoded = np.where(trial_arr>1, 2, trial_arr)
    # print(f"Integer encoded = {integer_encoded}")
    onehot_encoded = []
    for val in integer_encoded:
        letter = [0]*3
        letter[val] = 1
        onehot_encoded+=letter
    onehot_encoded = np.array(onehot_encoded)
    # print(f"Onehot encoded = {onehot_encoded}")
    return onehot_encoded

def gen_feature_vectors_for_slot(n_max, l=10, nodes=6, estimates=4, prev_truths=5):
    # given length of trial, number of nodes, previous estimates, generate a feature vector for student and teacher
    p_participate = min(1, 1.6*l/estimates)
    trial_arr = sim_balls_and_bins(nodes, p_participate, l)
    bit_p_onehot = bb_trial_arr_to_onehot(trial_arr)
    feature_vector_student = np.append(bit_p_onehot, np.array(estimates/n_max))
    feature_vector_student = np.append(feature_vector_student, np.array(nodes).flatten()/n_max)
    # print(f"Feature vec for student = {feature_vector_student}")
    
    feature_vector_teacher = trial_arr
    feature_vector_teacher = np.append(feature_vector_teacher/(n_max/l), prev_truths/n_max)
    feature_vector_teacher = np.append(feature_vector_teacher, np.array(nodes).flatten()/n_max)
    # print(f"Feature vec for teacher = {feature_vector_teacher}")
    return feature_vector_student, feature_vector_teacher

def gen_training_data_teacher_run_sim(tag, num_iters = num_iters, l = l,  n_max = n_max, n_min = n_min,  jumps = jumps, q=q, seed = 0):
    ctag = f"train_teacher_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    fname = f"./data/{ctag}.csv"
    rng = np.random.default_rng(seed)
    seed = seed
    if(not os.path.isfile(fname)):
        steps = get_steps(num_iters, l, n_max, n_min, jumps, q, seed)
        estimates = steps[0]
        # print(f"estimates {estimates}")
        prev_truths = steps[0]
        
        feature_vec_length = teacher_len
        teacher = Sequential()
        teacher.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
        teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
        teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
        teacher.add(Dense(1, activation='linear'))
        teacher.compile(loss='mean_squared_error', optimizer='adam')
        
        for i in range(num_iters):
            nodes = steps[i]
            fv_student, fv_teacher = gen_feature_vectors_for_slot(n_max, l, nodes, estimates, prev_truths)
            # print(f"fv sizes : student = {len(fv_student)}, teacher = {len(fv_teacher)} \nlens : student = {student_len}, teacher = {teacher_len}")
            target = fv_teacher[-1]
            X_teacher = fv_teacher[:-1]
            X_teacher = X_teacher.reshape(1, X_teacher.shape[0])
            # print(f"X_teacher shape = {X_teacher.shape}")
            prediction = teacher.predict(X_teacher, verbose = -1)
            err = ((target - prediction[0][0])**2)
            print(f"i={i}/{num_iters}, pred={prediction[0][0]*n_max:3.2f}, target={target*n_max:3.2f}, err = {err:.3e}", end="\r")
            estimates = prediction[0]
            prev_truths = nodes
            y_teacher = target.reshape((1, 1))
            teacher.fit(X_teacher, y_teacher, verbose=-1)
            with open(fname, 'a') as f:
                writer=csv.writer(f)
                writer.writerow(fv_teacher)
    else:
        print(f"Teacher training data already exists \n")
    return

def train_teacher_offline(tag, epochs = 500):
    ctag = f"train_teacher_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    teacher_model_fname = f"./models/teacher_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    if(not os.path.isdir(f"{teacher_model_fname}/")):
        fname = f"./data/{ctag}.csv"
        data = np.genfromtxt(fname, delimiter=',')
        np.random.shuffle(data)
        split = 0.9
        num_samples = data.shape[0]
        split_sample = int(split*num_samples)
        print(data.shape)
        X = data[:split_sample, :-1]
        y = data[:split_sample, -1]
        X_test = data[split_sample:, :-1]
        y_test = data[split_sample:, -1]
        y_test = np.reshape(y_test, (num_samples-split_sample, 1))
        
        feature_vec_length = teacher_len
        teacher = Sequential()
        teacher.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
        teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
        teacher.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
        teacher.add(Dense(1, activation='linear'))
        teacher.compile(loss='mean_squared_error', optimizer='adam')
        
        history = teacher.fit(X,y, validation_data=(X_test, y_test), epochs = epochs, batch_size = 32, shuffle=True)
        teacher.save(teacher_model_fname)
        plt.figure(0)
        plt.plot(history.history['loss'], alpha=0.6)
        plt.plot(history.history['val_loss'], alpha=0.8)
        plt.grid()
        # plt.ylim(0,1e-4)
        plt.yscale('log')
        plt.title('Homogenous nodes : Teacher loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.savefig("./plots/train_het_test_teacher_1.png")
        # plt.show()
        plt.close()
        return teacher
    else:
        print(f"Trained teacher already exists \n")
        teacher = load_model(f"{teacher_model_fname}")
        return teacher

def gen_training_data_student_run_sim(teacher, tag , num_iters = num_iters, l = l,  n_max = n_max,n_min = n_min,  jumps = jumps, q=q, alpha=alpha, seed=100):
    # teacher, mc, num_iters, l, jumps, ID_bits, tag, split, alpha=0.1, feature_vec_length = feature_vec_length, seed = 1
    # given teacher, generate student training data
    ctag = f"train_student_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    fname = f"./data/{ctag}.csv"
    rng = np.random.default_rng(seed)
    if((not os.path.isfile(fname))):
        steps = get_steps(num_iters, l, n_max, n_min, jumps, q, seed)
        estimates = steps[0]
        # print(f"estimates {estimates}")
        prev_truths = steps[0]
        feature_vec_length = student_len
        student = Sequential()
        student.add(Dense(feature_vec_length, input_shape=(feature_vec_length, ), activation='relu'))
        student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
        student.add(Dense(int(feature_vec_length*(0.5)), activation='sigmoid'))
        student.add(Dense(1, activation='linear'))
        student.compile(loss='mean_squared_error', optimizer='adam')
        
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
        
        for i in range(num_iters):
            nodes = steps[i]
            if(not i):
                estimates = np.random.randint(n_min, n_max, 1)
                prev_truths = np.random.randint(n_min, n_max, 1)
            fv_student, fv_teacher = gen_feature_vectors_for_slot(n_max, l, nodes, estimates, prev_truths)
            data_vec = np.append(fv_student[:-1], fv_teacher)
            predict_input = data_vec[:-1]
            predict_input = np.reshape(predict_input, (1, predict_input.shape[0]))
            target = fv_teacher[-1]
            X_teacher = fv_teacher[:-1]
            X_teacher = X_teacher.reshape(1, X_teacher.shape[0])
            prediction = distiller.predict(predict_input, verbose = -1)
            err = ((target - prediction['student_prediction'][0][0])**2)
            print(f"i={i}, pred1={prediction['student_prediction'][0][0]:.3f}, target1={target:.3f}, err={err:.3e}", end="\r")
            estimates = prediction['student_prediction'][0][0]*n_max
            prev_truths = target*n_max
            target = np.reshape(target, (1,1))
            distiller.fit(predict_input, target, verbose=-1)
            with open(fname, 'a') as f:
                writer=csv.writer(f)
                writer.writerow(data_vec)
    else:
        print(f"Student training data already generated \n")
    return

def train_student_offline(teacher,tag, alpha=alpha, test_train_split=0.9, epochs=500, batch_size=16):
    ctag = f"train_student_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    fname = f"./data/{ctag}.csv"
    student_model_fname = f"./models/student_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    if((not os.path.isdir(f"{student_model_fname}/")) or True):
        data = np.genfromtxt(fname, delimiter=',')
        np.random.shuffle(data)
        split = 0.95
        num_samples = data.shape[0]
        split_sample = int(split*num_samples)
        print(data.shape)
        X = data[:split_sample, :-1]
        y = data[:split_sample, -1]
        X_test = data[split_sample:, :-1]
        y_test = data[split_sample:, -1]
        y_test = np.reshape(y_test, (num_samples-split_sample, 1))
        
        student = Sequential()
        student.add(Dense(student_len, input_shape=(student_len, ), activation='relu'))
        student.add(Dense(int(student_len*(0.5)), activation='sigmoid'))
        student.add(Dense(int(student_len*(0.5)), activation='sigmoid'))
        student.add(Dense(1, activation='linear'))
        student.compile(loss='mean_squared_error', optimizer='adam')
        
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
        
        history = distiller.fit(X,y, validation_data=(X_test, y_test), epochs = epochs, batch_size = 50, shuffle=True, verbose=1)
        student = distiller.student
        student.save(student_model_fname)
        plt.figure(10)
        plt.plot(history.history['student_loss'], alpha=0.6)
        plt.plot(history.history['val_student_loss'], alpha=0.8)
        plt.grid()
        # plt.ylim(0,1e-4)
        plt.yscale('log')
        plt.title('Homogeneous Network : Student Loss')
        plt.ylabel('Mean Squared Error (normalised)')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig("./plots/train_homo_test_student.png")
        return distiller.student
    else:
        print(f"Trained student model exists \n")
        student = load_model(student_model_fname)
        return student

# Reform start

def evaluate_student_run_sim(student, tag, num_iters = num_iters, l = l,  n_max = n_max,n_min = n_min,  jumps = jumps, q=q, alpha=0.1, seed=100):
    steps = get_steps(num_iters, l, n_max, n_min, jumps, q, seed)
    perf = np.zeros((num_iters, 4))
    ctag = f"perf_student_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    fname = f"./data/{ctag}.csv"
    if(os.path.isfile(fname)):
        os.remove(fname)
    for i in range(num_iters):
        nodes = steps[i]
        # SRCs with length l
        srcs_estimates = srcs(nodes, l=srcs_l, num_lof=num_lof)/n_max
        if(not i):
            estimates = np.random.randint(n_min, n_max, 1)
            bb_aware_estimates = estimates
            prev_truths = np.random.randint(n_min, n_max, 1)
        bb_aware_estimates = bb_aware(nodes, bb_aware_estimates, l=l)/n_max
        fv_student, fv_teacher = gen_feature_vectors_for_slot(n_max, l, nodes, estimates, prev_truths)
        predict_input = fv_student[:-1]
        predict_input = np.reshape(predict_input, (1, predict_input.shape[0]))
        target = fv_student[-1]
        prediction = student.predict(predict_input, verbose = -1)
        err = ((target - prediction[0][0])**2)
        srcs_err = ((target - srcs_estimates)**2)
        bb_aware_err = ((target - bb_aware_estimates)**2)
        estimates = prediction[0][0]*n_max
        prev_truths = target*n_max
        print(f"i={i}/{num_iters}, est={estimates:.3f}, truth={prev_truths:.3f}, err={err:.3e}, srcs_err={srcs_err:.3e}, bb_aware_err={bb_aware_err:.3e}", end="\r")
        perf[i,0] = err
        perf[i,1] = srcs_err
        perf[i,2] = bb_aware_err
        perf[i, 3] = target
        perf_row = perf[i, :]
        with open(fname, 'a') as f:
            writer=csv.writer(f)
            writer.writerow(perf_row)
    return perf   

def plot_perf(perf, tag):
    nn_mean = np.mean(perf[:,0])
    srcs_mean = np.mean(perf[:,1])
    bb_aware_mean = np.mean(perf[:,2])
    plt.plot(perf[:, 0], '-b',label = "NN")
    plt.plot(perf[:, 1], '--r',label = "SRCs", alpha = 0.7)
    plt.plot(perf[:, 2], '-.c',label = "BB-Aware", alpha = 0.6)
    plt.title(f"Performance (MSE) of Different Algorithms in a Homogeneous Network\n Avg NN MSE = {np.mean(perf[:,0]):.3e} \n Avg SRCs MSE = {np.mean(perf[:,1]):.3e}, Avg BB-Aware MSE = {np.mean(perf[:,2]):.3e}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mean Squared Error (normalised)")
    plt.grid()
    plt.legend()
    plt.yscale('log')
    plt.savefig(f"./plots/{tag}_l{int(l)}_j{jumps}_n{num_iters}.png")
    plt.close()
    return np.array([nn_mean, srcs_mean, bb_aware_mean]).reshape((1,3))

# Reform end

if __name__=="__main__":
    # fv_student, fv_teacher = gen_feature_vectors_for_slot(100)
    # ctag = f"train_teacher_{tag}_l{int(l)}_j{jumps}_n{num_iters}"
    # fname = f"./data/{ctag}.csv"
    # if(os.path.isfile(fname)):
    #     os.remove(fname)
    gen_training_data_teacher_run_sim(tag, seed=90)
    teacher = train_teacher_offline(tag = tag, epochs = 500)

    gen_training_data_student_run_sim(teacher, tag=tag, num_iters=num_iters, seed=38)
    student = train_student_offline(teacher, tag=tag)
    # student = load_model("./models/student_homo_jumps_l100_j10_n10000")
    eval_arr = np.zeros((1, 3))
    for i in range(num_eval_runs):
        print(f"Run {i+1}/{num_eval_runs} \n")
        perf = evaluate_student_run_sim(student,tag = tag, num_iters=num_eval_iters, seed=i+5)
        res = plot_perf(perf, tag=f"{tag}_s{i}")
        eval_arr = np.append(eval_arr, res, axis=0)
    eval_df = pd.DataFrame(eval_arr, columns=["NN_mean", "SRCs_mean", "BB_aware_mean"])
    eval_df.drop(0, inplace=True)
    eval_df.to_csv(f"./data/eval_student_{tag}_l{int(l)}_j{jumps}_n{num_iters}.csv", index=False)