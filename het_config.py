import numpy as np

num_iters = 25000
l = 150
T = 7
# n_max = int((2**6)*3/T)
n_max = int((2**6)*3)
total_n_max = (2**6)*3
n_min = 2**1
# srcs_l = int(l/T)
jumps = 5
norm_jumps = int(5*3/T)
q=0.2
student_len = (T//2)*l*4 + T 
teacher_len = (T)*l + T 
num_lof = 3
ID_bits = 8
num_eval_runs = 20
num_eval_iters = 2500
alpha=0.1
if(T%2 == 0):
    srcs_l = int((T/2)*l*1.0/T - num_lof*ID_bits)
    print(srcs_l)
else:
    srcs_l = int((np.floor(T-1)/2)*l*1.0/T - num_lof*ID_bits)
    print(srcs_l)


	