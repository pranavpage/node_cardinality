tag = "homo_alpha1"
num_iters = 10000 
l = 100
n_max = int((2**6))
n_min = int(2**3)
jumps = 5
q = 0.2
student_len = 3*l + 1
teacher_len = l + 1 # trial results + prev truth  
num_lof = 3
ID_bits = 8
num_eval_runs = 25
num_eval_iters = 1000
alpha=0.1
srcs_l = (l - num_lof*ID_bits)