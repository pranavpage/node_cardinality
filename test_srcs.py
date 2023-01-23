import numpy as np
import pydtmc
import matplotlib.pyplot as plt 
p = [[0.8, 0.2], [0.3, 0.7]]
mc = pydtmc.MarkovChain(p, ['A', 'B'])
print(mc)
