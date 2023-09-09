import numpy as np

sims = [0,1,2,3,4,5]

argsorted_sims = np.argsort(sims)[::-1] #[length * 2]

print(argsorted_sims)