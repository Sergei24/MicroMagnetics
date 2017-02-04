import numpy as np
import H_ef
import LLG

# setup mesh and material constants
n     = (10, 10, 1)
dx    = (5e-9, 5e-9, 3e-9)
gamma = 2.211e5
ms    = 8e5
A     = 1.3e-11
alpha = 0.02

# initialize magnetization that relaxes into s-state
m = np.zeros(n + (3,))
m[1:-1,:,:,0]   = 1.0
m[(-1,0),:,:,1] = 1.0
  
# relax
alpha = 1.00
for i in range(5000):
    LLG.llg_rk4(m, 2e-13, n, dx, gamma, alpha, ms)
    print i