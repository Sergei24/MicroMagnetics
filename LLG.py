
# compute llg step using Euler method
def llg_eu(m, h_eff, dt, gamma, alpha):
  h = h_eff(m)
  dmdt = - gamma/(1+alpha**2) * np.cross(m, h) - alpha*gamma/(1+alpha**2) * np.cross(m, np.cross(m, h))
  m += dt * dmdt
  return  m/np.repeat(np.sqrt((m*m).sum(axis=3)), 3).reshape(m.shape)
  
# compute llg step using RK4
def llg_rk4(m, h_eff, dt, gamma, alpha):
  h = h_eff(m)
  
