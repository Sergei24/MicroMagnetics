#compute llg right side
def llg_rhs (m, h_eff, gamma, alpha):
  h = h_eff(m)
  llg_rhs = - gamma/(1+alpha**2) * np.cross(m, h) - alpha*gamma/(1+alpha**2) * np.cross(m, np.cross(m, h))
  return llg_rhs


# compute llg step using Euler method
def llg_eu(m, dt, gamma, alpha):
  dmdt = llg_rhs (m, h_eff, gamma, alpha)
  m += dt * dmdt
  return  m/np.repeat(np.sqrt((m*m).sum(axis=3)), 3).reshape(m.shape)
  
# compute llg step using RK4
def llg_rk4(m, dt, gamma, alpha):
  k1 = llg_rhs (m, h_eff, gamma, alpha)
  k2 = llg_rhs (m+(dt/2.0)*k1, h_eff, gamma, alpha)
  k3 = llg_rhs (m+(dt/2.0)*k2, h_eff, gamma, alpha)
  k4 = llg_rhs (m+dt*k3, h_eff, gamma, alpha)
  m += (dt/6.0)*(k1+2*k2+2*k3+k4)
  return  m/np.repeat(np.sqrt((m*m).sum(axis=3)), 3).reshape(m.shape)
