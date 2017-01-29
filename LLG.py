from H_ef import h_eff

#compute llg right side
def llg_rhs (m, n, dx, h_eff, gamma, alpha, ms):
  h = h_eff(m, n, dx, ms)
  llg_rhs = - gamma/(1+alpha**2) * np.cross(m, h) - alpha*gamma/(1+alpha**2) * np.cross(m, np.cross(m, h))
  return llg_rhs


# compute llg step using Euler method
def llg_eu(m, dt, n, dx, gamma, alpha, ms):
  dmdt = llg_rhs (m, n, dx, h_eff, gamma, alpha, ms)
  m += dt * dmdt
  return  m/np.repeat(np.sqrt((m*m).sum(axis=3)), 3).reshape(m.shape)
  
# compute llg step using RK4
def llg_rk4(m, dt, n, dx, gamma, alpha, ms):
  k1 = llg_rhs (m, n, dx, h_eff, gamma, alpha, ms)
  k2 = llg_rhs (m+(dt/2.0)*k1, n, dx, h_eff, gamma, alpha, ms)
  k3 = llg_rhs (m+(dt/2.0)*k2, n, dx, h_eff, gamma, alpha, ms)
  k4 = llg_rhs (m+dt*k3, n, dx, h_eff, gamma, alpha, ms)
  m += (dt/6.0)*(k1+2*k2+2*k3+k4)
  return  m/np.repeat(np.sqrt((m*m).sum(axis=3)), 3).reshape(m.shape)
