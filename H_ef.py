import numpy as np
from math import asinh, atan, sqrt, pi

# a very small number
eps = 1e-18

mu0 = mu0   = 4e-7 * pi

# newell f
def f(p):
  x, y, z = abs(p[0]), abs(p[1]), abs(p[2])
  return + y / 2.0 * (z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps)) \
         + z / 2.0 * (y**2 - x**2) * asinh(z / (sqrt(x**2 + y**2) + eps)) \
         - x*y*z * atan(y*z / (x * sqrt(x**2 + y**2 + z**2) + eps))       \
         + 1.0 / 6.0 * (2*x**2 - y**2 - z**2) * sqrt(x**2 + y**2 + z**2)

# newell g
def g(p):
  x, y, z = p[0], p[1], abs(p[2])
  return + x*y*z * asinh(z / (sqrt(x**2 + y**2) + eps))                         \
         + y / 6.0 * (3.0 * z**2 - y**2) * asinh(x / (sqrt(y**2 + z**2) + eps)) \
         + x / 6.0 * (3.0 * z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps)) \
         - z**3 / 6.0 * atan(x*y / (z * sqrt(x**2 + y**2 + z**2) + eps))        \
         - z * y**2 / 2.0 * atan(x*z / (y * sqrt(x**2 + y**2 + z**2) + eps))    \
         - z * x**2 / 2.0 * atan(y*z / (x * sqrt(x**2 + y**2 + z**2) + eps))    \
         - x*y * sqrt(x**2 + y**2 + z**2) / 3.0

# setup demag tensor
def demag_tensor (n, dx):
    
  def set_n_demag(c, permute, func):
    it = np.nditer(n_demag[:,:,:,c], flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
      value = 0.0
      for i in np.rollaxis(np.indices((2,)*6), 0, 7).reshape(64, 6):
        idx = map(lambda k: (it.multi_index[k] + n[k]) % (2*n[k]) - n[k], range(3))
        value += (-1)**sum(i) * func(map(lambda j: (idx[j] + i[j] - i[j+3]) * dx[j], permute))
    it[0] = - value / (4 * pi * np.prod(dx))
    it.iternext()

  n_demag = np.zeros([1 if i==1 else 2*i for i in n] + [6])
    for i, t in enumerate(((f,0,1,2),(g,0,1,2),(g,0,2,1),(f,1,2,0),(g,1,2,0),(f,2,0,1))):
    set_n_demag(i, t[1:], t[0])

  m_pad     = np.zeros([1 if i==1 else 2*i for i in n] + [3])
  f_n_demag = np.fft.rfftn(n_demag, axes = filter(lambda i: n[i] > 1, range(3)))
  
  return (m_pad, f_n_demag)

        
        
# demag field
def h_demag(m, N): 
  
  m_pad = N[0]
  f_n_demag = N[1]
    
  m_pad[:n[0],:n[1],:n[2],:] = m
  f_m_pad = np.fft.rfftn(m_pad, axes = filter(lambda i: n[i] > 1, range(3)))
  f_h_demag_pad = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
  f_h_demag_pad[:,:,:,0] = (f_n_demag[:,:,:,(0, 1, 2)]*f_m_pad).sum(axis = 3)
  f_h_demag_pad[:,:,:,1] = (f_n_demag[:,:,:,(1, 3, 4)]*f_m_pad).sum(axis = 3)
  f_h_demag_pad[:,:,:,2] = (f_n_demag[:,:,:,(2, 4, 5)]*f_m_pad).sum(axis = 3)
  h_demag = np.fft.irfftn(f_h_demag_pad, axes = filter(lambda i: n[i] > 1, range(3)))[:n[0],:n[1],:n[2],:]
  print 'demag field computed'
  return h_demag

# exchange field
def h_ex(m, n, dx):  
  h_ex = - 2 * m * sum([1/x**2 for x in dx])
  for i in range(6):
    h_ex += np.repeat(m, 1 if n[i%3] == 1 else [i/3*2] + [1]*(n[i%3]-2) + [2-i/3*2], axis = i%3) / dx[i%3]**2
  print 'exchange field computed'
  return h_ex

# effective field
def h_eff(m, n, N, dx, ms):
    return ms*h_demag(m, N) + 2*A/(mu0*ms)*h_ex(m, n)
