import numpy as np
from scipy.spatial.transform import Rotation as R

t_max = 0.05
step_UAV = 100
step = 0.01
t = np.arange(0, t_max, step)

x = np.array([0.4*t, 0.4*np.sin(np.pi*t), 0.6*np.cos(np.pi*t)])
p0 = np.array([np.cos(0.2*np.pi*t), np.sin(0.2*np.pi*t), np.zeros(np.max(np.shape(t),))]) + x

angles = np.arccos(p0/np.sqrt(np.sum(p0**2, axis= 0)))
