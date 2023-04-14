import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos
from numpy.linalg import inv
import pandas as pd


def flight_model_lin(time = 1):
  ''' линейная модель полета бпла как материальной точки. '''

  coords = []

  # координаты бпла в начальный момент времени
  x0 = 3.
  y0 = 3.
  q = 0.5  # угол курса 
  # в текущий момент времени
  x = x0
  y = y0
  v = 20.
  dt = 0.5  # шаг по времени
  t = 0.  # текущее время
  coords.append([x, y])

  while t<time:
    v_x = v*np.cos(q)
    v_y = v*np.sin(q)
    x = x + v_x*dt
    y = y + v_y*dt
    # q += 0.5  # поворачивает
    t += dt
    coords.append([x, y])
  
  return np.asarray(coords)

def flight_model_parab(time = 1):
  ''' линейная модель полета бпла как материальной точки. '''

  coords = []

  # координаты бпла в начальный момент времени
  x0 = 0.
  y0 = 0.
  q = 0.5  # угол курса 
  # в текущий момент времени
  x = x0
  y = y0
  v = 10.
  dt = 0.5  # шаг по времени
  t = 0.  # текущее время
  coords.append([x, y])

  while t<time:
    v_x = v*np.cos(q)
    v_y = v*np.sin(q)
    x = x + v_x*dt
    # y = y + v_y*dt
    y = 0.02*x*x
    # q += 0.5  # поворачивает
    t += dt
    coords.append([x, y])
  
  return np.asarray(coords)

def flight_model_round(time = 1):
  ''' модель полета бпла как материальной точки по окружности. '''

  coords = []

  # координаты бпла в начальный момент времени
  x0 = 0.
  y0 = 0.
  q = 0.5  # угол курса 
  # в текущий момент времени
  x = x0
  y = y0
  v = 20.
  dt = 0.5  # шаг по времени
  t = 0.  # текущее время
  r = 5.
  coords.append([x, y])

  while t<time:
    v_x = r*v*np.cos(q)
    v_y = r*v*np.sin(q)
    x = x + v_x*dt
    y = y + v_y*dt
    q += 0.5  # поворачивает
    t += dt
    coords.append([x, y])
  
  return np.asarray(coords)

def flight_model_elllips(time = 1):
  ''' модель полета бпла как материальной точки по эллиптическое траектории. '''

  coords = []

  # координаты бпла в начальный момент времени
  x0 = 0.
  y0 = 0.
  q = 0.5  # угол курса 
  # в текущий момент времени
  x = x0
  y = y0
  v = 20.
  dt = 0.5  # шаг по времени
  t = 0.  # текущее время
  a = 8.
  b = 5.
  coords.append([x, y])

  while t<time:
    v_x = a*v*np.cos(q)
    v_y = b*v*np.sin(q)
    x = x + v_x*dt
    y = y + v_y*dt
    q += 0.5  # поворачивает
    t += dt
    coords.append([x, y])
  
  return np.asarray(coords)

# смещения
def get_dxdy(xy):
  ''' получаем из перемещения бпла векторы перемещения '''

  dx_dy = []
  x = xy[:, 0]
  y = xy[:, 1]
  for i in range(xy.shape[0]-1):
    dx = x[i+1]-x[i]
    dy = y[i+1]-y[i]
    dx_dy.append([dx, dy])

  return dx_dy

def get_traj_on_offsets(dxdys, p0):
  ''' восстановить траекторию по начальной точке и смещениям '''

  trajectory = []
  cur_point = p0
  for dxdy in dxdys:
    cur_point += dxdy
    trajectory.append(cur_point.copy())

  return np.asarray(trajectory)


def generate_uv(dx, dy, H, ang_1, ang_2, M):
  ''' функция моделирования оптического потока (1 итерация)
  
      dx, dy -- смещение 
      ang_1, ang_2 -- ориентации 1 и 2 кадров
      M -- калибровочная матрица '''

  # смещение
  offset = np.array([dx, dy])

  # матрицы поворота первый кадр
  MX1 = np.array([[1, 0, 0], [0, cos(ang_1[0]), -sin(ang_1[0])], [0, sin(ang_1[0]), cos(ang_1[0])]])
  MY1 = np.array([[cos(ang_1[1]), 0, sin(ang_1[1])], [0, 1, 0], [-sin(ang_1[1]), 0, cos(ang_1[1])]])
  MZ1 = np.array([[cos(ang_1[2]), -sin(ang_1[2]), 0], [sin(ang_1[2]), cos(ang_1[2]), 0], [0, 0, 1]])
  XY1 = np.matmul(MX1, MY1)
  Rin = np.matmul(XY1, MZ1)

  # матрицы поворота второй кадр
  MX2 = np.array([[1, 0, 0], [0, cos(ang_2[0]), -sin(ang_2[0])], [0, sin(ang_2[0]), cos(ang_2[0])]])
  MY2 = np.array([[cos(ang_2[1]), 0, sin(ang_2[1])], [0, 1, 0], [-sin(ang_2[1]), 0, cos(ang_2[1])]])
  MZ2 = np.array([[cos(ang_2[2]), -sin(ang_2[2]), 0], [sin(ang_2[2]), cos(ang_2[2]), 0], [0, 0, 1]])
  XY2 = np.matmul(MX2, MY2)
  Rout = np.matmul(XY2, MZ2)

  MRinINV = inv(np.matmul(M, Rin))
  # MRoutINV = inv(np.matmul(M, Rout))

  RinRout = np.matmul(Rin, Rout)

  ExpRinRout = np.zeros((4, 4))
  ExpRinRout[:3, :3] = RinRout
  ExpRinRout[:2, 3] = offset

  # точки первого изображения
  U1V1 = np.ones((3, 25))
  x = (M[0][2] + M[1][2]) / 6
  x_ = 0
  for i in range(len(U1V1[0])):
      if i % 5 == 0:
          x_ += x
      U1V1[0][i] = np.round(x_)
  x_ = 0
  for i in range(len(U1V1[0])):
      x_ += x
      U1V1[1][i] = np.round(x_)
      if i % 5 == 4:
          x_ = 0

  XYZ = np.matmul(MRinINV, U1V1)

  ExpXYZ = np.ones((4, len(U1V1[0])))
  ExpXYZ[:3, :] = XYZ

  XYZ_H = ExpXYZ[:]
  XYZ_H[:3, :] *= H

  XYZ_new = np.matmul(ExpRinRout, XYZ_H)
  XYZ_H_new = np.zeros((3, len(U1V1[0])))
  XYZ_H_new = XYZ_new[:3, :] / H

  U2V2 = np.matmul(M, XYZ_H_new)
  U2V2 = np.array(list(map(lambda x: np.round(x), U2V2)))

  a = np.matmul(RinRout, XYZ)

  y = np.zeros((2, len(U1V1[0])))
  for i in range(len(U1V1[0])):
      y[0][i] = U2V2[0][i] - M[0][0] * a[0][i] - M[0][2] * a[2][i]
      y[1][i] = U2V2[1][i] - M[1][1] * a[1][i] - M[1][2] * a[2][i]

  t = np.zeros((2, len(U1V1[0])))
  for i in range(len(U1V1[0])):
      t[0][i] = H * y[0][i] / M[0][0]
      t[1][i] = H * y[1][i] / M[1][1]

  temp = []
  for i in range(len(U1V1[0])):
      temp.append(U2V2[0][i] - U1V1[0][i])
      temp.append(U2V2[1][i] - U1V1[1][i])
  
  t = list(offset) + list(RinRout.flatten())

  return temp, t


def generate_uv_dataset(dxdys, h):
  ''' функция генерации выборки с учетом траектории '''

  data_x = list()
  data_y = list()
  
  degree = np.pi / 180

  # задаем случайные ориентации для 1 и 2 кадров
  ang_1_rand = np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)])
  ang_2_rand = np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(-10, 10)])

  angles1 = np.array([ang_1_rand[0] * degree, ang_1_rand[1] * degree, ang_1_rand[2] * degree])
  angles2 = np.array([ang_2_rand[0] * degree, ang_2_rand[1] * degree, ang_2_rand[2] * degree])

  # калибровочная матрица
  mtx_calibrate = np.array([[300, 0, 200], [0, 300, 200], [0, 0, 1]])

  for dxdy in dxdys:
    dx = dxdy[0]
    dy = dxdy[1]
    x, y = generate_uv(dx, dy, h, angles1, angles2, mtx_calibrate)
    data_x.append(x)
    data_y.append(y)

  return data_x, data_y

def write_to_dataframe(data_x, data_y, data_pathname):
  ''' запись в dataframe (csv) '''

  data_x = np.asarray(data_x)
  data_y = np.asarray(data_y)

  data_dict = {}

  for i in range(data_x.shape[1]):
      data_dict['x'+str(i)] = data_x[:, i]

  for i in range(data_y.shape[1]):
      data_dict['y'+str(i)] = data_y[:, i]

  data = pd.DataFrame(data=data_dict)
  data.to_csv(data_pathname)


dataframe_pathname = './data/round_traj.csv'
h = 1000

# xy = flight_model_round(6)
xy = flight_model_elllips(6)
dxdys = get_dxdy(xy)
# xy2 = get_traj_on_offsets(dxdys, xy[0])

data_x, data_y = generate_uv_dataset(dxdys, h)
write_to_dataframe(data_x, data_y, dataframe_pathname)

plt.scatter(xy[:, 0], xy[:, 1], c='k')
plt.plot(xy[:, 0], xy[:, 1], c='k')
# plt.scatter(xy2[:, 0], xy2[:, 1], c='b')
# plt.plot(xy2[:, 0], xy2[:, 1], c='b')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('траектория бпла')
plt.axis('equal')
plt.show()