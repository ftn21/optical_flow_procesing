import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import MinMaxScaler

import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import datasets, models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision


# classes, functions
class Net(nn.Module):
    
  def __init__(self):

    super(Net, self).__init__()

    self.fc1 = nn.Linear(50, 20)
    self.fc2 = nn.Linear(20, 8)
    self.fc3 = nn.Linear(8, 8)
    self.fc4 = nn.Linear(8, 4)
    self.fc5 = nn.Linear(4, 4)
    self.fc6 = nn.Linear(4, 2)

  def forward(self, x):

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = self.fc6(x)
    
    return x
  
def str2list(strlist):
  ''' преобразование строки в список '''
  
  strlist = strlist.replace('[', '')
  strlist = strlist.replace(']', '')
  strlist = strlist.replace(',', '')
  strlist = strlist.replace("'", '')
  strlist = strlist.replace("\\n", '')
  result = strlist.split(' ')

  return result

def get_floats(str_):
  ''' проебразование строки (так массив хранится в pd.dataframe)
      в numpy array типа float'''
  
  str_list = str2list(str_)
  float_list = [float(a) for a in str_list]

  return np.asarray(float_list)

class flow_dataset(Dataset):
  """ dataset 
  x -- 25 точек по 2 координаты (50)
  y -- смещение по x и y (2) """

  def __init__(self, df_pathname, tensor = True, dataset_type='full', train_part=0.8, norm=True, half_precision=True):
    ''' df_pathname -- путь к datafram'у
        tensor -- в каком формате возвращать сэмпл (в тензоре, если True)
        dataset_type -- тип датасета (train, test, full)
        train_part -- часть датасета для train выборки
        norm -- нормализация данных (True/False)
        half_precision -- рассчет в float16 (True/False)'''

    self.tensor = tensor
    self.half_precision = half_precision
    full_frame = pd.read_csv(df_pathname)
    train_size = int(train_part*len(full_frame))
    test_size = len(full_frame)-train_size

    if dataset_type=='full':
      frame = full_frame
    elif dataset_type=='train':
      frame = full_frame.head(train_size)
    elif dataset_type=='test':
      frame = full_frame.tail(test_size)

    if norm:
      scaler = MinMaxScaler()
      frame.iloc[:, 1:51] = scaler.fit_transform(frame.iloc[:, 1:51])
    
    self.np_data = frame.to_numpy()[:, 1:]
    print(f'data shape: {self.np_data.shape}')

  def __len__(self):
    return self.np_data.shape[0]

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    x = self.np_data[idx, 0:50]
    y = self.np_data[idx, 50:52]  # только смещение
    
    if self.tensor:
      if self.half_precision:
        x = torch.as_tensor(x, dtype=torch.float16)
        y = torch.as_tensor(y, dtype=torch.float16)
      else:
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

    return {'x': x, 'y': y}
  
def get_dataloaders(dataframe_pathname, seed_worker, generator, shuffle=True, num_workers=1):
  train_dataset = flow_dataset(dataframe_pathname, dataset_type='train')
  test_dataset = flow_dataset(dataframe_pathname, dataset_type='test')
  test_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)
  train_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, worker_init_fn=seed_worker, generator=generator)

  return train_loader, test_loader

def plot_history(train_history, val_history, title="loss"):
  plt.figure()
  plt.title('{}'.format(title))
  plt.plot(train_history, c='c', label="train", zorder=1)
  
  points = np.array(val_history)
  steps = list(range(0, len(train_history) + 1, int(len(train_history) / len(val_history))))[1:]
  
  plt.plot(steps, val_history, c="orange", label="test", zorder=2)
  plt.xlabel("train steps")
  
  plt.legend(loc="best")
  plt.grid(alpha=0.25)
  
  # plt.savefig(os.path.join(directory, 'AE1_mse_sigm'+str(len(train_history))+str(len(val_history))+'.png'),
  #                 format='png', dpi=200, bbox_inches='tight')

  plt.show()

def train(model, train_loader, test_loader, criterion, device, weights_pn, writer, epochs=100, batch_size=8):
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.001,
                            momentum=0.9, weight_decay=0.0005)
  train_loss_log, test_loss_log = [], []
  for epoch in range(epochs):
    # train
    train_epoch_loss = (torch.empty(0)).to(device)
    model.train()
    for data_num, data in enumerate(train_loader):
      X = data['x'].to(device)
      y = data['y'].to(device)
      
      y_pred = model(X)
      loss = criterion(y_pred, y) 
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_epoch_loss = torch.cat((train_epoch_loss, loss.unsqueeze(0) / y.size(0)))
      train_loss_log.append(loss.item() / y.size(0))
    print('train', epoch)
    
    # test
    test_epoch_loss = (torch.empty(0)).to(device)
    model.eval()
    with torch.no_grad():
      for data_num, data in enumerate(test_loader):
        X = data['x'].to(device)
        y = data['y'].to(device)
        
        y_pred = model(X)
        loss = criterion(y_pred, y) 
        test_epoch_loss = torch.cat((test_epoch_loss, loss.unsqueeze(0) / y.size(0)))
        
      test_loss_log.append(test_epoch_loss.mean().item() / y.size(0))
      # plot_history(train_loss_log, test_loss_log, "loss")
      loss_train = train_epoch_loss.mean().item()
      loss_test = test_epoch_loss.mean().item()
      writer.add_scalars(f'loss', {
          'train': loss_train,
          'test': loss_test,
      }, epoch)
      print(f"epoch: {epoch}, train loss: {train_epoch_loss.mean().item()}, test: {test_epoch_loss.mean().item()}")
  
    writer.flush()
    
    # save weights
    if ((epoch % 10 == 0) and (epoch > 9)):
      w_pathname = weights_pn+str(epoch)+".torch"
      torch.save(model.state_dict(), w_pathname)

def inference(model, device, loader):
  
  model.eval()
  ys = []
  predicts = []
  with torch.no_grad():
    for data_num, data in enumerate(loader):
      X = data['x'].to(device)
      y = data['y'].to(device)
      y_pred = model(X)
      ys.append(y.cpu().numpy()[0])
      predicts.append(y_pred.cpu().numpy()[0])
      print(data_num)
  
  return ys, predicts
    

def set_seed(seed, generator):
  ''' фиксирование сида PyTorch '''

  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.cuda.manual_seed(seed)
  generator.manual_seed(seed)

  return generator
  
def seed_worker(worker_id):
  ''' seed_worker функция для фиксирования сида worker'ов dataloader'е '''

  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  # random.seed(worker_seed)

def seed_all(max_seed = 99999, seed=None):
  ''' фиксирование сидов '''

  if seed is None:
    seed = np.random.randint(0, max_seed)
  generator = torch.Generator()
  generator = set_seed(seed, generator)

  return generator

def init_logging(project, log_name):
  ''' логгирование '''

  logger = logging.getLogger(log_name)
  logger.setLevel(logging.DEBUG)
  handler = logging.FileHandler('log/{}.log'.format(project), 'w', 'utf-8')
  handler.setFormatter(logging.Formatter('%(name)s %(message)s'))
  logger.addHandler(handler)
  logger.debug("start time {0}".format(datetime.now()))

  return logger