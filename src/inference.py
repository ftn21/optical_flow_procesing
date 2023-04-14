from model_back import *


def get_traj_on_offsets(dxdys, p0):
  ''' восстановить траекторию по начальной точке и смещениям '''
  
  trajectory = []
  cur_point = p0.copy()
  for dxdy in dxdys:
    cur_point += dxdy
    trajectory.append(cur_point.copy())

  return np.asarray(trajectory)


# source code
if __name__ == '__main__':
 
  # const
  weights_pathname = '/Users/ftn/Documents/другое/диплом Саша/weights_h/net1000_sch299.torch'
  dataframe_pathname = './data/round_traj.csv'
  num_workers = 8

  # device
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  print(device)

  model = Net()
  model.to(device)
  checkpoint = torch.load(weights_pathname, map_location=device)
  model.load_state_dict(checkpoint)
  model.eval()

  ineference_dataset = flow_dataset(dataframe_pathname, dataset_type='full', norm=False, half_precision=False)
  ineference_loader = DataLoader(ineference_dataset, num_workers=num_workers)

  ys, preds = inference(model, device, ineference_loader)

  print('inference done')

  p_start = np.asarray([0., 0.])

  true_traj = get_traj_on_offsets(ys, p_start)
  pred_traj = get_traj_on_offsets(preds, p_start)

  plt.scatter(true_traj[:, 0], true_traj[:, 1], alpha=0.7, c='k')
  plt.plot(true_traj[:, 0], true_traj[:, 1], c='k', alpha=0.7, label='истинная траектория')
  plt.scatter(pred_traj[:, 0], pred_traj[:, 1], alpha=0.7 , c='b')
  plt.plot(pred_traj[:, 0], pred_traj[:, 1], c='b', alpha=0.7,  label='предсказание сети')
  plt.grid()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('траектория бпла')
  plt.legend()
  plt.axis('equal')
  plt.show()