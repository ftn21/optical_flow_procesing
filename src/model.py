from model_back import *

# source code
if __name__ == '__main__':

  # const
  data_pathname = './data/data1_part.csv'
  weights_pathname = './data/weights/net1_'
  num_workers = 2  # количество потоков

  # set tensorboard writer
  writer = SummaryWriter() 

  # device
  if torch.backends.mps.is_available():
    device = torch.device("mps")
  elif torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  print(device)

  # set seed
  generator = seed_all()

  # train
  model = Net()
  model.to(device)
  train_loader, test_loader = get_dataloaders(data_pathname, seed_worker, generator, shuffle=True, num_workers=num_workers)
  criterion = nn.MSELoss()
  train(model, train_loader, test_loader, criterion, device, weights_pathname, writer, epochs=100, batch_size=256)