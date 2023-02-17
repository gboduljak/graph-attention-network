import torch


class PPIEarlyStopping():

  def __init__(self, patience: int):
    self.best_f1 = -1
    self.best_loss = pow(10, 9)
    self.patience = patience
    self.underperformed = 0

  def register(self, f1, loss):
    if self.should_stop():
      return False

    if (f1 > self.best_f1 or loss < self.best_loss):
      self.best_f1 = max(self.best_f1, f1)
      self.best_loss = min(self.best_loss, loss)
      self.underperformed = 0
      return self.best_f1 == f1 and self.best_loss == loss
    else:
      self.underperformed += 1
      return False

  def should_stop(self):
    return self.underperformed >= self.patience


class ModelSaver():

  def __init__(self, path: str):
    self.path = path

  def save_best_model(self, epoch: int, model, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, self.path)

  def load_best_model(self, model):
    state = torch.load(self.path)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return state['epoch'], model

  def download_best_model_state(self):
    from google.colab import files
    files.download(self.path)