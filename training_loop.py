import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MultilabelF1Score
from evaluation import evaluate
from training_callbacks import PPIEarlyStopping, ModelSaver
from dataset import num_labels, train_loader, val_loader, test_loader
from device import device


def print_model_metrics(mode, loss, score):
  print('\t{}_loss: {:.4f} | {}_micro_f1: {:.4f}'.format(mode, loss, mode, score))


def train(model, params: dict, verbose: bool = True) -> torch.nn.Module:
  print('training model {}'.format(params['model_name']))
  print(model)
  print('training...')

  optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
  model.reset_parameters()

  def apply_batch(train_batch):
    model.train()

    node_features, edge_index, labels = (train_batch.x.to(device), train_batch.edge_index.to(device),
                                         train_batch.y.to(device))

    batch_loss = nn.BCEWithLogitsLoss()
    batch_score = MultilabelF1Score(num_labels, average='micro').to(device)

    logits = model(node_features, edge_index)
    pred = torch.where(logits >= 0, 1, 0)

    # perform training step

    optimizer.zero_grad()
    loss = batch_loss(logits, labels)
    loss.backward()
    optimizer.step()

    return (loss.item(), batch_score(pred, labels))

  earlyStopping = PPIEarlyStopping(params['patience'])
  model_saver = ModelSaver(params['model_name'])

  train_losses, train_scores = [], []
  val_losses, val_scores = [], []

  for epoch in range(params['epochs']):
    if (earlyStopping.should_stop()):
      print('early stopping...')
      break

    total_train_loss = 0
    total_train_score = 0
    for (batch_ix, batch) in enumerate(train_loader):
      batch_loss, batch_score = apply_batch(batch)
      total_train_loss += batch_loss
      total_train_score += batch_score

    avg_train_loss = total_train_loss / (batch_ix + 1)
    avg_train_score = total_train_score / (batch_ix + 1)

    val_loss, val_score = evaluate(model, val_loader)

    train_losses.append(avg_train_loss)
    train_scores.append(avg_train_loss)
    val_losses.append(val_loss)
    val_scores.append(val_score)

    was_best_so_far = earlyStopping.register(val_score, val_loss)
    if was_best_so_far:
      model_saver.save_best_model(epoch, model, optimizer)

    if verbose:
      print('epoch {:05d}'.format(epoch))
      print_model_metrics('train', avg_train_loss, avg_train_score)
      print_model_metrics('val', val_loss, val_score)

  best_epoch, best_model = model_saver.load_best_model(model)
  best_val_loss, best_val_score = evaluate(model, val_loader)
  best_test_loss, best_test_score = evaluate(model, test_loader)

  print('best model performance @ epoch {:05d}: '.format(best_epoch))
  print_model_metrics('val', best_val_loss, best_val_score)
  print_model_metrics('test', best_test_loss, best_test_score)

  return best_model, train_losses, train_scores, val_losses, val_scores, model_saver