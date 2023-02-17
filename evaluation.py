import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelF1Score
from dataset import num_labels
from device import device


def evaluate(model, batch_loader):
  model = model.to(device)
  model.eval()

  def evaluate_batch(node_features, edge_index, labels):
    f1_score = MultilabelF1Score(num_labels, average='micro').to(device)
    loss_fcn = nn.BCEWithLogitsLoss()
    model.eval()

    with torch.no_grad():
      logits = model(node_features, edge_index)
      pred = torch.where(logits >= 0, 1, 0)
      return (loss_fcn(logits, labels), f1_score(labels, pred))

  total_score = 0
  total_loss = 0

  for (batch_id, batched_graph) in enumerate(batch_loader):
    node_features = batched_graph.x.to(device)
    edge_index = batched_graph.edge_index.to(device)
    labels = batched_graph.y.to(device)
    loss, score = evaluate_batch(node_features, edge_index, labels)
    total_loss += loss
    total_score += score

  avg_loss = total_loss / (batch_id + 1)
  avg_score = total_score / (batch_id + 1)

  return avg_loss, avg_score