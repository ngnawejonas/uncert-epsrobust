import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


"""# Temp Scaling"""

#@title temp visu utils
def calc_bins(preds, labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins, True)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels_oneh):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE

import matplotlib.patches as mpatches

def draw_reliability_graph(preds, labels_oneh):
  ECE, MCE = get_metrics(preds, labels_oneh)
  bins, _, bin_accs, _, _ = calc_bins(preds, labels_oneh)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True) 
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  plt.legend(handles=[ECE_patch, MCE_patch])

  #plt.show()
  
  plt.savefig('calibrated_network.png', bbox_inches='tight')

#draw_reliability_graph(preds)

#@title ece loss
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

#@title temp optimization
def get_logits_list(model, val_loader):
  # collect all the logits and labels for the validation set
  logits_list = []
  labels_list = []

  for i, data in enumerate(tqdm(val_loader, 0)):
      images, labels = data[0].to(device), data[1].to(device)
      model.eval()
      with torch.no_grad():
          logits_list.append(model(images))
          labels_list.append(labels)

  # Create tensors
  logits_list = torch.cat(logits_list).to(device)
  labels_list = torch.cat(labels_list).to(device)
  return logits_list, labels_list

def temperature_scale(logits, temperature):
    """
    Perform temperature scaling on logits
    """
    # # Expand temperature to match the size of logits
    # temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    # return logits / temperature
    # temperature = self.args.get('temperature', None)
    return torch.div(logits, temperature)

def grid_search_temperature(model, val_loader, Tgrid):
  model.to(device)

  nll_criterion = nn.CrossEntropyLoss().to(device)
  ece_criterion = _ECELoss().to(device)
  
  # First: collect all the logits and labels for the validation set
  logits_list, labels_list = get_logits_list(model, val_loader)

  # # Calculate NLL and ECE before temperature scaling
  before_temperature_nll = nll_criterion(logits_list, labels_list).item()
  before_temperature_ece = ece_criterion(logits_list, labels_list).item()
  print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
  
  #grid search 
  min_nll = before_temperature_nll
  optimal_temp = 1.0
  print(len(Tgrid))
  for temperature in tqdm(Tgrid):
    nll = nll_criterion(temperature_scale(logits_list, temperature), labels_list)
    if nll < min_nll:
      min_nll = nll
      optimal_temp = temperature 
  
  # Calculate NLL and ECE after temperature scaling
  after_temperature_nll = nll_criterion(temperature_scale(logits_list, optimal_temp), labels_list).item()
  after_temperature_ece = ece_criterion(temperature_scale(logits_list, optimal_temp), labels_list).item()
  print('Optimal temperature: %.3f' % optimal_temp)
  print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

  return optimal_temp


def optimize_temperature(model, val_loader):
  model.to(device)

  nll_criterion = nn.CrossEntropyLoss().to(device)
  ece_criterion = _ECELoss().to(device)

  # First: collect all the logits and labels for the validation set
  logits_list, labels_list = get_logits_list(model, val_loader)

  # # Calculate NLL and ECE before temperature scaling
  before_temperature_nll = nll_criterion(logits_list, labels_list).item()
  before_temperature_ece = ece_criterion(logits_list, labels_list).item()
  print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))


  temperature = nn.Parameter(torch.ones(1).to(device))
  # Next: optimize the temperature w.r.t. NLL
  # Removing strong_wolfe line search results in jump after 50 epochs
  optimizer = torch.optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')

  losses = []
  temps = []

  def _eval():
      loss = nll_criterion(temperature_scale(logits_list, temperature), labels_list)
      loss.backward()
      temps.append(temperature.item())
      losses.append(loss)
      return loss

  optimizer.step(_eval)
  
  # Calculate NLL and ECE after temperature scaling
  after_temperature_nll = nll_criterion(temperature_scale(logits_list, temperature), labels_list).item()
  after_temperature_ece = ece_criterion(temperature_scale(logits_list, temperature), labels_list).item()
  print('Optimal temperature: %.3f' % temperature.item())
  print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

  plt.subplot(121)
  plt.plot(list(range(len(temps))), temps)
  plt.title('temps')

  plt.subplot(122)
  xlosses = [x.detach().cpu() for x in losses]
  plt.plot(list(range(len(xlosses))), xlosses)
  plt.title('losses')
  plt.show()
  return temperature.item()

def get_probs(logits, temp=None, verbose=True):
    sm = nn.Softmax(dim=1)
    if temp is not None:
      logits = temperature_scale(logits, temp)
    probs = sm(logits)
    return probs.cpu().detach().numpy()

#@title ModelWithTemperature
class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = 1.0 #nn.Parameter(torch.ones(1).to(device))


    def forward(self, input):
        logits = self.model(input)
        return temperature_scale(logits, self.temperature)

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, val_loader=None, temp=None, Tgrid = None):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        val_loader (DataLoader): validation set loader
        """
        if val_loader is None:
          assert temp is not None
          self.temperature = temp #nn.Parameter(val.to(device))
        elif Tgrid is None :
          assert val_loader is not None
          self.temperature = optimize_temperature(self.model, val_loader)
        else:
          self.temperature = grid_search_temperature(self.model, val_loader, Tgrid)
