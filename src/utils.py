import random
import numpy as np
import torch 

def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

def compute_norm(x, norm):
  with torch.no_grad():
      if norm == np.inf:
          return torch.linalg.norm(torch.ravel(x.cpu()), ord=np.inf).numpy()
      elif norm == 2:
          return torch.linalg.norm(x.cpu()).numpy()
      else:
          raise NotImplementedError
