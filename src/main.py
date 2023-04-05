# from dis import dis
from pprint import pprint
import os
import argparse
import sys
import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
from ray import tune
from ray.tune import CLIReporter
import wandb
import seaborn as sns

from attacks import deepfool, is_attack_successful
from robustbench.utils import load_model
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

from utils import compute_norm, get_dataset, set_seeds
from scores import brier_score, compute_entropy, get_clever_scores, get_min_max_pixel_values, max_probability, quadratic_score
from tqdm import tqdm

from temp_scaling import ModelWithTemperature, get_logits_list, get_probs



def parse_args(args: list) -> argparse.Namespace:
    """Parse command line parameters.

    :param args: command line parameters as list of strings (for example
        ``["--help"]``).
    :return: command line parameters namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the models for this experiment."
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--no-ray",
        action="store_true",
        default=False,
        help="run without ray",
    )
    parser.add_argument(
        "--dataset-path",
        default="/home-local2/jongn2.extra.nobkp/data",
        help="the path to the dataset",
        type=str,
    )
    parser.add_argument(
        "--cpus-per-trial",
        default=1,
        help="the number of CPU cores to use per trial",
        type=int,
    )
    parser.add_argument(
        "--project-name",
        help="the name of the Weights and Biases project to save the results",
        # required=True,
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="dataset used",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--debug-strategy",
        help="the strategy to use in debug mode",
        default="Random",
        type=str,
    )

    return parser.parse_args(args)


def tune_report(no_ray, **args):
    if not no_ray and tune.is_session_enabled():
        tune.report(**args)

def _avoid_zero_len(metric, dis_list):
    return metric(dis_list) if len(dis_list)>0 else 0.

def logdist_metrics(dis_list, name, rd, n_labeled):
    logdict = {'avg '+name: _avoid_zero_len(np.mean, dis_list),
               'min '+name: _avoid_zero_len(np.min, dis_list),
               'max '+name: _avoid_zero_len(np.max, dis_list),
               'median '+name: _avoid_zero_len(np.median, dis_list),
               'round ': rd,
               'n_labeled': n_labeled}
    return logdict

def test(model,test_dataloader, device=None, verbose=True, attackfn=None, **kwargs):
  preds = []
  labels_oneh = []
  correct = 0
  model.eval()
  with torch.no_grad():
      looper = tqdm(test_dataloader) if verbose else test_dataloader
      for data in looper:
          images, labels = data[0].to(device), data[1].to(device)
          if attackfn is None:
            pred = model(images)
          else:
            with torch.enable_grad():
              images = attackfn(model, images, **kwargs)
            pred = model(images)
          
          # Get softmax values for net input and resulting class predictions
          sm = nn.Softmax(dim=1)
          pred = sm(pred)

          _, predicted_cl = torch.max(pred.data, 1)
          pred = pred.cpu().detach().numpy()

          # Convert labels to one hot encoding
          label_oneh = torch.nn.functional.one_hot(labels, num_classes=10)
          label_oneh = label_oneh.cpu().detach().numpy()

          preds.extend(pred)
          labels_oneh.extend(label_oneh)

          # Count correctly classified samples for accuracy
          correct += sum(predicted_cl == labels).item()

  preds = np.array(preds).flatten()
  labels_oneh = np.array(labels_oneh).flatten()

  correct_perc = correct / len(test_dataloader)
  if verbose:
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct_perc))
    # print(correct_perc)
  
  return preds, labels_oneh


def run_trial_empty(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    print("DO NOTHING AND EXIT")


def run_trial(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    """Train a single model according to the configuration provided.

    :param config: The trial and model configuration.
    :param params: The hyperparameters.
    :param args: The program arguments.
    """

    #
    resultsDirName = 'results'
    try:
        os.mkdir(resultsDirName)
        print("Results directory ", resultsDirName,  " Created ")
    except FileExistsError:
        print("Results directory ", resultsDirName,  " already exists")

    # fix random seed
    set_seeds(config['seed'])
    # if args.dry_run:
    #     wandb.init(project=args.project_name, mode="disabled")
    # else:
    #     id = 0 if args.no_ray else tune.get_trial_id()
    #     exp_name = '{}_run_{}_{}_seed{}'.format(
    #         config['dataset_name'], id, config['strategy_name'], config['seed'])
    #     wandb.init(project=args.project_name, name=exp_name, config=config)
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using GPU: {use_cuda}')

    """# Dataset"""

    #@title cifar10
    # Normalize the images by the imagenet mean/std since the nets are pretrained
    transform = transforms.Compose([transforms.ToTensor(),])
        #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    valid_size = 5000
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-valid_size, valid_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                            shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128,
                                            shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                            shuffle=False, num_workers=2)

    # x_test, y_test = load_cifar10(n_examples=10000)

    #@title single batch test loaders
    xtest_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                            shuffle=False, num_workers=2)
    
    subset_test_set = torch.utils.data.Subset(test_set, np.arange(params['nsubset']))
    xsubset_test_loader = torch.utils.data.DataLoader(subset_test_set, batch_size=1,
                                            shuffle=False, num_workers=2)


    #Load Model
    model = load_model(model_name=config['model_name'], dataset=config['dataset_name'], threat_model='Linf')
    model = model.to(device)
    model.eval()

    ## Temp Scaling
    scaled_model = ModelWithTemperature(model)
    scaled_model.set_temperature(val_loader)

    scaled_model.eval()

    ## 
    xlogits_list, xlabels_list = get_logits_list(scaled_model, xsubset_test_loader) 
    preds_confidence = get_probs(xlogits_list)
    entropies = compute_entropy(preds_confidence, 1)
    qscores = quadratic_score(xlabels_list, preds_confidence, 1)
    br_scores = brier_score(xlogits_list.cpu().numpy(), preds_confidence, 1)
    max_probs = max_probability(preds_confidence)
    names = ['entropies', 'qscores', 'br_scores', 'max_probs']
    scores = [entropies, qscores, br_scores, max_probs]
    for i, val in enumerate(scores):
        with open(f'{names[i]}.npy', 'wb') as f:
            np.save(f, val)

    """## clever score"""
    minpx , maxpx = get_min_max_pixel_values(test_loader)

    clever_args={'min_pixel_value':  minpx,
                'max_pixel_value': maxpx,
                'nb_batches':20,
                'batch_size':50,
                'norm':np.inf,
                'radius':0.3,
                'pool_factor':5}
    clever_scores = get_clever_scores(scaled_model, xsubset_test_loader, **clever_args)
    with open('clever_scores_inf.npy', 'wb') as f:
        np.save(f, clever_scores)

    clever_args={'min_pixel_value':  minpx,
                'max_pixel_value': maxpx,
                'nb_batches':20,
                'batch_size':50,
                'norm':2,
                'radius':0.5,
                'pool_factor':5}
    clever_scores = get_clever_scores(scaled_model, xsubset_test_loader, **clever_args)
    with open('clever_scores_2.npy', 'wb') as f:
        np.save(f, clever_scores)

    """## attacks distances"""

    deepfool_args={'max_iter': 200}

    deepfool_scores_inf = []
    deepfool_scores_2 = []
    for images, labels in tqdm(xsubset_test_loader):
        images, labels = images.to(device), labels.to(device)
        x_adv = deepfool(scaled_model, images, **deepfool_args).detach()
        with torch.no_grad():
            attack_success = is_attack_successful(model, images, x_adv)
            dis = compute_norm((x_adv-images), 2) if attack_success else np.inf
            deepfool_scores_2.append(dis)
            dis = compute_norm((x_adv-images), np.inf) if attack_success else np.inf
            deepfool_scores_inf.append(dis)

    deepfool_scores_inf = np.array(deepfool_scores_inf)
    deepfool_scores_2 = np.array(deepfool_scores_2)
    with open('deepfool_inf.npy', 'wb') as f:
        np.save(f, deepfool_scores_inf)
    with open('deepfool_2.npy', 'wb') as f:
        np.save(f, deepfool_scores_2)

    pgd_params = {'eps': 0.3, 'eps_iter': 0.001, 'nb_iter': 100, 'norm': np.inf, 'targeted': False, 'rand_init': False}

    bim_inf_scores2 = []
    bim_inf_scores_inf = []
    for images, labels in tqdm(xsubset_test_loader):
        images, labels = images.to(device), labels.to(device)
        x_adv = pgd(scaled_model, images, **pgd_params).detach()
        with torch.no_grad():
            attack_success = is_attack_successful(scaled_model, images, x_adv)
            dis = compute_norm((x_adv-images), 2) if attack_success else np.inf
            bim_inf_scores2.append(dis)
            dis = compute_norm((x_adv-images), np.inf) if attack_success else np.inf
            bim_inf_scores_inf.append(dis)
    bim_inf_scores_inf = np.array(bim_inf_scores_inf)
    bim_inf_scores2 = np.array(bim_inf_scores2)
    with open('bim_inf_score2.npy', 'wb') as f:
        np.save(f, bim_inf_scores2)
    with open('bim_inf_score_inf.npy', 'wb') as f:
        np.save(f, bim_inf_scores_inf)


    pgd_params = {'eps': 0.5, 'eps_iter': 0.01, 'nb_iter': 100, 'norm': 2, 'targeted': False, 'rand_init': False}

    bim_2_scores2 = []
    bim_2_scores_inf = []
    for images, labels in tqdm(xsubset_test_loader):
        images, labels = images.to(device), labels.to(device)
        x_adv = pgd(scaled_model, images, **pgd_params).detach()
        with torch.no_grad():
            attack_success = is_attack_successful(scaled_model, images, x_adv)
            dis = compute_norm((x_adv-images), 2) if attack_success else np.inf
            bim_2_scores2.append(dis)
            dis = compute_norm((x_adv-images), np.inf) if attack_success else np.inf
            bim_2_scores_inf.append(dis)
    bim_2_scores_inf = np.array(bim_2_scores_inf)
    bim_2_scores2 = np.array(bim_2_scores2)
    with open('bim_2_score2.npy', 'wb') as f:
        np.save(f, bim_2_scores2)
    with open('bim_2_score_inf.npy', 'wb') as f:
        np.save(f, bim_2_scores_inf)

    cw_params = {'n_classes':10, 'targeted': False}

    cwdistances2 = []
    cwdistancesinf = []
    for images, labels in tqdm(xsubset_test_loader):
        images, labels = images.to(device), labels.to(device)
        x_adv = carlini_wagner_l2(scaled_model, images, **cw_params).detach()
        with torch.no_grad():
            attack_success = is_attack_successful(scaled_model, images, x_adv)
            dis = compute_norm((x_adv-images), 2) if attack_success else np.inf
            cwdistances2.append(dis)
            dis = compute_norm((x_adv-images), np.inf) if attack_success else np.inf
            cwdistancesinf.append(dis) 
    cwdistances2 = np.array(cwdistances2)
    cwdistancesinf = np.array(cwdistancesinf)
    with open('cw2_inf.npy', 'wb') as f:
        np.save(f, cwdistancesinf)
    with open('cw2_2.npy', 'wb') as f:
        np.save(f, cwdistances2)


def run_experiment(params: dict, args: argparse.Namespace) -> None:
    """Run the experiment using Ray Tune.

    :param params: The hyperparameters.
    :param args: The program arguments.
    """
    config = {
        "dataset_name": params['dataset_name'],
        "model_name": tune.grid_search(params["model_names"]),
        "seed": tune.grid_search(params["seeds"]),
    }
    if args.dry_run:
        config = {
            "strategy_name": args.debug_strategy,
            "seed": 42,
        }

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpus_per_trial = 1 if use_cuda else 0

    if args.no_ray:
        run_trial(config=config, params=params, args=args, num_gpus=gpus_per_trial)
    else:
        reporter = CLIReporter(
            parameter_columns=["seed", "model_name"],
            # metric_columns=["round"],
        )

        tune.run(
            tune.with_parameters(
                run_trial, params=params, args=args, num_gpus=gpus_per_trial
            ),
            resources_per_trial={
                "cpu": args.cpus_per_trial, "gpu": gpus_per_trial},
            config=config,
            progress_reporter=reporter,
            name=args.project_name,
        )


def main(args: list) -> None:
    """Parse command line args, load training params, and initiate training.

    :param args: command line parameters as list of strings.
    """
    args = parse_args(args)
    paramsfilename = f'./params_{args.dataset}.yaml'
    with open(paramsfilename, 'r') as param_file:
        params = yaml.load(param_file, Loader=yaml.SafeLoader)
    run_experiment(params, args)


def run() -> None:
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`
    This function can be used as entry point to create console scripts.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
