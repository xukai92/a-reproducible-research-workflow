import os, sys
PROJ_DIR = os.path.expanduser("~/projects/a-reproducible-research-workflow")
DATA_DIR = os.path.join(PROJ_DIR, "data")
SRC_DIR = os.path.join(PROJ_DIR, "src")
LOGS_DIR = os.path.join(PROJ_DIR, "logs")
SCRIPTS_DIR = os.path.join(PROJ_DIR, "scripts")
RESULTS_DIR = os.path.join(PROJ_DIR, "results", "exp_1")

import torch
import numpy as np

from munch import Munch
from icecream import ic
import wandb

import time

def run(project=None, name=None, notes=None, **kwargs):
    hps = Munch(**kwargs)
    ic("Running experiment 1")
    ic(hps)

    with wandb.init(
        project=project, name=name, notes=notes, dir=LOGS_DIR, config=kwargs, reinit=True,
        config_exclude_keys=["gpu_id", "num_epochs"]
    ) as run:
        results_dir = os.path.join(RESULTS_DIR, run.name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for epoch in range(hps.num_epochs):
            time.sleep(0.2)
            wandb.log({"epoch": epoch,
                       "train/loss": hps.num_epochs - np.random.randn(),
                       "test/loss": hps.num_epochs - np.random.randn(),
                       "test/accuracy": np.random.rand()})

    torch.save(hps, os.path.join(results_dir, "hps.pt"))
    #torch.save(model.state_dict(), os.path.join(results_dir, "parameters.pt"))

    metrics = {"accuracy": np.random.rand()}
    torch.save(metrics, os.path.join(results_dir, "metrics.pt"))

import toml
import argparse

if __name__ == "__main__":
    hps = toml.load(os.path.join(SCRIPTS_DIR, "hps-exp_1.toml"))["default"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="a-reproducible-research-workflow")
    parser.add_argument("--name",    default=None)
    parser.add_argument("--notes",   default=None)
    # Based on https://stackoverflow.com/a/37367814
    args, unknown = parser.parse_known_args() # this is an 'internal' method
    # which returns 'parsed', the same as what parse_args() would return
    # and 'unknown', the remainder of that
    # the difference to parse_args() is that it does not exit when it finds redundant arguments
    for arg in unknown:
        if arg.startswith(("-", "--")):
            k, v = arg.split('=')
            k = k.replace("--", "")
            k = k.replace("-", "_")
            assert k in hps, f"unknown arg: {k=}"
            v_new = type(hps[k])(eval(v))
            print(f"Overwriting hps.{k} from {hps[k]} to {v_new}")
            hps[k] = v_new
    run(project=args.project, name=args.name, notes=args.notes, **hps)
