"""
********************************************************************************
executes training
********************************************************************************
"""

import argparse
import pathlib
import torch

from pinn import PINN

def parse_option():


def main(args):


if __name__ == "__main__":

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PINN().to(device)

model.train()
torch.save(model.state_dict(), "./saved/saved_model.pth")
