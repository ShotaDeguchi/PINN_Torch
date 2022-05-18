"""
********************************************************************************
executes training
********************************************************************************
"""

import argparse
import pathlib
import torch



if __name__ == "__main__":



model = PINN()
model.load_state_dict(torch.load("./saved/saved_model.pth"))
model.infer()

# or if fine-tuning, ....

