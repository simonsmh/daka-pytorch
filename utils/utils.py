import torch
import numpy as np
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger("PyTorch Daka")

def StrtoLabel(Str):
    return [int(t) for t in Str]

def LabeltoStr(Label):
    Str = str()
    for i in Label:
        Str += chr(ord('0') + int(i))
    return Str

if __name__ == "__main__":
    a = StrtoLabel("98")
    print(a)
    print(LabeltoStr(a))
