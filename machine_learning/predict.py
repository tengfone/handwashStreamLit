import sys
from machine_learning.model import *
from machine_learning.utils import *
import torch


def pred(argv) -> list:
    compiledOutput = []
    # Change Saved Model Here
    savedModelPath = './machine_learning/model/alexnet_128.pt'
    arch = 'alexnet'

    # Set CPU
    device = torch.device('cpu')

    model = build_model(arch)
    model.to(device)

    model = load_model(model, savedModelPath)

    for eachVideoPath in argv:
        initFileName = eachVideoPath.split('/')[-1].split('.')[0]
        output = predict(model, eachVideoPath)
        tempOutput = str(initFileName + ':' + output)
        compiledOutput.append(tempOutput)

    return compiledOutput
