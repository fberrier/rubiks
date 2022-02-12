#!/bin/bash
conda install pytorch torchvision torchaudio -c pytorch -y
conda install cudatoolkit -c nvidia -y
conda install mlpack -c conda-forge -y
conda install matplotlib ipython pandas numpy thrift tensorboard scipy tabulate sortedcontainers coloredlogs -y
