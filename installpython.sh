conda create -n rubikspy python=3.8 -y
conda activate rubikspy
conda install pytorch torchvision torchaudio -c pytorch -y
conda install cudatoolkit -c nvidia -y
conda install ipython pandas numpy thrift -y
conda install matplotlib -y
conda install mlpack -c conda-forge -y
conda install tensorboard scipy tabulate sortedcontainers -y
