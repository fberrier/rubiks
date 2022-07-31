

c5a.16xlarge


sudo apt-get update
sudo apt install build-essential
sudo apt-get install emacs
sudo apt-get install git
git clone https://github.com/fberrier/rubiks
sudo apt-get install curl
sudo apt update
cd /tmp/
curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash anaconda.sh
cd
source .bashrc

cd
pip install -e rubiks
conda create -n rubiks python=3.8
conda activate rubiks
conda install coloredlogs tabulate pandas numpy -y
conda install -c pytorch pytorch -y
conda install -c conda-forge matplotlib -y
conda install -c conda-forge/label/gcc7 sortedcontainers -y


