#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n fastbdt37_2 gcc python=3.7
conda activate fastbdt37_2
echo -e "\n"
echo -e "\n"
echo "--> which python3 $(which python3)"
echo "--> which pip3 $(which pip3)"

conda install cmake
echo -e "\n"
echo -e "\n"
echo "--> which cmake $(which cmake)"

conda install make
echo -e "\n"
echo -e "\n"
echo "--> which make $(which make)"

echo -e "\n"
echo -e "\n"
echo "--> Installing setuptools, numpy, scikit-learn..."
pip3 install setuptools==58.2.0
pip3 install "numpy<=1.20"
pip3 install scikit-learn

echo -e "\n"
echo -e "\n"
echo "--> Clonning FastBDT..."
mkdir FastBDT_git
cd FastBDT_git

rm -rf FastBDT
git clone https://github.com/thomaskeck/FastBDT.git
cd FastBDT
sed -ie '19i #include <limits>\n' include/FastBDT.h.in

echo "--> PWD $(pwd)"
echo "--> Installing FastBDT..."
cmake .
make
sudo make install
sudo python3 setup.py install

echo -e "\n"
echo -e "\n"
echo "---->>>> DONE!"
