# Advanced Programming Techniques - Project

## Project Description
In this project C++ is used to create a Neural Network to detect digits from the MNIST dataset.
Batching is implemented from the get-go.

## Project Structure
The project is divided into the following directories:
- **src**: Contains the source files of the project. In Layers the FullyConnectedLayer, the ReLU and the SoftMax are implemented. In the folder Optimization the CrossEntropyLoss and the SGD as optimizer are implemented.
- **mnist-datasets**: Contains the MNIST dataset.
- **eigen**: Contains the Eigen library. This is setup with git submodule.
- **pytorch**: Contains a pytorch implementation of the project.
- **sandbox**: Contains all toy and test files to play around with the different layers of the Neural Network.

## How to compile the project
Use the makefile to compile the project.
Make sure that the executables files are .exe files such that they are ignored.

## How to run the project
To run the project, you need to run the following commands:
```bash
bash read_dataset_images.sh mnist-datasets/train-images.idx3-ubyte image_out.txt 0
bash read_dataset_labels.sh mnist-datasets/train-labels.idx1-ubyte label_out.txt 0
python3 compare_files.py label_out.txt expected-results/out-tensor-single-label.txt
```
Note that the executable files are located in the exes directory.
Look into the .gitlab-ci.yml file to see more details.

### Reference: pytorch implementation
One can easily run the pytorch implementation by running the following command:
```bash
python3 ./pytorch/main.py
```

## Setup Eigen library on gitlab
To setup the Eigen library on gitlab, you need to run the following commands:
```bash
cd /path/to/your/project
git submodule add https://gitlab.com/libeigen/eigen.git
git submodule update --init --recursive
```

## Cloning the project
To clone the project, you need to run the following commands:
```bash
git clone --recurse-submodules <repository-url>
```

## TODOs
- [ ] Later push this project first to "musketeer" git and then to i10gitlab server to check tests
  - setup eigen
  - copy files

- [ ] Accelerate code.
