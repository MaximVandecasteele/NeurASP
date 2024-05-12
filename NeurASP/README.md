# NeurASP
This is the implementation of [NeurASP: Embracing Neural Networks into Answer Set Programming](https://www.ijcai.org/proceedings/2020/0243.pdf).  
[Lab page](https://azreasoners.github.io/ARG-webpage/)
# Introduction
NeurASP is a simple extension of answer set programs by embracing neural networks. By treating the neural network output as the probability distribution over atomic facts in answer set programs, NeurASP provides a simple and effective way to integrate sub-symbolic and symbolic computation. This repository includes examples to show
1. how NeurASP can make use of pretrained neural networks in symbolic computation and how it can improve the perception accuracy of a neural network by applying symbolic reasoning in answer set programming; and
2. how NeurASP is used to train a neural network better by training with rules so that a neural network not only learns from implicit correlations from the data but also from the explicit complex semantic constraints expressed by ASP rules.

## Installation
0. We assume Anaconda is installed. One can install it according to its [installation page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
1. Clone this repo:
```
git clone https://github.com/azreasoners/NeurASP
cd NeurASP
```
2. Create a virtual environment `neurasp`. Install clingo (ASP solver) and tqdm (progress meter).
```
conda create --name neurasp python=3.9
conda activate neurasp
conda install -c potassco clingo=5.5 tqdm
```
3. Install Pytorch according to its [Get-Started page](https://pytorch.org/get-started/locally/). Below is an example command we used on Linux with cuda 10.2. (PyTorch version 1.12.0 is tested.)
```bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Examples
We provide 3 inference and 5+4 learning examples as shown below. Each example is stored in a separate folder with a readme file.
### Inference Examples
* [Sudoku](https://github.com/azreasoners/NeurASP/tree/master/examples/sudoku)
* [Offset Sudoku](https://github.com/azreasoners/NeurASP/tree/master/examples/offset_sudoku)
* [Toy-car](https://github.com/azreasoners/NeurASP/tree/master/examples/toycar)

### Learning Examples
* [MNIST Addition](https://github.com/azreasoners/NeurASP/tree/master/examples/mnistAdd)
* [Shorstest Path](https://github.com/azreasoners/NeurASP/tree/master/examples/shortest_path)
* [Sudoku Solving](https://github.com/azreasoners/NeurASP/tree/master/examples/solvingSudoku_70k)
* [Top-k](https://github.com/azreasoners/NeurASP/tree/master/examples/top_k)
* [Most Reliable Path](https://github.com/azreasoners/NeurASP/tree/master/examples/most_reliable_path)
* Examples from NeuroLog paper: [add2x2](https://github.com/azreasoners/NeurASP/tree/master/examples/add2x2), [apply2x2](https://github.com/azreasoners/NeurASP/tree/master/examples/apply2x2), [member3](https://github.com/azreasoners/NeurASP/tree/master/examples/member3), [member5](https://github.com/azreasoners/NeurASP/tree/master/examples/member5)

## Related Work
You may also be interested in our work [Injecting Logical Constraints into Neural Networks via Straight-Through-Estimators](https://azreasoners.github.io/ARG-webpage/pdfs/ste-ns-icml.pdf). Its codes are available [here](https://github.com/azreasoners/cl-ste).

## Citation
Please cite our paper as:
```
@inproceedings{ijcai2020p243,
  title     = {NeurASP: Embracing Neural Networks into Answer Set Programming},
  author    = {Yang, Zhun and Ishay, Adam and Lee, Joohyung},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Christian Bessiere},
  pages     = {1755--1762},
  year      = {2020},
  month     = {7},
  note      = {Main track},
  doi       = {10.24963/ijcai.2020/243},
  url       = {https://doi.org/10.24963/ijcai.2020/243},
}

```
