# Explainable Banana Ripeness Classification

## About the Repository

This repository contains the code related to the paper titled "Stop overkilling simple tasks with black-box models, use
more transparent models instead." The paper challenges the overuse of opaque, deep learning models for simple tasks and
advocates for more transparent, explainable models in machine learning.

## Repository Overview

The code in this repository demonstrates an approach to classifying the ripeness of bananas. The approach is designed to
be more transparent and explainable compared to traditional deep learning methods.

## Abstract

The ability of deep learning-based approaches to extract features autonomously from raw data while outperforming
traditional methods has led to several breakthroughs in artificial intelligence. However, it is well-known that deep
learning models suffer from an intrinsic opacity, making it difficult to explain why they produce specific predictions.
This is problematic not only because it hinders debugging but, most importantly, because it negatively affects the
perceived trustworthiness of the systems. What is often overlooked is that many relatively simple tasks can be solved
efficiently and effectively with data processing strategies paired with traditional models that are inherently more
transparent. This work highlights the frequently neglected perspective of using knowledge-based and
explainability-driven problem-solving in ML. To support our guidelines, we propose a simple strategy for solving the
task of classifying the ripeness of banana crates. This is done by planning explainability and model design together. We
showcase how the task can be solved using opaque deep learning models and more transparent strategies. Notably, there is
a minimal loss of accuracy but a significant gain in explainability, which is truthful to the model's inner workings.
Additionally, we perform a user study to evaluate the perception of explainability by end users and discuss our
findings.

## Installation

```bash
# Create a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install Requirements
pip install -r requirements.txt
```

The code in this repository is based on the [PyTorch](https://pytorch.org/) machine learning framework. To install the
required packages, please run `pip3 install -r requirements` within the root folder of the project.

### Method and Usage

We compare accuracy and explainability for the banana ripeness classification task (from 1, under-ripe, to 4, over-ripe)
using four different models. Three of these models are neural networks operating on raw data:

* Vanilla CNN (**CNN**)

* **MobileNet V2**

* Vision Transformer (**ViT**)

We try to explain these models using the model agnostic approach called [SHAP](https://arxiv.org/abs/1705.07874). Albeit
intuitive, the explanations produced by these methods do not seem human-intelligible.

We compare the deep learning approach to a simple **decision tree** working on the average RGB channels of the input
images, that reaches comparable accuracy while offering more room for explainability. Namely, we design the
explainability of the model toegether with the model itself. Thus, we can propose as explanation a mapping between the
regions of the RGB color space the decision tree associates with each class and the RGB values of some input.

## Accessing the Dataset

To download the dataset that was used for the experiments reported in the paper please use [this
link](https://drive.google.com/file/d/15Y4hZIMrieDhUvDoL4bMkAB574y0BLsz/view?usp=sharing)

## Running the Deep Learning Models

To train the three deep learning models available, please
run `python3 src/classifiers/deep_learning/run_dl_experiments.py`. The run can be configured by editing the yaml files
within the `src/classifiers/deep_learning/params` folder. To choose from the three deep learning model please edit the
network type in the `experiment.yml` file.

### Explanation Methods

To run SHAP please refer to the main.py at `python3 src/classifiers/deep_learning/explainability/main.py`.

## Running the Decision Tree

To train the decision tree, please run `python3 src/classifiers/decision_tree/run_dt_experiments.py`. The run can be
configured by editing the yaml files within the `src/classifiers/decision_tree/params` folder. To choose the parameters
for the decision tree, edit them in the `experiment.yml` file.

### Explanation Method

To explain the decision tree please follow these commands:

* `python3 src/classifiers/decision_tree/scripts/standardize_luma.py` to standardize the luminosity of the image based
  on the Y channel of the YUV color space;

* `python3 src/classifiers/decision_tree/scripts/generate_dt_features.py` to compute the features to be used by the
  decision tree;

* `python3 src/classifiers/decision_tree/scripts/explain.py` to generate the explanations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.