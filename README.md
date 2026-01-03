# Enhancing Movie Recommender Systems via Transformer and GNN Integration
**Final Project of Social Network Analysis**  
**Authors: Ewan Long, Murad Aladdinzade**

## Abstract
Recent advancements in recommender systems have been driven by Graph Neural Networks (GNNs) and Transformer architectures, each excelling in structural and sequential modeling, respectively. GNNs effectively capture localized user–item interaction patterns but often suffer from limited receptive fields and vulnerability to structural noise. Conversely, Transformers model long-range dependencies but lack an inherent understanding of graph topology. In this work, we adapt the TransGNN framework (Zhang et al., 2024), which strategically alternates Transformer and GNN layers to leverage their complementary strengths. We implement and evaluate TransGNN on the MovieLens 10M dataset, achieving a Recall@20 of 0.3068 and an NDCG@20 of 0.2891, with Recall@40 and NDCG@40 reaching 0.4350 and 0.3240, respectively. Experimental results demonstrate that TransGNN achieves competitive performance on top-K recommendation tasks, offering a balanced trade-off between global semantic aggregation and local structural modeling. The proposed adaptation shows strong potential for deployment in large-scale, real-world recommendation scenarios.

## Table of Contents
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [Resources](#resources)

## Requirements
We recommend using **conda** to set up the environment:

```bash
# Create and activate environment
conda create -n S25_DS5720_Project-main python=3.9.12
conda activate S25_DS5720_Project-main

# Install core dependencies
conda install numpy=1.22.3 scipy=1.7.3
conda install pytorch=2.2.2 torchvision=0.17.2 torchaudio=2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Install additional packages
conda install -c conda-forge pytorch_geometric
conda install -c conda-forge setproctitle
conda install -c conda-forge scikit-learn
conda install matplotlib
conda install seaborn
```

## Dataset

We use the [MovieLens 10M Dataset](https://grouplens.org/datasets/movielens/10m/), which contains:
- 10,000,054 ratings and 95,580 tags
- 71,567 users and 10,681 movies
- Each user has rated at least 20 movies
- No demographic information is included; users are anonymized

**Files:**
- `ratings.dat`: Contains entries of the form `UserID::MovieID::Rating::Timestamp`
- `tags.dat`: Contains entries of the form `UserID::MovieID::Tag::Timestamp`
- `movies.dat`: Contains entries of the form `MovieID::Title::Genres`

The dataset is publicly available and free for research use under the [GroupLens Usage License](https://grouplens.org/datasets/movielens/).

## Usage

After setting up the environment and downloading the dataset:

1. **Create folders** in the parent directory:
   ```bash
   mkdir Models
   mkdir History
   ```
### Training Output

- Training history will be automatically saved into the `History/` directory.
- Each `.his` file contains four tracked metrics:
  - `TrainLoss`: Training loss per epoch
  - `TrainpreLoss`: Pre-training loss (if applicable)
  - `TestRecall`: Recall@*K* on the validation/test set
  - `TestNDCG`: NDCG@*K* on the validation/test set

### Running the Model

- After setting up the environment and dataset, you can run the training or evaluation scripts by executing the following commands:
```bash
python preprocess_ml10m.py
python Main.py
```
- Make sure that `Models/` and `History/` folders exist in the parent directory.
- Make sure renaming the downloaded dataset file name to `ml10m` and placing it in the `data/` folder.

## Visualizations

After training the model, you can generate visualizations by running:

```bash
python model_visualization.py
python visualize_training.py
```
- `model_visualization.py`:
    - Visualizes user and item embeddings via t-SNE and PCA.
    - Generates user–item recommendation heatmaps.
    - Plots Recall@*K* and NDCG@*K* performance curves.
- `visualize_training.py`:
    - Visualizes training and testing loss curves.
    - Plots additional metrics such as Recall and NDCG over epochs.
  
All output images are saved in the `Visualization/` directory.

## Resources

This project was completed as part of the **Social Network Analysis** course final project.  
It is based on the adaptation and modification of the following original resources:

- **TransGNN: Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems**
  - [Original GitHub Repository](https://github.com/Peiyance/TransGNN-torch/tree/main?tab=readme-ov-file)
  - [Original Paper on arXiv](https://arxiv.org/abs/2308.14355)

- **MovieLens 10M Dataset**
  - [Official Dataset Website](https://grouplens.org/datasets/movielens/10m/)

We acknowledge and appreciate the contributions of the original authors and data providers.
