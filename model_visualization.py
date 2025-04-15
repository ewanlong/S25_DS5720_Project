import torch as t
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from Params import args
from Model import TransGNN
from DataHandler import DataHandler
import random

# Set the device
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

# Ensure the output directory exists
output_dir = '../Visualization/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the model
models_dir = '../Models/'
if not os.path.exists(models_dir):
    print("Models directory does not exist. Please make sure the model has been trained and saved.")
    exit()

# Check if a model name has been specified; if not, prompt the user for input
if args.load_model is None:
    print("No model name specified for loading.")
    print("Available model files:")
    available_models = []
    for file in os.listdir(models_dir):
        if file.endswith('.mod'):
            model_name = file[:-4]
            available_models.append(model_name)
            print(f"  - {model_name}")
    
    if available_models:
        model_name = input("Please enter the name of the model to visualize (without the .mod extension): ").strip()
        if not model_name:
            model_name = available_models[0]  # Use the first available model as the default
            print(f"Using default model: {model_name}")
    else:
        print("No available model files found")
        exit()
else:
    model_name = args.load_model

# Try loading the specified model
try:
    ckp = t.load(models_dir + model_name + '.mod', map_location=device)
    model = ckp['model']
    print(f"Successfully loaded model: {model_name}")
except Exception as e:
    print(f"Unable to load model: {model_name}. Please check the file path and name.")
    # List available model files
    print("Available model files:")
    for file in os.listdir(models_dir):
        if file.endswith('.mod'):
            print(f"  - {file[:-4]}")
    exit()

# Load data
handler = DataHandler()
handler.LoadData()

# 1. Embedding Space Visualization
def visualize_embeddings(model_name):
    print("Generating embedding space visualization...")
    # Get user and item embeddings
    user_embeds, item_embeds = model.predict(handler.torchBiAdj)
    
    # Convert embeddings to numpy arrays
    user_embeds = user_embeds.detach().cpu().numpy()
    item_embeds = item_embeds.detach().cpu().numpy()
    
    # Randomly select a subset of users and items for visualization (to avoid clutter)
    num_users_to_plot = min(500, user_embeds.shape[0])
    num_items_to_plot = min(500, item_embeds.shape[0])
    
    user_indices = np.random.choice(user_embeds.shape[0], num_users_to_plot, replace=False)
    item_indices = np.random.choice(item_embeds.shape[0], num_items_to_plot, replace=False)
    
    selected_user_embeds = user_embeds[user_indices]
    selected_item_embeds = item_embeds[item_indices]
    
    # Combine user and item embeddings for dimensionality reduction
    combined_embeds = np.vstack([selected_user_embeds, selected_item_embeds])
    
    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    combined_2d = tsne.fit_transform(combined_embeds)
    
    # Separate the t-SNE results for users and items
    user_2d = combined_2d[:num_users_to_plot]
    item_2d = combined_2d[num_users_to_plot:]
    
    # Plot the scatter plot for t-SNE results
    plt.figure(figsize=(10, 8))
    plt.scatter(user_2d[:, 0], user_2d[:, 1], c='blue', label='Users', alpha=0.5)
    plt.scatter(item_2d[:, 0], item_2d[:, 1], c='red', label='Items', alpha=0.5)
    plt.title('t-SNE Visualization of User and Item Embeddings')
    plt.legend()
    plt.savefig(output_dir + model_name + '_embeddings_tsne.png')
    plt.close()
    
    # Reduce dimensions using PCA for comparison
    pca = PCA(n_components=2)
    combined_2d_pca = pca.fit_transform(combined_embeds)
    
    # Separate the PCA results for users and items
    user_2d_pca = combined_2d_pca[:num_users_to_plot]
    item_2d_pca = combined_2d_pca[num_users_to_plot:]
    
    # Plot the scatter plot for PCA results
    plt.figure(figsize=(10, 8))
    plt.scatter(user_2d_pca[:, 0], user_2d_pca[:, 1], c='blue', label='Users', alpha=0.5)
    plt.scatter(item_2d_pca[:, 0], item_2d_pca[:, 1], c='red', label='Items', alpha=0.5)
    plt.title('PCA Visualization of User and Item Embeddings')
    plt.legend()
    plt.savefig(output_dir + model_name + '_embeddings_pca.png')
    plt.close()
    
    print(f"Embedding space visualization saved to {output_dir}")

# 2. Recommendation Results Visualization
def visualize_recommendations(model_name):
    print("Generating recommendation visualization...")
    # Get user and item embeddings
    user_embeds, item_embeds = model.predict(handler.torchBiAdj)
    
    # Randomly choose several users for recommendations
    num_users_to_recommend = 5
    num_recommendations = 10
    
    random_users = np.random.choice(args.user, num_users_to_recommend, replace=False)
    
    # Compute recommended items for each user
    for i, user_id in enumerate(random_users):
        # Get the user embedding
        user_embed = user_embeds[user_id].unsqueeze(0)
        
        # Compute the similarity between the user and all items
        scores = t.mm(user_embed, item_embeds.t()).squeeze()
        
        # Get the items that the user has already interacted with in the training set
        user_interacted_items = set()
        trnMat = handler.loadOneFile(handler.trnfile)
        
        # Fix the sparse matrix access issue:
        # Convert the sparse matrix to coordinate format and then check the user interactions
        if hasattr(trnMat, 'tocoo'):
            trnMat = trnMat.tocoo()
            
        # Gather the user's interactions
        for u, i, v in zip(trnMat.row, trnMat.col, trnMat.data):
            if u == user_id and v > 0:
                user_interacted_items.add(i)
        
        # Filter out the items the user has already interacted with
        scores_with_indices = [(score.item(), idx) for idx, score in enumerate(scores) if idx not in user_interacted_items]
        scores_with_indices.sort(reverse=True)
        
        # Get the top N recommended items
        top_recommendations = scores_with_indices[:num_recommendations]
        
        # Print the recommendation results
        print(f"\nRecommendations for User {user_id}:")
        for rank, (score, item_id) in enumerate(top_recommendations):
            print(f"  Rank {rank+1}: Item {item_id}, Score {score:.4f}")
    
    # Visualize the user-item similarity heatmap
    plt.figure(figsize=(12, 8))
    similarity_matrix = t.mm(user_embeds[random_users], item_embeds.t()).detach().cpu().numpy()
    
    # Randomly select a subset of items to avoid an overly large heatmap
    num_items_to_plot = min(50, item_embeds.shape[0])
    random_items = np.random.choice(item_embeds.shape[0], num_items_to_plot, replace=False)
    
    sns.heatmap(similarity_matrix[:, random_items], cmap='YlGnBu')
    plt.title('User-Item Similarity Heatmap')
    plt.xlabel('Item ID')
    plt.ylabel('User ID')
    plt.savefig(output_dir + model_name + '_user_item_similarity.png')
    plt.close()
    
    print(f"Recommendation visualization saved to {output_dir}")

# 3. Model Performance Metrics Visualization
def visualize_performance(model_name):
    print("Generating performance metrics visualization...")
    # Load training history data
    try:
        with open('../History/' + model_name + '.his', 'rb') as fs:
            metrics = pickle.load(fs)
    except Exception as e:
        print(f"Could not load model history: {model_name}")
        return
    
    # Extract performance metrics
    epochs = []
    recall = []
    ndcg = []
    
    for epoch, data in metrics.items():
        if isinstance(epoch, int) and 'Recall' in data and 'NDCG' in data:
            epochs.append(epoch)
            recall.append(data['Recall'])
            ndcg.append(data['NDCG'])
    
    if not epochs:
        print("No performance metrics found")
        return
    
    # Plot the performance metric curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, recall, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall@K')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, ndcg, 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('NDCG')
    plt.title('NDCG@K')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir + model_name + '_performance.png')
    plt.close()
    
    print(f"Performance metrics visualization saved to {output_dir}")

# Execute visualizations
if __name__ == "__main__":
    print(f"Starting visualization for model: {model_name}")
    visualize_embeddings(model_name)
    visualize_recommendations(model_name)
    visualize_performance(model_name)
    print("Visualization complete!")
