import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from Params import args

# Ensure the History directory exists
history_dir = '../History/'
if not os.path.exists(history_dir):
    print("History directory does not exist. Please make sure the model has been trained and saved.")
    exit()

# Allow the user to specify the model name
model_name = input("Enter the model name to visualize (without .his suffix, or press Enter to use args.save_path): ").strip()
if not model_name:
    model_name = args.save_path

# Load the training history data
try:
    with open(history_dir + model_name + '.his', 'rb') as fs:
        metrics = pickle.load(fs)
    print(f"Successfully loaded model history: {model_name}")
    
    # Print detailed information about the data structure for debugging
    print("Detailed data structure:")
    if isinstance(metrics, dict):
        print(f"metrics is a dictionary with {len(metrics)} keys")
        for key in metrics.keys():
            if isinstance(metrics[key], list):
                print(f"  - Key: {key}, Type: list, Length: {len(metrics[key])}")
                if len(metrics[key]) > 0:
                    print(f"    First element type: {type(metrics[key][0])}, Value: {metrics[key][0]}")
                else:
                    print(f"    List is empty")
            else:
                print(f"  - Key: {key}, Type: {type(metrics[key])}")
    else:
        print(f"metrics is not a dictionary, but {type(metrics)}")
except Exception as e:
    print(f"Failed to load model history: {model_name}, Error: {str(e)}")
    print("Available history files:")
    for file in os.listdir(history_dir):
        if file.endswith('.his'):
            print(f"  - {file[:-4]}")
    exit()

# Check data structure and extract training loss
if 'TrainLoss' in metrics and isinstance(metrics['TrainLoss'], list):
    # Use list index as epoch
    epochs = list(range(len(metrics['TrainLoss'])))
    train_losses = metrics['TrainLoss']
    train_pre_losses = metrics.get('TrainpreLoss', [None] * len(epochs))

    # Ensure all lists have the same length
    min_length = min(len(epochs), len(train_losses), len(train_pre_losses))
    epochs = epochs[:min_length]
    train_losses = train_losses[:min_length]
    train_pre_losses = train_pre_losses[:min_length]

    print(f"Found {len(epochs)} epochs of training data.")

    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Total Training Loss')

    # Plot prediction loss if available
    if train_pre_losses and not all(x is None for x in train_pre_losses):
        plt.plot(epochs, train_pre_losses, 'r-', label='Training Prediction Loss')

    # Plot testing loss if available
    if 'TestLoss' in metrics and isinstance(metrics['TestLoss'], list) and len(metrics['TestLoss']) >= min_length:
        test_losses = metrics['TestLoss'][:min_length]
        plt.plot(epochs, test_losses, 'g--', label='Total Testing Loss')

    if 'TestpreLoss' in metrics and isinstance(metrics['TestpreLoss'], list) and len(metrics['TestpreLoss']) >= min_length:
        test_pre_losses = metrics['TestpreLoss'][:min_length]
        plt.plot(epochs, test_pre_losses, 'm--', label='Testing Prediction Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Process Loss Curve')
    plt.legend()
    plt.grid(True)

    output_dir = '../Visualization/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + model_name + '_loss.png')
    plt.show()

    print(f"Loss curve saved to {output_dir + model_name + '_loss.png'}")

    # Plot all available additional metrics
    metric_keys = [key for key in metrics.keys() if isinstance(metrics[key], list) and 
                   key not in ['TrainLoss', 'TestLoss', 'TrainpreLoss', 'TestpreLoss']]

    if metric_keys:
        print(f"Found the following extra metrics: {metric_keys}")

        for key in metric_keys:
            if len(metrics[key]) == 0:
                print(f"Warning: Metric {key} list is empty")
            else:
                print(f"Metric {key} has {len(metrics[key])} data points")

        # Group into training and testing metrics
        train_metrics = [key for key in metric_keys if key.startswith('Train')]
        test_metrics = [key for key in metric_keys if key.startswith('Test')]

        print(f"Training metrics: {train_metrics}")
        print(f"Testing metrics: {test_metrics}")

        processed_metrics = set()

        # Process paired training and testing metrics
        for key in metric_keys:
            if key in processed_metrics:
                continue

            metric_name = key[5:] if key.startswith('Train') else key[4:]
            train_key = 'Train' + metric_name
            test_key = 'Test' + metric_name

            processed_metrics.add(train_key)
            processed_metrics.add(test_key)

            has_train_data = train_key in metrics and len(metrics[train_key]) > 0
            has_test_data = test_key in metrics and len(metrics[test_key]) > 0

            if not has_train_data and not has_test_data:
                print(f"Metric {metric_name} has no data, skipping plot")
                continue

            print(f"Processing metric: {metric_name}")
            plt.figure(figsize=(10, 6))

            if has_train_data:
                train_data_len = len(metrics[train_key])
                train_epochs = list(range(train_data_len))
                plt.plot(train_epochs, metrics[train_key], 'b-', label=f'Training {metric_name}')
                print(f"  Plotted {train_key} with {train_data_len} points")

            if has_test_data:
                test_data_len = len(metrics[test_key])
                test_epochs = list(range(test_data_len))
                plt.plot(test_epochs, metrics[test_key], 'r--', label=f'Testing {metric_name}')
                print(f"  Plotted {test_key} with {test_data_len} points")

            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plt.title(f'{model_name} {metric_name} Curve')
            plt.legend()
            plt.grid(True)

            plt.savefig(output_dir + model_name + f'_{metric_name}.png')
            plt.show()

            print(f"{metric_name} curve saved to {output_dir + model_name + f'_{metric_name}.png'}")

        # Create a combined plot showing all non-empty metrics
        non_empty_metrics = [key for key in metric_keys if len(metrics[key]) > 0]

        if len(non_empty_metrics) > 1:
            plt.figure(figsize=(12, 8))

            for key in non_empty_metrics:
                data_len = len(metrics[key])
                data_epochs = list(range(data_len))
                label = f"{'Training' if key.startswith('Train') else 'Testing'} {key[5:] if key.startswith('Train') else key[4:]}"
                plt.plot(data_epochs, metrics[key], label=label)

            plt.xlabel('Epoch')
            plt.ylabel('Metrics Value')
            plt.title(f'{model_name} All Metrics')
            plt.legend()
            plt.grid(True)

            plt.savefig(output_dir + model_name + '_all_metrics.png')
            plt.show()

            print(f"All metrics combined plot saved to {output_dir + model_name + '_all_metrics.png'}")

else:
    print("Expected data structure not found. Please check the file format.")
    print("Expected keys like: 'TrainLoss', 'TrainpreLoss', etc.")
    print("Actual keys:", list(metrics.keys()))
