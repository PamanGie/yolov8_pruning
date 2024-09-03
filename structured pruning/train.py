import time
import torch
from ultralytics import YOLO
from prune import StructuredPruner, StandardPruner
import os

# Hyperparameters
learning_rate = 0.001
batch_size = 16
num_epochs = 10
prune_threshold = 0.01

def main():
    # Load YOLO model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO('models/yolov8m.pt').to(device)

    # Initialize structured pruner
    pruner = StructuredPruner(model=model, amount=0.5)

    # Apply pruning to the model with logging
    pruner.apply_pruning()

    # Path to the dataset configuration file
    data_yaml = 'data/data.yaml'

    # Create the directory if it doesn't exist
    os.makedirs('results/models', exist_ok=True)

    # Start training with YOLO's built-in train method
    start_time = time.time()
    model.train(data=data_yaml, epochs=num_epochs, batch=batch_size, imgsz=640)
    total_time = time.time() - start_time

    print(f"Total Training Time: {total_time:.2f} seconds")

    # Save pruned model after training
    model.save('results/models/yolov8_pruned_trained.pt')

if __name__ == '__main__':
    main()
