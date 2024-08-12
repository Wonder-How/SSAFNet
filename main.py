from models import CAModel as ConvNet
from models import MLPFeatureExtractor, CombinedModel
from datasets import EEGDataset, EntropyDataset, CombinedDataset
from train_test import train, test
from utils import set_seed,freeze_parameters
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

def main():
    # Seed initialization to ensure reproducibility of results
    set_seed()

    # Setup for Tensorboard logging
    writer = SummaryWriter()

    # Selecting the appropriate computing device (GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model loading and optional pre-trained weights application
    pre_train = True  # Flag to load pre-trained weights
    conv_model = ConvNet(EEG_length=1000)
    if pre_train:
        conv_model.load_state_dict(
            torch.load(r'./SSANet.pth'),
            strict=False)
    conv_model.eval()

    mlp_extractor = MLPFeatureExtractor()
    if pre_train:
        mlp_extractor.load_state_dict(
            torch.load(r'./mlp_weights.pth'), strict=False)
    mlp_extractor.eval()

    # Freeze model parameters to prevent updates during training (if required)
    freeze = False
    if freeze:
        freeze_parameters(conv_model)
        freeze_parameters(mlp_extractor)

    # Data loading and dataset preparation for EEG signals and entropy features
    eeg_csv_paths = [
        r'./origin.csv',
        r'./delta.csv',
        r'./theta.csv',
        r'./alpha.csv',
        r'./beta.csv',
        r'./gamma.csv'
    ]
    eeg_dataset = EEGDataset(eeg_csv_paths, 1000)
    entropy_dataset = EntropyDataset(r'./entropy.csv')

    # Split datasets into training and testing subsets
    total_indices = torch.randperm(len(eeg_dataset))
    train_size = int(0.8 * len(eeg_dataset))
    train_indices, test_indices = total_indices[:train_size], total_indices[train_size:]

    eeg_train_dataset = Subset(eeg_dataset, train_indices)
    entropy_train_dataset = Subset(entropy_dataset, train_indices)
    combined_train_dataset = CombinedDataset(eeg_train_dataset, entropy_train_dataset)

    eeg_test_dataset = Subset(eeg_dataset, test_indices)
    entropy_test_dataset = Subset(entropy_dataset, test_indices)
    combined_test_dataset = CombinedDataset(eeg_test_dataset, entropy_test_dataset)

    # Creating data loaders for batch processing
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=64, shuffle=True)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=64, shuffle=False)

    # Setup the combined model for training
    combined_model = CombinedModel(conv_model, mlp_extractor).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(combined_model.get_classifier_parameters(), lr=0.0002)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)

    # Training and evaluation
    max_accuracy = 0
    best_metrics = {}
    for epoch in range(1, 51):
        train(epoch, combined_model, device, combined_train_loader, optimizer, criterion, scheduler, writer)
        current_accuracy, recall, specificity, precision, f1 = test(combined_model, device, combined_test_loader, criterion, writer, epoch)

        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_metrics = {
                'Accuracy': current_accuracy,
                'Recall': recall,
                'Specificity': specificity,
                'Precision': precision,
                'F1 Score': f1
            }
            # Save the best model
            torch.save(combined_model.state_dict(), 'model_best.pth')

    # Log the best metrics achieved throughout training
    print(f"Best Epoch Metrics: Accuracy: {best_metrics['Accuracy']:.6f}, Recall: {best_metrics['Recall']:.6f}, Specificity: {best_metrics['Specificity']:.6f}, Precision: {best_metrics['Precision']:.6f}, F1 Score: {best_metrics['F1 Score']:.6f}")

if __name__ == "__main__":
    main()
