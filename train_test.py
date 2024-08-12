import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def train(epoch, model, device, train_loader, optimizer, criterion, scheduler, writer):
    """Train the model for one epoch."""
    model.train()  # Set the model to training mode
    classification_losses = []  # Store losses for classification
    mmd_losses = []  # Store MMD losses, if applicable

    for batch_idx, ((data_conv, data_mlp), labels) in enumerate(train_loader):
        data_conv = data_conv.to(device)  # Move convolutional data to device (GPU/CPU)
        data_mlp = data_mlp.to(device)  # Move MLP data to device
        labels = labels.to(device)  # Move labels to device

        optimizer.zero_grad()  # Clear gradients before calculating them
        output, conv_features, mlp_features = model(data_conv, data_mlp)  # Perform a forward pass
        classification_loss = criterion(output, labels)  # Calculate classification loss
        mmd_loss = model.mmd_loss(conv_features, mlp_features)  # Calculate MMD loss

        # Combine losses with MMD loss scaled by 1. In our paper, we did not employ MMD loss, so the factor is set to 0.
        total_loss = classification_loss + 1 * mmd_loss
        total_loss.backward()  # Backpropagate the error
        optimizer.step()  # Update model parameters
        scheduler.step()  # Update learning rate schedule

        classification_losses.append(classification_loss.item())  # Log classification loss
        mmd_losses.append(mmd_loss.item())  # Log MMD loss

        # Log loss and training progress every 10 batches
        if batch_idx % 10 == 0:
            writer.add_scalar('Loss/train', total_loss.item(), epoch * len(train_loader) + batch_idx)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data_conv)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {total_loss.item():.6f}')

    return classification_losses, mmd_losses

def test(model, device, test_loader, criterion, writer, epoch):
    """Evaluate the model on the test set."""
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():  # No gradient needed for evaluation
        for ((data_conv, data_mlp), labels) in test_loader:
            data_conv = data_conv.to(device)
            data_mlp = data_mlp.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(data_conv, data_mlp)  # Get model outputs
            test_loss += criterion(outputs, labels).item()  # Sum up batch loss
            pred = outputs.argmax(dim=1, keepdim=False)  # Get the index of the max log-probability
            all_targets.extend(labels.cpu().numpy())  # Store true labels
            all_outputs.extend(pred.cpu().numpy())  # Store predictions

    # Calculate overall test loss and other metrics
    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(all_targets, all_outputs)
    recall = recall_score(all_targets, all_outputs, average='macro')
    precision = precision_score(all_targets, all_outputs, average='macro')
    f1 = f1_score(all_targets, all_outputs, average='macro')
    tn, fp, fn, tp = confusion_matrix(all_targets, all_outputs).ravel()
    specificity = tn / (tn + fp)  # Calculate specificity

    # Log test metrics
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

    # Print test metrics
    print(f'\nTest set: Average loss: {test_loss:.6f}, Accuracy: {accuracy*100:.6f}%, Recall: {recall*100:.6f}%, Specificity: {specificity*100:.6f}%, Precision: {precision*100:.6f}%, F1 Score: {f1*100:.6f}%\n')
    return accuracy, recall, specificity, precision, f1
