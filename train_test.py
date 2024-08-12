import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def train(epoch, model, device, train_loader, optimizer, criterion, scheduler, writer):
    model.train()
    classification_losses = []
    mmd_losses = []

    for batch_idx, ((data_conv, data_mlp), labels) in enumerate(train_loader):
        data_conv = data_conv.to(device)
        data_mlp = data_mlp.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output, conv_features, mlp_features = model(data_conv, data_mlp)
        classification_loss = criterion(output, labels)
        mmd_loss = model.mmd_loss(conv_features, mlp_features)

        total_loss = classification_loss + 1 * mmd_loss 
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        classification_losses.append(classification_loss.item())
        mmd_losses.append(mmd_loss.item())

        if batch_idx % 10 == 0:
            writer.add_scalar('Loss/train', total_loss.item(), epoch * len(train_loader) + batch_idx)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_conv), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss.item()))

    return classification_losses, mmd_losses



def test(model, device, test_loader, criterion, writer, epoch):
    model.eval()
    test_loss = 0
    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for ((data_conv, data_mlp), labels) in test_loader:
            data_conv = data_conv.to(device)
            data_mlp = data_mlp.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(data_conv, data_mlp)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=False)
            all_targets.extend(labels.cpu().numpy())
            all_outputs.extend(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = accuracy_score(all_targets, all_outputs)
    recall = recall_score(all_targets, all_outputs, average='macro')
    precision = precision_score(all_targets, all_outputs, average='macro')
    f1 = f1_score(all_targets, all_outputs, average='macro')
    tn, fp, fn, tp = confusion_matrix(all_targets, all_outputs).ravel()
    specificity = tn / (tn + fp)

    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)

    print('\nTest set: Average loss: {:.6f}, Accuracy: {:.6f}%, Recall: {:.6f}%, Specificity: {:.6f}%, Precision: {:.6f}%, F1 Score: {:.6f}%\n'.format(
        test_loss, accuracy*100, recall*100, specificity*100, precision*100, f1*100))
    return accuracy, recall, specificity, precision, f1
