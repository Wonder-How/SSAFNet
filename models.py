import torch
import torch.nn as nn

class MFB(nn.Module):
    """Implements Multi-factor Bilinear Pooling."""
    def __init__(self, input_dim, factor_dim):
        super(MFB, self).__init__()
        self.W1 = nn.Parameter(torch.randn(input_dim, factor_dim))
        self.W2 = nn.Parameter(torch.randn(input_dim, factor_dim))

    def forward(self, x1, x2):
        h1 = torch.matmul(x1, self.W1)
        h2 = torch.matmul(x2, self.W2)
        return h1 * h2

class MLPFeatureExtractor(nn.Module):
    """Feature extractor model for MLP based inputs."""
    def __init__(self):
        super(MLPFeatureExtractor, self).__init__()
        self.layer1 = nn.Linear(15, 200)
        self.layer2 = nn.Linear(200, 128)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x

class CombinedModel(nn.Module):
    """Combines convolutional and MLP features using MFB and classifies them."""
    def __init__(self, conv_model, mlp_extractor, factor_dim=128):
        super(CombinedModel, self).__init__()
        self.conv_feature = conv_model
        self.mlp_feature = mlp_extractor
        self.mfb = MFB(128, factor_dim)
        self.classifier = nn.Sequential(
            nn.Linear(factor_dim, 200), nn.ReLU(),
            nn.Linear(200, 100), nn.ReLU(),
            nn.Linear(100, 2), nn.Sigmoid()
        )
        self.mmd_loss = MMDLoss(kernel_bandwidth=100)

    def forward(self, x_conv, x_mlp):
        conv_features = self.conv_feature(x_conv).view(x_conv.size(0), -1)
        mlp_features = self.mlp_feature(x_mlp)
        combined_features = self.mfb(conv_features, mlp_features)
        return self.classifier(combined_features), conv_features, mlp_features

    def get_classifier_parameters(self):
        return self.classifier.parameters()

class MMDLoss(nn.Module):
    """Implements Maximum Mean Discrepancy Loss."""
    def __init__(self, kernel_bandwidth=285.0):
        super(MMDLoss, self).__init__()
        self.kernel_bandwidth = kernel_bandwidth

    def gaussian_kernel(self, source, target):
        n_samples = int(source.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        l2_distance = ((total0 - total1) ** 2).sum(2)
        bandwidth = 2. * self.kernel_bandwidth ** 2
        return torch.exp(-l2_distance / bandwidth)[:n_samples, :n_samples], \
               torch.exp(-l2_distance / bandwidth)[n_samples:, n_samples:], \
               torch.exp(-l2_distance / bandwidth)[:n_samples, n_samples:]

    def forward(self, source, target):
        source_kernel, target_kernel, cross_kernel = self.gaussian_kernel(source, target)
        return source_kernel.mean() + target_kernel.mean() - 2 * cross_kernel.mean()

class SSAM(nn.Module):
    def __init__(self, num_channels, ratio=16):
        super(SSAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(num_channels, num_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(num_channels // ratio, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SSANet(nn.Module):
    def __init__(self,EEG_length):
        super(SSANet, self).__init__()
        self.EEG_length = EEG_length
        self.conv1 = nn.Conv1d(6, 32, 3)
        self.ca1 = SSAM(32)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv1d(32, 64, 3)
        self.ca2 = SSAM(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv1d(64, 128, 3)
        self.ca3 = SSAM(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.conv4 = nn.Conv1d(128, 256, 3)
        self.ca4 = SSAM(256)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)
        self.relu4 = nn.LeakyReLU(inplace=True)

        self.conv5 = nn.Conv1d(256, 512, 3)
        self.ca5 = SSAM(512)
        self.bn5 = nn.BatchNorm1d(512)
        self.pool5 = nn.MaxPool1d(2)
        self.relu5 = nn.LeakyReLU(inplace=True)

        self.fc1 = nn.Linear(44544, 256)  # 根据卷积层后的输出计算输入大小
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1, 3).reshape(batch_size * 3, 6, self.EEG_length)

        x = self.conv1(x)
        x = self.ca1(x) * x
        x = self.relu1(self.pool1(self.bn1(x)))

        x = self.conv2(x)
        x = self.ca2(x) * x
        x = self.relu2(self.pool2(self.bn2(x)))

        x = self.conv3(x)
        x = self.ca3(x) * x
        x = self.relu3(self.pool3(self.bn3(x)))

        x = self.conv4(x)
        x = self.ca4(x) * x
        x = self.relu4(self.pool4(self.bn4(x)))

        x = self.conv5(x)
        x = self.ca5(x) * x
        x = self.relu5(self.pool5(self.bn5(x)))

        x = x.reshape(batch_size, -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        # x = self.fc3(x)
        return x