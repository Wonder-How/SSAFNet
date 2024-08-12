import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, csv_paths,EEG_length,stack_method="frequency"):
        self.data_list = [pd.read_csv(path, header=0) for path in csv_paths]
        for i in range(1, len(self.data_list)):
            if not np.array_equal(self.data_list[0].iloc[:, 2].values, self.data_list[i].iloc[:, 2].values):
                raise ValueError("All files must have consistent labels.")
        self.labels = torch.tensor(self.data_list[0].iloc[:, 2].values.astype(int), dtype=torch.long)
        if stack_method == "frequency":
            eeg_data = [torch.tensor(data.iloc[:, 3:].values.astype(float), dtype=torch.float32).reshape(-1, 3, EEG_length) for data in self.data_list]
        elif stack_method == "channel":
            eeg_data = [torch.tensor(data.iloc[:, 3:].values.astype(float), dtype=torch.float32).reshape(-1, EEG_length, 3) for data in self.data_list]
        self.eeg_data = torch.stack(eeg_data, dim=2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx]


class EntropyDataset(Dataset):
    def __init__(self, csv_path,feature_length=15):
        data = pd.read_csv(csv_path)
        self.labels = torch.tensor(data.iloc[:, 0].values.astype(int), dtype=torch.long)
        self.features = torch.tensor(data.iloc[:, 1:feature_length+1].values.astype(float), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class CombinedDataset(Dataset):
    def __init__(self, eeg_dataset, entropy_dataset):
        assert len(eeg_dataset) == len(entropy_dataset), "Datasets must have the same length"
        self.eeg_dataset = eeg_dataset
        self.entropy_dataset = entropy_dataset

    def __len__(self):
        return len(self.eeg_dataset)

    def __getitem__(self, idx):
        eeg_data, eeg_label = self.eeg_dataset[idx]
        entropy_data, entropy_label = self.entropy_dataset[idx]

        # 检查标签确保它们是匹配的
        assert eeg_label == entropy_label, "Labels do not match between datasets"

        return (eeg_data, entropy_data), eeg_label