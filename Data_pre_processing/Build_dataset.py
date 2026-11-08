import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset

class PCGDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = (bags)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        assert len(bags) == len(labels)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):

        bag = torch.as_tensor(
            self.bags[idx], dtype=torch.float32
        )
        label = self.labels[idx]

        return bag, label

def build_data_loaders(dataset, train_index, test_index):
  training_dataset = Subset(dataset, train_index.tolist())
  valid_dataset = Subset(dataset, test_index.tolist())
  #Building the train loader
  train_loader = DataLoader(
      training_dataset,
      batch_size=1,
      shuffle=False
  )
  #Building the validation loader
  valid_loader = DataLoader(
      valid_dataset,
      batch_size=1,
      shuffle=False
  )

  return train_loader, valid_loader