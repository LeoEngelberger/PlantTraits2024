import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Config import Config
from PIL import Image
import torch

class DataBuilder:
    def __init__(self):
        self.train_dataframe = None
        self.valid_dataframe = None
        self.test_dataset = None
        self.augmenter = None
        self.train_dataprovider = None
        self.valid_dataset = None
        self.FEATURE_COLS = None
        self.test_dataframe = None
        self.dataframe = None
        self.build()

    def build(self):
        # Train Data
        self.dataframe = pd.read_csv(f'{Config.BASE_PATH}/train.csv')
        self.dataframe['image_path'] = f'{Config.BASE_PATH}/train_images/' + self.dataframe['id'].astype(str) + '.jpeg'
        self.dataframe.loc[:, Config.aux_class_names] = self.dataframe.loc[:, Config.aux_class_names].fillna(-1)

        # Test Data
        self.test_dataframe = pd.read_csv(f'{Config.BASE_PATH}/test.csv')
        self.test_dataframe['image_path'] = f'{Config.BASE_PATH}/test_images/' + self.test_dataframe['id'].astype(
            str) + '.jpeg'
        self.FEATURE_COLS = self.test_dataframe.columns[1:-1].tolist()

        self.fold_dataframe()

        #scaler = StandardScaler()
        scaler = MinMaxScaler()
        sample_dataframe = self.dataframe.copy()

        self.train_dataframe = sample_dataframe[sample_dataframe.fold != Config.fold]
        train_features = scaler.fit_transform(self.train_dataframe[self.FEATURE_COLS].values)
        train_paths = self.train_dataframe.image_path.values
        train_labels = self.train_dataframe[Config.class_names].values
        train_aux_labels = self.train_dataframe[Config.aux_class_names].values
        self.train_dataprovider = self.build_dataset(train_paths, train_features, train_labels, train_aux_labels,
                                                     batch_size=Config.batch_size, shuffle=True, drop_last=True,
                                                     augment=True,
                                                     cache=False)
        self.valid_dataframe = sample_dataframe[sample_dataframe.fold == Config.fold]
        valid_features = scaler.transform(self.valid_dataframe[self.FEATURE_COLS].values)
        valid_paths = self.valid_dataframe.image_path.values
        valid_labels = self.valid_dataframe[Config.class_names].values
        valid_aux_labels = self.valid_dataframe[Config.aux_class_names].values
        self.valid_dataset = self.build_dataset(valid_paths, valid_features, valid_labels, valid_aux_labels,
                                                batch_size=Config.batch_size, shuffle=False,
                                                augment=False,
                                                cache=False)

        test_paths = self.test_dataframe.image_path.values
        test_features = scaler.transform(self.test_dataframe[self.FEATURE_COLS].values)
        self.test_dataset = self.build_dataset(test_paths, test_features, batch_size=Config.batch_size,
                                                shuffle=False, augment=False, cache=False)

    def fold_dataframe(self):
        skf = StratifiedKFold(n_splits=Config.num_folds, shuffle=True, random_state=42)

        # Create a Bin for each trait

        for i, trait in enumerate(Config.class_names):
            bin_edges = np.percentile(self.dataframe[trait], np.linspace(0, 100, Config.num_folds + 1))
            self.dataframe[f"bin_{i}"] = np.digitize(self.dataframe[trait], bin_edges)

        # Create a unified Bin

        self.dataframe["final_bin"] = (
            self.dataframe[[f"bin_{i}" for i in range(len(Config.class_names))]]
            .astype(str)
            .agg("".join, axis=1)
        )

        self.dataframe = self.dataframe.reset_index(drop=True)
        for fold, (train_idx, valid_idx) in enumerate(skf.split(self.dataframe, self.dataframe["final_bin"])):
            self.dataframe.loc[valid_idx, "fold"] = fold

    def check_dataset(self):
        for batch in self.train_dataprovider:
            inpts = batch[0]
            targets = batch[1]
            # Displaying images and targets
            self.display_images_with_targets(inpts, targets)
            break

    def display_images_with_targets(self,images, targets, num_images=8, num_cols=4):
        plt.figure(figsize=(4 * num_cols, num_images // num_cols * 5))
        for i in range(num_images):
            image = images[0][i].permute(1, 2, 0).numpy()
            target = targets[i].numpy()

            image = (image - image.min()) / (image.max() + 1e-4)

            formatted_tar = "\n".join(
                [
                    ", ".join(
                        f"{name.replace('_mean', '')}: {val:.2f}"
                        for name, val in zip(Config.class_names[j: j + 3], target[j: j + 3])
                    )
                    for j in range(0, len(Config.class_names), 3)
                ]
            )

            plt.subplot(num_images // num_cols, num_cols, i + 1)
            plt.imshow(image)
            plt.title(f"[{formatted_tar}]")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    def build_dataset(self, paths, features, labels=None, aux_labels=None, batch_size=32, cache=True,
                      augment=False, shuffle=True, cache_dir="", drop_last=False):
        transform = transforms.Compose([
            transforms.AutoAugment(),
            transforms.RandomResizedCrop(size=Config.image_size, antialias=True),
            transforms.ToTensor(),
        ])

        dataset = CustomDataset(paths, features, labels, aux_labels, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                drop_last=drop_last, pin_memory=True)
        return dataloader
class CustomDataset(Dataset):
    def __init__(self, paths, features, labels=None, aux_labels=None, transform=None):
        self.paths = paths
        self.features = features
        self.labels = labels
        self.aux_labels = aux_labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path)

        features = torch.FloatTensor(self.features[idx])

        if self.transform:
            image = self.transform(image)
        image /= 255.0
        image.float()

        if self.labels is not None:
            label = self.labels[idx]

            aux_label = self.aux_labels[idx] if self.aux_labels is not None else None
            return (image, features),label.astype(np.float32), aux_label.astype(np.float32)
        else:
            return image, features
