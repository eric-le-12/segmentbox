from torchvision import transforms
import torch
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import random
import cv2
import numpy as np

# define a data class
class SegmentationDataset:
    def __init__(self, data, data_path, transform, training=True):
        """Define the dataset for classification problems

        Args:
            data ([dataframe]): [a dataframe that contain 2 columns: image name and label]
            data_path ([str]): [path/to/folder that contains image file]
            transform : [augmentation methods and transformation of images]
            training (bool, optional): []. Defaults to True.
        """
        self.data = data
        self.imgs = data["file_name"].unique().tolist()
        self.data_path = data_path
        self.training = training
        self.transform = transform

    def __getitem__(self, idx):

        seed = random.randint(1,10000)
        img = cv2.imread(os.path.join(self.data_path, self.data.iloc[idx, 0]))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.data_path, self.data.iloc[idx, 1]),cv2.IMREAD_UNCHANGED)
        
        mask[mask==128]=0
        mask[mask!=0]= 1

        if self.transform is not None:
            transformed = self.transform(image = img,mask = mask)

        image = transformed['image']
        mask = transformed['mask']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)

        return torch.tensor(image, dtype=torch.float), torch.tensor(mask, dtype=torch.float)

    def __len__(self):
        return len(self.imgs)


def make_loader(dataset, train_batch_size, validation_split=0.2):
    """make dataloader for pytorch training

    Args:
        dataset ([object]): [the dataset object]
        train_batch_size ([int]): [training batch size]
        validation_split (float, optional): [validation ratio]. Defaults to 0.2.

    Returns:
        [type]: [description]
    """
    # number of samples in train and test set
    train_len = int(len(dataset) * (1 - validation_split))
    test_len = len(dataset) - train_len
    train_set, test_set = torch.utils.data.random_split(dataset, [train_len, test_len])
    # create train_loader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True,
    )
    # create test_loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,)
    return train_loader, test_loader


def data_split(data, test_size):
    x_train, x_test, y_train, y_test = train_test_split(
        data, data["mask"], test_size=test_size
    )
    return x_train, x_test, y_train, y_test
