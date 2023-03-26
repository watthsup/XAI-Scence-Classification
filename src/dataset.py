from torch.utils.data import Dataset
import cv2
import pandas as pd
from glob import glob
import torch

class SceneDataset(Dataset):
    def __init__(
            self,
            root_dir = '/data/users/6370327221/dataset/HCC-sc/',
            lookup_csv = "/data/users/6370327221/dataset/HCC-sc/lookup.csv",
            is_train = True,
            transform = None
            ):
        if is_train:
            self.im_paths = sorted(glob(root_dir + 'train/*.jpg'))
        elif not is_train:
            self.im_paths = sorted(glob(root_dir + 'test/*.jpg'))
        self.transform = transform
        self.lookup_df = pd.read_csv(lookup_csv)

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        im_path = self.im_paths[index]
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0

        im_name = im_path.split('/')[-1]
        label = self.lookup_df[self.lookup_df['image_name']==im_name]['label'].values[0]
        label = torch.tensor(int(label)).long()

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
            image = image.float()

        return image, label