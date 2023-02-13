from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import pathlib

class NSVD(Dataset):
  def __init__(self, root, train, transform=None):
    super(NSVD, self).__init__()

    self.path = pathlib.Path(root) / 'NSVD' 
    self.df = pd.read_csv(self.path / 'data.csv')
    self.files = list((self.path / ('train' if train else 'test')).glob('*.jpg'))
    self.transform = transform
  
  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    f = self.files[index]
    label = self.df.loc[self.df['filename'] == f.name][['lat', 'lng']].to_numpy()[0]
    
    img = Image.open(f)
    if self.transform:
      img = self.transform(img)
    
    return img, label
