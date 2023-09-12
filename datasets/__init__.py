import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm
from pathlib import Path
from PIL import Image
from zipfile import ZipFile
from typing import Union, Dict, Optional
import requests
import contextlib
import os

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            yield


class CelebAHQ256Dataset(Dataset):
    def __init__(self, root:Union[str, Path], download:Optional[bool]=True):
        super().__init__()
        if isinstance(root, Path):
            self.root = root
        elif isinstance(root, str):
            self.root = Path(root)
        else:
            raise NotImplementedError(f"type {type(root)} for root is invalid")

        data_folder = self.root / "celeba_hq_256"
        if download:
            if not data_folder.exists():
                url = "https://api.amadoussangare.com/download?file=celeba_hq_256.zip"
                self.download_data(url)

        self.data_files = list(data_folder.glob("*.jpg"))


    def download_data(self, link:str) -> None:
        res = requests.get(link, stream=True)
        file = self.root / res.headers['Content-Disposition'].split("filename=")[-1].strip('"')
        file.touch()
        total_size = int(res.headers['content-length'])
        chunk_size = 128
        with file.open("wb") as fd:
            for chunk in tqdm(iterable=res.iter_content(chunk_size=chunk_size), total=total_size/chunk_size, unit='kB'):
                fd.write(chunk)

        zipfile = ZipFile(file)
        zipfile.extractall(self.root)
        file.unlink()

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index:int) -> Dict[str, torch.Tensor]:
        file = self.data_files[index]
        img = Image.open(file)
        return {
            'x': torchvision.transforms.functional.to_tensor(img),
            'label': None,
            'label_ids': None
        }
    
class CelebAHQ32Dataset(Dataset):
    def __init__(self, root:Union[str, Path], download:Optional[bool]=True):
        super().__init__()
        if isinstance(root, Path):
            self.root = root
        elif isinstance(root, str):
            self.root = Path(root)
        else:
            raise NotImplementedError(f"type {type(root)} for root is invalid")

        data_folder = self.root / "celeba_hq_32"
        if download:
            if not data_folder.exists():
                url = "https://api.amadoussangare.com/download?file=celeba_hq_32.zip"
                self.download_data(url)

        self.data_files = list(data_folder.glob("*.jpg"))


    def download_data(self, link:str) -> None:
        res = requests.get(link, stream=True)
        file = self.root / res.headers['Content-Disposition'].split("filename=")[-1].strip('"')
        file.touch()
        total_size = int(res.headers['content-length'])
        chunk_size = 128
        with file.open("wb") as fd:
            for chunk in tqdm(iterable=res.iter_content(chunk_size=chunk_size), total=total_size/chunk_size, unit='kB'):
                fd.write(chunk)

        zipfile = ZipFile(file)
        zipfile.extractall(self.root)
        file.unlink()

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index:int) -> Dict[str, torch.Tensor]:
        file = self.data_files[index]
        img = Image.open(file)
        return {
            'x': torchvision.transforms.functional.to_tensor(img),
            'label': None,
            'label_ids': None
        }

class CIFAR10Dataset(Dataset):
    def __init__(self, root:Union[str, Path], train:Optional[bool]=True):
        super().__init__()
        if isinstance(root, Path):
            self.root = root
        elif isinstance(root, str):
            self.root = Path(root)
        else:
            raise NotImplementedError(f"Error: Invalid type for root. Got {type(root)}")

        self.train = train
        # Hide the prints inside de CIFAR10 constructor when download=True
        with suppress_stdout():
            self.dataset = CIFAR10(
                root=self.root,
                train=train,
                download=True
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index:int) -> Dict[str, torch.Tensor]:
        return {
            'x': torchvision.transforms.functional.to_tensor(self.dataset[index][0]),
            'label': None,
            'label_ids': None
        }
    

__all__ = [
    "CelebAHQ256Dataset",
    "CelebAHQ32Dataset",
    "CIFAR10Dataset"
]