"""Torch dataset wrapper for paired RAW/RGB with CFA-safe augmentation."""
from data.dataset import PairedRawRGBDataset

class LearnedISPDataset(PairedRawRGBDataset):
    def __init__(self,*args,noise_sigma=.015,**kwargs):
        super().__init__(*args,augment=kwargs.pop("augment",True),noise_sigma=noise_sigma,**kwargs)
