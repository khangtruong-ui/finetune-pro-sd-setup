# data/dataset.py
import random
from torchvision import transforms
import grain.python as grain
from datasets import load_dataset
import numpy as np
import os
import jax.numpy as jnp
import jax

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class TorchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds) * 10000

    def __getitem__(self, i):
        item = self.ds[i % len(self.ds)]
        output = jax.tree.map(lambda x: np.array(x), item)
        return output

def get_transforms(config):
    return transforms.Compose([
        transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
        transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

def get_dataloader(config, tokenizer):
    if config.dataset_name:
        dataset = load_dataset(config.dataset_name)
    else:
        dataset = load_dataset("imagefolder", data_files={"train": f"{config.train_data_dir}/**"})

    if config.max_train_samples:
        dataset["train"] = dataset["train"].shuffle(seed=config.seed).select(range(config.max_train_samples))

    transform = get_transforms(config)

    def preprocess(examples):
        images = [img.convert("RGB") for img in examples["image"]]
        examples["pixel_values"] = np.array([transform(img) for img in images])
        
        caption_key = random.choice(['raw', 'raw_1', 'raw_2', 'raw_3', 'raw_4'])
        captions = examples[caption_key]
        tokens = tokenizer(captions, padding="max_length", truncation=True, max_length=tokenizer.model_max_length)
        padded_tokens = tokenizer.pad(
            {'input_ids': tokens.input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="np"
        )
        examples["input_ids"] = padded_tokens['input_ids']
        ret = dict(
            input_ids=examples['input_ids'],
            pixel_values=examples['pixel_values']
        )
        return ret
        
    dataset = dataset["train"].with_transform(preprocess)
    
    mapped_ds = TorchDataset(dataset)
    
    torch_sampler = DistributedSampler(
        dataset=mapped_ds,
        num_replicas=jax.process_count(),   # == num_programs
        rank=jax.process_index(),                 # == program_index
        shuffle=True,
    )

    torch_loader = DataLoader(
        mapped_ds, 
        batch_size=config.train_batch_size * jax.local_device_count(),
        sampler=torch_sampler,
        drop_last=True,
        num_workers=os.cpu_count() // 2
     )

    loader_length = len(dataset) // (config.train_batch_size * jax.local_device_count())
    return {'loader': torch_loader, 'length': loader_length}
