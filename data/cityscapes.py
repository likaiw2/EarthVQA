import os
import glob
import numpy as np
import logging
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from collections import OrderedDict
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, OneOf, Compose
import ever as er
from ever.interface import ConfigurableMixin
from data import distributed
# import distributed
import cityscapesscripts.helpers.labels as CSLabels

logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict([
    ('road', (128, 64, 128)), 
    ('sidewalk', (244, 35, 232)), 
    ('building', (70, 70, 70)),
    ('wall', (102, 102, 156)), 
    ('fence', (190, 153, 153)), 
    ('pole', (153, 153, 153)),
    ('traffic light', (250, 170, 30)), 
    ('traffic sign', (220, 220, 0)),
    ('vegetation', (107, 142, 35)), 
    ('terrain', (152, 251, 152)), 
    ('sky', (70, 130, 180)),
    ('person', (220, 20, 60)), 
    ('rider', (255, 0, 0)), 
    ('car', (0, 0, 142)),
    ('truck', (0, 0, 70)), 
    ('bus', (0, 60, 100)), 
    ('train', (0, 80, 100)),
    ('motorcycle', (0, 0, 230)), 
    ('bicycle', (119, 11, 32))
])

class CityscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_paths = []
        self.mask_paths = []
        self.transforms = transforms

        self._load_paths(image_dir, mask_dir)

    def convert_label_ids_to_train_ids(self, mask):
        mask_copy = mask.copy()
        for label in CSLabels.labels:
            mask_copy[mask == label.id] = label.trainId
            
        # Set the ignored labels to 255
        mask_copy[mask_copy == 255] = 0
        
        return mask_copy

    def _load_paths(self, image_dir, mask_dir):
        img_glob = os.path.join(image_dir, '**', '*_leftImg8bit.png')
        self.image_paths = sorted(glob.glob(img_glob, recursive=True))
        logger.info('Found %d images in Cityscapes dataset.', len(self.image_paths))

        # if mask_dir is not None and mask are in the same directory
        if mask_dir is not None:
            self.mask_paths = [
                os.path.join(mask_dir, os.path.relpath(p, image_dir).replace('_leftImg8bit.png', '_gtFine_labelIds.png'))
                for p in self.image_paths
                ]
        else:
            self.mask_paths = []

    def __getitem__(self, idx):
        image = imread(self.image_paths[idx])
        raw_image = image.copy()
        mask = None

        if self.mask_paths:
            mask = imread(self.mask_paths[idx]).astype(np.int64)
            mask = self.convert_label_ids_to_train_ids(mask)
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']
            return image, dict(mask=mask, imagen=os.path.basename(self.image_paths[idx]), raw_image=raw_image)
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image)
                image = blob['image']
            return image, dict(imagen=os.path.basename(self.image_paths[idx]), raw_image=raw_image)

    def __len__(self):
        return len(self.image_paths)


@er.registry.DATALOADER.register()
class CityscapesLoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = CityscapesDataset(self.config.image_dir, self.config.mask_dir, self.config.transforms)

        sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(dataset)

        super(CityscapesLoader, self).__init__(dataset,
                                               self.config.batch_size,
                                               sampler=sampler,
                                               num_workers=self.config.num_workers,
                                               pin_memory=True)

    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))


if __name__ == "__main__":
    dataset = CityscapesDataset(image_dir='/data/likai/cityscapes/leftImg8bit/train', mask_dir='/data/likai/cityscapes/gtFine/train', transforms=None)
    print(len(dataset))
    image, meta = dataset[0]
    print(image.shape)
    print(meta['imagen'])
    print(meta['raw_image'].shape)
    print(meta['mask'].shape)
    print(meta['mask'])
    print(np.unique(meta['mask']))
    print()
    
    datasetloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for image, meta in datasetloader:
        print(image.shape)
        print(meta['imagen'])
        print(meta['raw_image'].shape)
        print(meta['mask'].shape)
        print(meta['mask'])
        print(np.unique(meta['mask']))
        break
    
    
    # for i in CSLabels.labels:
    #     print(i)