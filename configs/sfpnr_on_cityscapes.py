from configs.lovedav2 import train, test, data, optimizer, learning_rate
from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, RandomBrightnessContrast, Resize, Transpose
import ever as er

classes = 19

config = dict(
    data = dict(
        train=dict(
            type='CityscapesLoader',
            params=dict(
                image_dir='/data/likai/cityscapes/leftImg8bit/train',
                mask_dir='/data/likai/cityscapes/gtFine/train',
                transforms=Compose([
                    # RandomDiscreteScale([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
                    RandomCrop(512, 512),
                    RandomBrightnessContrast(p=0.5),
                    HorizontalFlip(p=0.5),
                    VerticalFlip(p=0.5),
                    RandomRotate90(p=0.5),
                    Normalize(mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                            max_pixel_value=1, always_apply=True),
                    er.preprocess.albu.ToTensor()
                ]),
                CV=dict(k=10, i=-1),
                training=True,
                batch_size=16,
                num_workers=2,
            ),
        ),
        val=dict(
            type='CityscapesLoader',
            params=dict(
                image_dir='/data/likai/cityscapes/leftImg8bit/val',
                mask_dir='/data/likai/cityscapes/gtFine/val',
                transforms=Compose([
                    Normalize(mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                            max_pixel_value=1, always_apply=True),
                    er.preprocess.albu.ToTensor()

                ]),
                CV=dict(k=10, i=-1),
                training=False,
                batch_size=4,
                num_workers=4,
            ),
        ),
        test=dict(
            type='CityscapesLoader',
            params=dict(
                image_dir='/data/likai/cityscapes/leftImg8bit/test',
                mask_dir='/data/likai/cityscapes/gtFine/test',
                transforms=Compose([
                    Normalize(mean=(123.675, 116.28, 103.53),
                            std=(58.395, 57.12, 57.375),
                            max_pixel_value=1, always_apply=True),
                    er.preprocess.albu.ToTensor()

                ]),
                CV=dict(k=10, i=-1),
                training=False,
                batch_size=4,
                num_workers=4,
            ),
        ),
    ),
    model=dict(
        type='SemanticFPN',
        params=dict(
            encoder=dict(
                name='resnet50',
                weights='imagenet',
                in_channels=3,
                output_stride=32,
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                num_groups_gn=None
            ),
            classes=classes,
            loss=dict(
                ignore_index=-1,
            )
        )),
        # data=data,
        optimizer=optimizer,
        learning_rate=learning_rate,
        train=train,
        test=test
    )
