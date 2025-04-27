import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------- LoveDA Dataset Loader ----------------------
class LoveDADataset(Dataset):
    """
    PyTorch Dataset for LoveDA remote sensing data.

    Directory structure:
        LoveDA/
            Train/  Val/  Test/
                Rural/images_png/*.png
                Urban/images_png/*.png

    Args:
        root_dir (str): Path to LoveDA folder.
        split (str): One of 'Train', 'Val', 'Test'.
        transform (callable, optional): Optional transform to be applied on an image.
    """
    def __init__(self, root_dir, split='Train', transform=None):
        self.split_dir = os.path.join(root_dir, split)
        assert os.path.isdir(self.split_dir), f"Split folder not found: {self.split_dir}"
        self.transform = transform
        self.classes = ['Rural', 'Urban']
        self.class_to_idx = {'Rural': 0, 'Urban': 1}
        self.samples = []  # list of (image_path, label_name)

        for cls in self.classes:
            img_dir = os.path.join(self.split_dir, cls, 'images_png')
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith('.png'):
                    path = os.path.join(img_dir, fname)
                    self.samples.append((path, cls, fname))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_name, img_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx[label_name]
        return image, label_idx, img_path



# ---------------------- Example DataLoader Usage ----------------------
if __name__ == '__main__':
    data_root = '/home/liw324/code/Segment/EarthVQA/dataset/LoveDA'
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    # Create loaders for Train, Val, Test
    train_ds = LoveDADataset(data_root, split='Train', transform=transform)
    # val_ds   = LoveDADataset(data_root, split='Val',   transform=transform)
    # test_ds  = LoveDADataset(data_root, split='Test',  transform=transform)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    # val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=4)
    # test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=4)

    # Iterate example
    for imgs, labels, img_path in train_loader:
        print(imgs.shape)       # e.g., [16, 3, 224, 224]
        print(len(labels))       # e.g., ['Rural', 'Urban', ...]
        print(img_path)         # e.g., ['/path/to/image1.png', '/path/to/image2.png', ...]
        break
