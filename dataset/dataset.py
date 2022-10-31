import random

from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os


class GANData(Dataset):
    def __init__(self, args, train=True):
        self.file_path = args.dataset
        dir_name = 'train' if train else 'test'
        self.imageA_path = os.path.join(self.file_path, dir_name + 'A')
        self.imageB_path = os.path.join(self.file_path, dir_name + 'B')

        self.imageA_list = os.listdir(self.imageA_path)
        self.imageB_list = os.listdir(self.imageB_path)

    def read_data(self, index_a, index_b):
        img_a = Image.open(os.path.join(self.imageA_path, self.imageA_list[index_a]))
        img_a = img_a.convert('RGB')
        img_a = T.Compose([T.ToTensor(), T.Resize(256)])(img_a)
        img_b = Image.open(os.path.join(self.imageB_path, self.imageB_list[index_b]))
        img_b = img_b.convert('RGB')
        img_b = T.Compose([T.ToTensor(), T.Resize(256)])(img_b)
        return img_a, img_b

    def __len__(self):
        return len(self.imageA_list)

    def __getitem__(self, index: int):
        try:
            img_a, img_b = self.read_data(index, index)
        except Exception as e:
            # open random image
            print('Cannot find paired B for A, use random B instead')
            img_a, img_b = self.read_data(index, random.randint(0, len(self.imageB_list)))
        return img_a, img_b


if __name__ == '__main__':
    from config import load_config
    from torch.utils.data import random_split, DataLoader
    import matplotlib.pyplot as plt
    args = load_config()
    args.dataset = '../data/apple2orange'
    print(args)

    dataset = GANData(args)
    train_dataset, validate_dataset = random_split(dataset,
                                                   [l := round(len(dataset) * (1 - args.test_ratio)), len(dataset) - l])
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=validate_dataset, batch_size=args.batch_size, shuffle=True)

    img_a, img_b = (next(iter(train_loader)))
    img = T.ToPILImage()(img_a[0])
    plt.imshow(img)
    plt.show()



