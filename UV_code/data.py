'''
'''

from torchvision import datasets, transforms
import torch.utils.data as data


class DataLoad:

    data_source = {
        'Art': '../OfficeHome/Art/',
        'Clipart': '../OfficeHome/Clipart/',
        'Product': '../OfficeHome/Product/',
        'Real World': '../OfficeHome/Real_World/'
    }
    
    data_target = {
        'Art': '../OfficeHome/Art/',
        'Clipart': '../OfficeHome/Clipart/',
        'Product': '../OfficeHome/Product/',
        'Real World': '../OfficeHome/Real_World/'
    }
    
    means = {
        'imagenet': [0.485, 0.456, 0.406]
    }
    stds = {
        'imagenet': [0.229, 0.224, 0.225]
    }
    transform = [
        transforms.Scale((256, 256)),
        transforms.Scale(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(means['imagenet'], stds['imagenet']),
    ]

    def __init__(self, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    
    def source_loader(self, case):
        print('[INFO] Loading source datasets: {}'.format(case))
        data_loader = data.DataLoader(
            dataset = datasets.ImageFolder(
                self.data_source[case],
                transform = transforms.Compose(self.transform)
            ),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
        )
        return data_loader


    def target_loader(self, case):
        print('[INFO] Loading target datasets: {}'.format(case))    
        data_loader = data.DataLoader(
            dataset = datasets.ImageFolder(
                self.data_target[case],
                transform = transforms.Compose(self.transform)
            ),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
        )
        return data_loader


    def val_loader(self, case):
        print('[INFO] Loading valid datasets: {}'.format(case))
        data_loader = data.DataLoader(
            dataset=datasets.ImageFolder(
                self.data_target[case],
                transform = transforms.Compose(self.transform)
            ),
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
        )
    
        return data_loader
