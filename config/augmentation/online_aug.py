import albumentations as A
from albumentations.pytorch import ToTensorV2

def data_transform(size):
    data_transforms = {
        'train': A.Compose([
            A.Resize(size[0], size[1], p=1),
            A.Flip(p=0.75),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=1),
        ],
            additional_targets={'image2': 'image', 'imageLH': 'image', 'imageHL': 'image', 'imageHH': 'image'}
        ),
        'val': A.Compose([
            A.Resize(size[0], size[1], p=1),
        ],
            additional_targets={'image2': 'image', 'imageLH': 'image', 'imageHL': 'image', 'imageHH': 'image'}
        ),
        'test': A.Compose([
            A.Resize(size[0], size[1], p=1),
        ],
            additional_targets={'image2': 'image', 'imageLH': 'image', 'imageHL': 'image', 'imageHH': 'image'}
        )
    }
    return data_transforms

def data_normalize(mean, std):
    data_normalize = A.Compose([
            A.Normalize(mean, std),
            ToTensorV2()
        ],
            additional_targets={'image2': 'image', 'imageLH': 'image', 'imageHL': 'image', 'imageHH': 'image'}
    )
    return data_normalize

def data_transform_aerial_lanenet(size):
    data_transforms = A.Compose([
            A.Resize(size[0], size[1], p=1),
            ToTensorV2()
        ])
    return data_transforms
