
import torchvision.transforms as transforms

from clip.open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_transforms_clip(input_size, min_crop):

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                input_size, scale=(min_crop, 1.0), \
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        ])

    return train_transform


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


def get_transforms_dinov2(input_size, min_crop):

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                input_size, scale=(min_crop, 1.0), \
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            # transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD)
        ])

    return train_transform
