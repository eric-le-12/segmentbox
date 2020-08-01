from torchvision import transforms
import albumentations as A

# define augmentation methods for training and validation/test set

train_transform = A.Compose({
        A.Resize(120, 120),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(),
        A.ElasticTransform(p=0.2)
})

val_transform = A.Compose({
    A.Resize(120, 120)
})
