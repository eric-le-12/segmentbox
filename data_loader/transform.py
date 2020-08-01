from torchvision import transforms
import albumentations as A

# define augmentation methods for training and validation/test set

train_transform = A.Compose({
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=(-90, 90)),
        A.VerticalFlip(p=0.5),
})

val_transform = A.Compose({
    A.Resize(224, 224)
})
