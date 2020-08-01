import torch
import torch.nn as nn
import ssl
import segmentation_models_pytorch as smp


class SegmentationModel:
    def __init__(self, model_name, pretrained="imagenet",class_num=1):
        """Make your model by using transfer learning technique:  
        Using a pretrained model (not including the top layer(s)) as a feature extractor and 
        add on top of that model your custom classifier

        Args:
            model_name ([str]): [name of pretrained model]
            pretrained (str, optional): [using pretrained weight or not]. Defaults to "imagenet".
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.class_num = class_num

    def create_model(self):
        # load your pretrained model
        model = smp.Unet('resnet34', encoder_weights='imagenet',classes=self.class_num, activation='sigmoid')
        return model
