import torch
import torch.nn as nn
import pretrainedmodels as ptm
import ssl


class ClassificationModel:
    def __init__(self, model_name, pretrained="imagenet", class_num=4):
        """Make your model by using transfer learning technique:  
        Using a pretrained model (not including the top layer(s)) as a feature extractor and 
        add on top of that model your custom classifier

        Args:
            model_name ([str]): [name of pretrained model]
            pretrained (str, optional): [using pretrained weight or not]. Defaults to "imagenet".
            class_num (int, optional): [number of target classes]. Defaults to 2.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.class_num = class_num

    def classifier(self, in_features):
        # initilize your classifier here
        classifier = nn.Sequential(
            nn.Linear(in_features, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, self.class_num, bias=True),
        )

        # output should be a sequential instance
        self.cls = classifier

    def create_model(self):
        # load your pretrained model
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b4') 
        
        # incoming features to the classifier
        in_features = model._fc.in_features
        # create classfier
        self.classifier(in_features)
        # replace the last linear layer with your custom classifier
        model._fc = self.cls
        model._avg_pooling = nn.AdaptiveMaxPool2d(1)
        # select with layers to unfreeze
        for param in model.parameters():
            param.requires_grad = True
        return model
