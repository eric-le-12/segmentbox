# based on torchbox - a fully customizable deep learning implementation solution in Pytorch
implement and fine-tune your deep learning model in an easy, customizable and fastest way 

@Lexuanhieu131297 - First version created by Oct 2019

# changed and useful notes:

1. Custom loss, custom metrics
2. Albumentations for both mask and image
3. Unet, FCN with different backbones


### 1. Custom metric and custom loss for segmentation model

* Change metric and loss name in : cfgs/segment.json

Current metric is implemented in utils.custom_metric class
Current loss is implemented utilizing pywick.losses class

* To add new metric : add another function in utils/custom_metric.py file
* To modify loss function, please change the current loss name into the pywick's loss function name

### 2. Model architectures

* To change the segmentation model : modify models/segmentation.py
* To change the transform refer to the data_loader/segmentation_dataloader.py and transform.py files

