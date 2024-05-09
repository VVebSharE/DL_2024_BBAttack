import torch
import torchvision.models as models
import torchvision.transforms as transforms
from enum import Enum
from typing import Tuple

# all major pretrained imagenet models
class ModelOptions(Enum):
    ALEXNET = models.alexnet
    VGG16 = models.vgg16
    VGG19 = models.vgg19
    RESNET18 = models.resnet18
    RESNET34 = models.resnet34
    RESNET50 = models.resnet50
    # GOOGLENET = models.googlenet

class ModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module)->None:
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(x)
        
        return self.model(x)
    
def get_pre_trained_model(model_option: ModelOptions)->Tuple[torch.nn.Module, transforms.Compose]:
    model = model_option(pretrained=True)

    model = ModelWrapper(model)
    model.eval()

    transform=transforms.Compose([  
        transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor()])

    return model, transform


# Usage:
# from imagenet import ModelOptions, get_pre_trained_model
# model, transform = get_pre_trained_model(ModelOptions.RESNET18)
# the model take image (transformed by transform) and give 1000 class probabilities as output