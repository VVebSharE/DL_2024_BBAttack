from enum import Enum, auto

class ModelsOptions(Enum):
    "model dataset type"
    MNIST=auto()
    IMAGENET=auto()

class ModelArchOptions(Enum):
    "model architecture type"
    RESNET=auto()
    VGG=auto()
    ALEXNET=auto()

def get_pre_trained_mnist(arch:ModelArchOptions=ModelArchOptions.VGG):
    "return a minst model"
    return "MNIST model"

def get_pre_trained_imagenet():
    "return an imagenet model"
    return "IMAGENET model"



def get_pre_trained_model(whichModel:ModelsOptions=ModelsOptions.MNIST, architecture:ModelArchOptions=ModelArchOptions.VGG):
    "return a pre-trained model"
    if whichModel==ModelsOptions.MNIST:
        return get_pre_trained_mnist()
    elif whichModel==ModelsOptions.IMAGENET:
        return get_pre_trained_imagenet()
    else:
        raise NotImplementedError()

class MyModel:
    "My model class"
    def __init__(self,which_model=ModelsOptions.MNIST):
        self.model=get_pre_trained_model(which_model)
    
    def check_accuracy(self):
        "print accuracy"
    

