import json
from torch.utils.data import Dataset
import os
from PIL import Image
from typing import Literal


class_idx = json.load(open("imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
ImageNett2Labels=['tench', 'English_springer', 'cassette_player', 'chain_saw', 'church', 'French_horn', 'garbage_truck', 'gas_pump', 'golf_ball', 'parachute']
ImageNett2ltoI={}
for l in ImageNett2Labels:
    ImageNett2ltoI[l]=idx2label.index(l)

class ImageNetT2Dataset(Dataset):
    def __init__(self, root_dir, transform=None, which_class:int|None=None, not_class:int|None=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        
        self.image_paths = []
        for i,class_name in enumerate(self.classes):
            if(which_class is not None and i!=which_class):
                continue
            if(not_class is not None and i==not_class):
                continue
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith('.JPEG'):
                    self.image_paths.append(os.path.join(class_dir, filename))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        class_name = os.path.basename(os.path.dirname(img_path))
        # label = self.classes.index(class_name)
        label=cls2label[class_name]
        label=ImageNett2ltoI[label]

        if self.transform:
            image = self.transform(image)
            

        return image, label


from torch.utils.data import DataLoader

def imagenette2_train_val(transform=None):
    root_dir = 'imagenette2'
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')


    return ImageNetT2Dataset(train_dir,transform), ImageNetT2Dataset(val_dir,transform)

