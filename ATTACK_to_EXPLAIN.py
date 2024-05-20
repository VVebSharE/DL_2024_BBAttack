import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from PretrainedModels.imagenet import ModelOptions, get_pre_trained_model
from PretrainedModels.imagenet_data import ImageNetT2Dataset, idx2label
from torch.utils.data import DataLoader

import datetime
current_datetime = datetime.datetime.now()
current_datetime = current_datetime.strftime("%Y-%m-%d__%H:%M:%S")

writer = SummaryWriter(log_dir='runs/experiment1')

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

class AdversarialAttack:
    def __init__(self, model, perturbation_min=0.0,perturbation_max=1.0, mini_batch_size=16, fooling_ratio=0.9, perturbation_norm=20):
        if(mini_batch_size%2!=0):
            raise ValueError("mini_batch_size should be even")
        

        self.perturbation_norm=perturbation_norm

        self.model = model

        # tensor of mini-batch size containing the target labels for each sample
        self.perturbation_max = torch.tensor(perturbation_max).to(device)
        self.perturbation_min = torch.tensor(perturbation_min).to(device)

        self.mini_batch_size = mini_batch_size
        self.fooling_ratio = fooling_ratio

        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters(): 
            param.requires_grad = False
        

    def generate_perturbation(self, seed, non_source_dataset, source_target:int, non_source_target:int,itr=None):
        # source_target is the label we want mdoel to predict after attack, non_source_target is actual label of non_source_samples

        SsTarget = torch.full((self.mini_batch_size//2,), source_target, dtype=torch.long).to(device)
        SoTarget = torch.full((self.mini_batch_size//2,), non_source_target, dtype=torch.long).to(device)

        p = torch.zeros_like(source_samples[0]).to(device)  # Initialize perturbation vector
        v = torch.zeros_like(p).to(device)  # Initialize velocity vector
        w = torch.zeros_like(p).to(device)  # Initialize squared velocity vector
        t = 0  # Initialize time step


        # source_samples=source_samples.to(device)
        # source_samples of one image only
        source_samples = torch.zeros_like(p).unsqueeze(0).to(device)
        non_source_samples= non_source_samples.to(device)

        beta1 = 0.9
        beta2 = 0.999

        lr = 0.01

        while True:
            # Check fooling ratio
            fooling_ratio = self.compute_fooling_ratio(source_samples,non_source_samples, p, source_target)
            if fooling_ratio >= self.fooling_ratio:
                if(itr == None):
                    break
                else:
                    if(t>=itr):
                        break
            
            print("fooling ratio: ",fooling_ratio)
            print(torch.cuda.memory_allocated(device)/1e9,"GB")
            print()



            # Randomly select mini-batches
            Ss = source_samples
            So = non_source_samples[torch.randperm(len(non_source_samples))[:1-self.mini_batch_size]]


            # print("Min:",Ss.min(),So.min())
            # print("Max:",Ss.max(),So.max())

            Ss.requires_grad = True
            So.requires_grad = True

            print(p.shape)
            print("perturbation norm:",torch.norm(p,p=2))
            
            Ss = torch.clamp(Ss - p, self.perturbation_min, self.perturbation_max)
            So = torch.clamp(So - p, self.perturbation_min, self.perturbation_max)


            t += 1

            SsLoss = F.cross_entropy(self.model(Ss), SsTarget)
            SoLoss = F.cross_entropy(self.model(So), SoTarget)

            writer.add_scalars(
                "loss",
                {
                    "Source class": SsLoss,
                    "Non Source class": SoLoss
                },
                t
            )

            SsG = torch.autograd.grad(SsLoss, Ss)[0]
            SoG = torch.autograd.grad(SoLoss, So)[0]

            # torch.Size([10, 3, 32, 32]) ?

            # expectation of gradiant of loss w.r.t So
            delta = (torch.norm(SsG,p=2,dim=(1,2,3)).mean())/(torch.norm(SoG,p=2,dim=(1,2,3)).mean())


            # Compute average gradient direction
            xi_t = 0.5 * ((SsG.mean(dim=0)) + delta * SoG.mean(dim=0))

            # p = p +  xi_t

            # Update velocity vectors
            v = beta1 * v + (1 - beta1) * xi_t
            w = beta2 * w + (1 - beta2) * (xi_t * xi_t)

            # # Compute perturbation update
            p_update = ((1 - beta2 ** t) ** 0.5)/ (1 - beta1 ** t) * v / (torch.sqrt(w) + 1e-8)

            # # infinity norm of p
            p_inf = torch.norm(p_update,p=float('inf'))

            # print(p_inf)

            # # Update perturbation vector
            p = p + lr * p_update/p_inf

            # p = self.lp_ball_projection(p)

            # # Clip perturbation vector
            # p = torch.clamp(p, self.perturbation_min, self.perturbation_max)

            # # Apply projection operator
            # p = self.project_perturbation(p)

            # free memory
            del Ss, So ,
            torch.cuda.empty_cache()

        print("fooling ratio: ",fooling_ratio)
        print("steps",t)
        return p

    def compute_fooling_ratio(self, source_samples,non_source_samples, perturbation, target):
        # percentage of source_smaples predicted as target class instances
        with torch.no_grad():
            source_predictions = torch.argmax(self.model(source_samples - perturbation), dim=1)
            non_source_predictions = torch.argmax(self.model(non_source_samples - perturbation), dim=1)
            target= torch.full((source_predictions.size(0),), target, dtype=torch.long).to(device)
            print("source_predictions:",source_predictions)
            print("non_source_predictions:",non_source_predictions)
            print("correct source:",(source_predictions == target).sum())
            fooling_ratio = (source_predictions == target).sum() / source_predictions.size(0)

            return fooling_ratio

    def lp_ball_projection(self,p):
        # sign(p)*min(abs(p),n)
        return torch.sign(p) * torch.min(torch.abs(p), self.perturbation_norm)

from PIL import Image
def toImage(tensor):
    numpy_array = tensor.cpu().numpy()

    # Rescale values to the range [0, 255]
    min_val = numpy_array.min()
    max_val = numpy_array.max()
    numpy_array = (255.0 * (numpy_array - min_val) / (max_val - min_val)).astype('uint8')

    # Convert NumPy array to PIL image
    image = Image.fromarray(numpy_array.transpose(1, 2, 0))  # Assuming tensor shape is (3, height, width)
    # convert to rgb
    image = image.convert("RGB")

    # Display the image
    return image

def saveImages(images_folder,samples,n=100):

    if(not os.path.exists(images_folder)):
        os.makedirs(images_folder)

    for i in range(n):
        image = toImage(samples[i])
        image.save(os.path.join(images_folder,f"class1_{i}.png"))

def savePerturbation(perturbation ,target=None):
    name = "Perturbations/"+current_datetime 

    if(target is not None):
        name += "__" + idx2label[target]

    image = toImage(perturbation)
    image.save(name+".png")

    #save perturbation as tensor
    torch.save(perturbation,name+".pt")



if(__name__=="__main__"):

    model, transform = get_pre_trained_model(ModelOptions.RESNET18)

    train_dir = 'imagenette2/train'

    class1_dataset = ImageNetT2Dataset(train_dir,transform=transform,which_class=0)

    class2_dataset = ImageNetT2Dataset(train_dir,transform=transform)

    print(class1_dataset.__len__())
    print(class2_dataset.__len__())

    n=10000
    mini_batch_size=20 #make even no.
    # n samples from each class

    seed = torch.stack([class1_dataset[i][0] for i in range(1)])

    non_source_dataset = torch.utils.data.Subset(class2_dataset,range(min(n,class2_dataset.__len__())))

    non_source_dataloader = DataLoader(non_source_dataset, batch_size=mini_batch_size-1, shuffle=True, num_workers=1)

    attack = AdversarialAttack(model,mini_batch_size=mini_batch_size)

    targets=[3,4,8,35,55,62,72,84,134,980,453,502,562,571,574,579,677,679,680,696,738,996,987,975,973]

    # target = source_target
    for t in targets:

        target = torch.tensor(t)

        # for explaination 
        non_source_target = target
        
        perturbation = attack.generate_perturbation(seed, non_source_dataloader, target,itr=1000) 

        perturbation = perturbation.cpu()

        savePerturbation(perturbation,target)


# return a list of image paths predicted as cls
def cls_predicted_images(cls,cls_dataloader,model,limit=100,):
    cls_images=[]

    if(torch.cuda.is_available()):
        model.to(device)

    with torch.no_grad():
        for i, (images, labels, paths) in enumerate(cls_dataloader):
            images = images.to(model.device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for j in range(len(predicted)):
                if predicted[j]==cls:
                    cls_images.append(paths[j])
                    if len(cls_images)==limit:
                        return cls_images
    
    return cls_images

 
