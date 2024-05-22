import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from enum import Enum
from PretrainedModels.imagenet import ModelOptions, get_pre_trained_model
from PretrainedModels.imagenet_data import ImageNetT2Dataset, idx2label
from torch.utils.data import DataLoader
import torch

writer = SummaryWriter(log_dir='runs/experiment1')

import datetime
current_datetime = datetime.datetime.now()
current_datetime = current_datetime.strftime("%Y-%m-%d__%H:%M:%S")

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

class AdversarialAttack:
    def __init__(self, model, perturbation_min=0.0,perturbation_max=1.0, mini_batch_size=16, fooling_ratio=0.9, perturbation_norm=20):
        if(mini_batch_size%2!=0):
            raise ValueError("mini_batch_size should be even")
        
        self.norm = 15
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
        

    def generate_perturbation(self, source_dataloader, non_source_dataloader, target:int,itr=None,save_pi=None):
        # source_target is the label we want mdoel to predict after attack, non_source_target is actual label of non_source_samples

        SsTarget = torch.full((self.mini_batch_size,), target, dtype=torch.long).to(device)

        samples,labels = next(iter(source_dataloader))


        p = torch.zeros_like(samples[0]).to(device)  # Initialize perturbation vector
        v = torch.zeros_like(p).to(device)  # Initialize velocity vector
        w = torch.zeros_like(p).to(device)  # Initialize squared velocity vector
        t = 0  # Initialize time step

        beta1 = 0.9
        beta2 = 0.999

        lr = 0.01

        while True:

            # Randomly select mini-batches
            Ss=next(iter(source_dataloader))[0].to(device)
            So,SoTarget=next(iter(non_source_dataloader))

            So = So.to(device)
            SoTarget = SoTarget.to(device)


            # print("Min:",Ss.min(),So.min())
            # print("Max:",Ss.max(),So.max())

            Ss.requires_grad = True
            So.requires_grad = True

            print(p.shape)
            print("perturbation norm:",torch.norm(p,p=2))
            
            Ss = torch.clamp(Ss + p, self.perturbation_min, self.perturbation_max)
            So = torch.clamp(So + p, self.perturbation_min, self.perturbation_max)


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
            p_update = ((1 - beta2 ** t) ** 0.5)/ (1 - beta1 ** t) * ( v / (torch.sqrt(w) + 1e-8))

            # # infinity norm of p
            p_inf = torch.norm(p_update,p=float('inf'))

            # print(p_inf)

            # # Update perturbation vector
            p = p - lr * p_update/p_inf

            # p = self.lp_ball_projection(p)

            # Check fooling ratio for source class
            fooling_ratio = self.compute_fooling_ratio(Ss,So, p, target)
            if fooling_ratio >= self.fooling_ratio:
                if(itr == None):
                    break
                else:
                    if(t>=itr):
                        break

            
            print("fooling ratio: ",fooling_ratio)
            print(torch.cuda.memory_allocated(device)/1e9,"GB")
            print()

            if(save_pi is not None):
                # save the p in save_pi dir
                pi=p.cpu()

                saveImages(save_pi,[pi],t)

            # free memory
            del Ss, So ,
            torch.cuda.empty_cache()

        print("fooling ratio: ",fooling_ratio)
        print("steps",t)
        return p

    def compute_fooling_ratio(self, source_samples,non_source_samples, perturbation, target):
        # percentage of source_smaples predicted as target class instances
        with torch.no_grad():
            source_predictions = torch.argmax(self.model(source_samples + perturbation), dim=1)
            non_source_predictions = torch.argmax(self.model(non_source_samples + perturbation), dim=1)
            target= torch.full((source_predictions.size(0),), target, dtype=torch.long).to(device)
            print("source_predictions:",source_predictions)
            print("non_source_predictions:",non_source_predictions)
            print("correct source:",(source_predictions == target).sum())
            fooling_ratio = (source_predictions == target).sum() / source_predictions.size(0)

            return fooling_ratio

    
    def lp_ball_projection(self,p,type="two"):
        if(type=='inf'):
            return torch.sign(p) * torch.min(torch.abs(p), torch.tensor(self.norm))
        elif(type=="two"):
            norm_p = torch.norm(p, p=2)
            scale =  torch.min(torch.tensor(1.0), self.norm / norm_p)
            return p * scale
        else:
            NotImplementedError


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

def saveImages(images_folder,samples,t=""):
    "t is for tag"
    # Saving perturbed images

    if(not os.path.exists(images_folder)):
        os.makedirs(images_folder)

    for i in range(len(samples)):
        image = toImage(samples[i])
        image.save(os.path.join(images_folder,f"{t}class1_{i}.png"))



def attack_on_differnet_models():
   models = [
         ModelOptions.RESNET152,
         ModelOptions.RESNET50,
         ModelOptions.RESNET18,
         ModelOptions.VGG16,
         ModelOptions.VGG19,
         ModelOptions.ALEXNET,
         ModelOptions.vit,
         ModelOptions.vit_l,
   ]

   targets = [270,285,289,322,340,390,441,677]

   for i in range(8):
         model, transform = get_pre_trained_model(models[i])
    
         train_dir = 'imagenette2/train'

         mini_batch_size=20 
         attack = AdversarialAttack(model,mini_batch_size=mini_batch_size,fooling_ratio=1)
    
         class1_dataset = ImageNetT2Dataset(train_dir,transform=transform,which_class = i)
         class2_dataset = ImageNetT2Dataset(train_dir,transform=transform,not_class = i)
    
         n=10000000
    
         class1_dataset = torch.utils.data.Subset(class1_dataset,range(min(n,class1_dataset.__len__())))
         class2_dataset = torch.utils.data.Subset(class2_dataset,range(min(n,class2_dataset.__len__())))
    
    
         source_dataloader = DataLoader(class1_dataset,batch_size=mini_batch_size,shuffle=True,num_workers=1)
         non_source_dataloader = DataLoader(class2_dataset,batch_size=mini_batch_size,shuffle=True,num_workers=1)
    
    
         # target = source_target
         target = torch.tensor(targets[i])
    
         perturbation = attack.generate_perturbation(source_dataloader, non_source_dataloader, target,itr=50,save_pi=f"./Pi_{models[i].__name__}_target={idx2label[targets[i]]}")
    
         perturbation = perturbation.cpu()
    
         # # save perturbation
         image = toImage(perturbation)
         image.save(f"perturbation_{models[i].__name__}_{idx2label[targets[i]]}.png")
         torch.save(perturbation,f"perturbation_{models[i].__name__}_{idx2label[targets[i]]}.pt")



if(__name__=="__main__"):

    attack_on_differnet_models()

    # model, transform = get_pre_trained_model(ModelOptions.RESNET152)

    # train_dir = 'imagenette2/train'

    # class1_dataset = ImageNetT2Dataset(train_dir,transform=transform)
    # class2_dataset = ImageNetT2Dataset(train_dir,transform=transform)

    # n=100

    # class1_dataset = torch.utils.data.Subset(class1_dataset,range(min(n,class1_dataset.__len__())))
    # class2_dataset = torch.utils.data.Subset(class2_dataset,range(min(n,class2_dataset.__len__())))

    # print("class1:",len(class1_dataset))
    # print("class2:",len(class2_dataset))

    # mini_batch_size=20 #make even no.
    # # n samples from each class

    # source_dataloader = DataLoader(class1_dataset,batch_size=mini_batch_size,shuffle=True,num_workers=1)
    # non_source_dataloader = DataLoader(class2_dataset,batch_size=mini_batch_size,shuffle=True,num_workers=1)


    # attack = AdversarialAttack(model,mini_batch_size=mini_batch_size,fooling_ratio=1)

    # # target = source_target
    # target = torch.tensor(94)

    # perturbation = attack.generate_perturbation(source_dataloader, non_source_dataloader, target,itr=150,save_pi="./Pi")

    # perturbation = perturbation.cpu()

    # # perturbed_source_samples = source_samples - perturbation
    # # saveImages("perturbed_images",perturbed_source_samples,n)
    # # saveImages("actual_images",source_samples,n)


    # # # save perturbation
    # image = toImage(perturbation)
    # image.save("perturbation.png")
    # # torch.save(perturbation,"perturbation.pt")