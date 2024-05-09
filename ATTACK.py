import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import os

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

class AdversarialAttack:
    def __init__(self, model, perturbation_min=0.0,perturbation_max=1.0, mini_batch_size=16, fooling_ratio=0.7):
        if(mini_batch_size%2!=0):
            raise ValueError("mini_batch_size should be even")
        



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
        

    def generate_perturbation(self, source_samples, non_source_samples, source_target:int, non_source_target:int):
        # source_target is the label we want mdoel to predict after attack, non_source_target is actual label of non_source_samples

        SsTarget = torch.full((self.mini_batch_size//2,), source_target, dtype=torch.long).to(device)
        SoTarget = torch.full((self.mini_batch_size//2,), non_source_target, dtype=torch.long).to(device)

        p = torch.zeros_like(source_samples[0]).to(device)  # Initialize perturbation vector
        v = torch.zeros_like(p).to(device)  # Initialize velocity vector
        w = torch.zeros_like(p).to(device)  # Initialize squared velocity vector
        t = 0  # Initialize time step


        source_samples=source_samples.to(device)
        non_source_samples= non_source_samples.to(device)

        beta1 = 0.9
        beta2 = 0.999

        while True:
            # Randomly select mini-batches
            Ss = source_samples[torch.randperm(len(source_samples))[:self.mini_batch_size // 2]]
            So = non_source_samples[torch.randperm(len(non_source_samples))[:self.mini_batch_size // 2]]

            print("Ss:",Ss.shape)
            print("So:",So.shape)
            print("Min:",Ss.min(),So.min())
            print("Max:",Ss.max(),So.max())

            Ss.requires_grad = True
            So.requires_grad = True

            print(p.shape)
            print("pmax:",p.max())
            print("pmin:",p.min())
            print("perturbation norm:",torch.norm(p,p=2))
            
            Ss = torch.clamp(Ss - p, self.perturbation_min, self.perturbation_max)
            So = torch.clamp(So - p, self.perturbation_min, self.perturbation_max)


            t += 1

            print("loss",F.cross_entropy(model(Ss), SsTarget))
            SsG=torch.autograd.grad(F.cross_entropy(model(Ss), SsTarget), Ss)[0]
            SoG=torch.autograd.grad(F.cross_entropy(model(So), SoTarget), So)[0]
            # torch.Size([10, 3, 32, 32]) ?

            # expectation of gradiant of loss w.r.t So
            delta = (torch.norm(SsG,p=2,dim=(1,2,3)).mean())/(torch.norm(SoG,p=2,dim=(1,2,3)).mean())


            # Compute average gradient direction
            xi_t = 0.5 * ((SsG.mean(dim=0)) + delta * SoG.mean(dim=0))

            p = p +  xi_t

            # # Update velocity vectors
            # v = beta1 * v + (1 - beta1) * xi_t
            # w = beta2 * w + (1 - beta2) * (xi_t * xi_t)

            # # Compute perturbation update
            # p_update = (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t) * v / (torch.sqrt(w) + 1e-8)

            # # Update perturbation vector
            # p = p + p_update

            # # Clip perturbation vector
            # p = torch.clamp(p, self.perturbation_min, self.perturbation_max)

            # # Apply projection operator
            # p = self.project_perturbation(p)



            # Check fooling ratio
            fooling_ratio = self.compute_fooling_ratio(source_samples,non_source_samples, p, source_target)
            if fooling_ratio >= self.fooling_ratio:
                break
            
            print("fooling ratio: ",fooling_ratio)
            print(torch.cuda.memory_allocated(device)/1e9,"GB")
            print()

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

    def project_perturbation(self, perturbation):
        # Apply projection operator (optional)
        # For example, clip the perturbation to be within a specific range
        return perturbation


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
    # Saving perturbed images

    if(not os.path.exists(images_folder)):
        os.makedirs(images_folder)

    for i in range(n):
        image = toImage(samples[i])
        image.save(os.path.join(images_folder,f"class1_{i}.png"))



if(__name__=="__main__"):
    from PretrainedModels.imagenet import ModelOptions, get_pre_trained_model
    from PretrainedModels.imagenet_data import ImageNetT2Dataset
    from torch.utils.data import DataLoader
    import torch

    model, transform = get_pre_trained_model(ModelOptions.RESNET18)

    train_dir = 'imagenette2/train'

    class1_dataset = ImageNetT2Dataset(train_dir,transform=transform,which_class=0)

    class2_dataset = ImageNetT2Dataset(train_dir,transform=transform,which_class=1)

    print(class1_dataset.__len__())
    print(class2_dataset.__len__())

    n=100
    mini_batch_size=20 #make even no.
    # n samples from each class
    source_samples = torch.stack([class1_dataset[i][0] for i in range(n)])
    non_source_samples = torch.stack([class2_dataset[i][0] for i in range(n)])

    source_target = torch.tensor(class1_dataset[0][1]) #0
    non_source_target =torch.tensor( class2_dataset[0][1]) #217


    attack = AdversarialAttack(model,mini_batch_size=mini_batch_size)

    # target = source_target
    target = torch.tensor(1)
    perturbation = attack.generate_perturbation(source_samples, non_source_samples, target, non_source_target)


    perturbation = perturbation.cpu()

    perturbed_source_samples = source_samples - perturbation




    # Saving perturbed images
    saveImages("perturbed_images",perturbed_source_samples,n)
    #also saving actual images
    saveImages("actual_images",source_samples,n)


    # save perturbation
    image = toImage(perturbation)
    image.save("perturbation.png")

    #save perturbation as tensor
    torch.save(perturbation,"perturbation.pt")



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

 
