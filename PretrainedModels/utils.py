import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_accuracy(model,val_dataloader):
    acc = 0
    total_samples = 0

    # Set model to evaluation mode
    model.eval()

    # Move model to CUDA if available
    if torch.cuda.is_available():
        model.to(device)

    # Disable gradient computation
    with torch.no_grad():
        for images, labels in val_dataloader:
            # Move data to CUDA if available
            if torch.cuda.is_available():
                images = images.to(device)
                labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate accuracy
            predictions = torch.argmax(outputs, dim=1)
            acc += torch.sum(predictions == labels).item()

            # Count total samples
            total_samples += labels.size(0)

            if(total_samples>200):
                break

    # Calculate overall accuracy
    overall_accuracy = acc / total_samples
    return overall_accuracy


if(__name__=="__main__"):
    from imagenet import ModelOptions, get_pre_trained_model
    from imagenet_data import imagenette2_train_val
    from torch.utils.data import DataLoader


    model, transform = get_pre_trained_model(ModelOptions.RESNET18)

    train_dataset, val_dataset = imagenette2_train_val(transform)
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=True)

    print("Overall accuracy:", model_accuracy(model,val_dataloader))
