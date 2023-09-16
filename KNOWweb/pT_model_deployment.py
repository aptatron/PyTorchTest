# For this notebook to run with updated APIs, we need torch 1.12+ and torchvision 0.13+

import torch
import torchvision



# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary


# Try to import the going_modular directory, download it from GitHub if it doesn't work

# from going_modular.going_modular import data_setup, engine
import going_modular.going_modular.data_setup
import engine
from helper_functions import download_data, set_seeds, plot_loss_curves

def main():
        
    device = "cpu" 
    print(device)

    # Download pizza, steak, sushi images from GitHub
    data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                        destination="pizza_steak_sushi_20_percent")

    print(data_20_percent_path)

    # Setup directory paths to train and test images
    train_dir = data_20_percent_path / "train"
    test_dir = data_20_percent_path / "test"

    print("1")

    # 1. Setup pretrained EffNetB2 weights
    effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    print("2")
    print(effnetb2_weights)

    # 2. Get EffNetB2 transforms
    effnetb2_transforms = effnetb2_weights.transforms()
    print("3")
    print(effnetb2_transforms)

    # 3. Setup pretrained model
    effnetb2 = torchvision.models.efficientnet_b2(weights="DEFAULT") # could also use weights="DEFAULT"
    # effnetb2 = torch.load("pT_training/PyTorchTest/KNOWweb/efficientnet_b2_rwightman-bcdf34b7.pth")
    print("4")
    # print(effnetb2)
    # 4. Freeze the base layers in the model (this will freeze all layers to begin with)
    for param in effnetb2.parameters():
        param.requires_grad = False

    from torchinfo import summary
    summary(effnetb2, input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20, row_settings=["var_names"])

    print(effnetb2.classifier)

    # 5. Update the classifier head
    effnetb2.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), # keep dropout layer same
        nn.Linear(in_features=1408, # keep in_features same 
                out_features=3)) # change out_features to suit our number of classe

    def create_effnetb2_model(num_classes:int=3, 
                            seed:int=42):
        """Creates an EfficientNetB2 feature extractor model and transforms.

        Args:
            num_classes (int, optional): number of classes in the classifier head. 
                Defaults to 3.
            seed (int, optional): random seed value. Defaults to 42.

        Returns:
            model (torch.nn.Module): EffNetB2 feature extractor model. 
            transforms (torchvision.transforms): EffNetB2 image transforms.
        """
        # 1, 2, 3. Create EffNetB2 pretrained weights, transforms and model
        weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
        transforms = weights.transforms()
        model = torchvision.models.efficientnet_b2(weights=weights)

        # 4. Freeze all layers in base model
        for param in model.parameters():
            param.requires_grad = False

        # 5. Change classifier head with random seed for reproducibility
        torch.manual_seed(seed)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1408, out_features=num_classes),
        )
        
        return model, transforms

    effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3,
                                                        seed=42)

    # Setup DataLoaders
    from going_modular.going_modular import data_setup
    train_dataloader_effnetb2, test_dataloader_effnetb2, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                    test_dir=test_dir,
                                                                                                    transform=effnetb2_transforms,
                                                                                                    batch_size=32)

    from going_modular.going_modular import engine

    # Setup optimizer
    optimizer = torch.optim.Adam(params=effnetb2.parameters(),
                                lr=1e-3)
    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Set seeds for reproducibility and train the model
    set_seeds()
    # effnetb2_results = engine.train(model=effnetb2,
                                    # train_dataloader=train_dataloader_effnetb2,
                                    # test_dataloader=test_dataloader_effnetb2,
                                    # epochs=10,
                                    # optimizer=optimizer,
                                    # loss_fn=loss_fn,
                                    # device=device)
    
    # torch.save(effnetb2, "effnetb2.pth")

    vit = torchvision.models.vit_b_16()
    print(vit.heads)

    def create_vit_model(num_classes:int=3, 
                     seed:int=42):
        """Creates a ViT-B/16 feature extractor model and transforms.

        Args:
            num_classes (int, optional): number of target classes. Defaults to 3.
            seed (int, optional): random seed value for output layer. Defaults to 42.

        Returns:
            model (torch.nn.Module): ViT-B/16 feature extractor model. 
            transforms (torchvision.transforms): ViT-B/16 image transforms.
        """
        # Create ViT_B_16 pretrained weights, transforms and model
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        transforms = weights.transforms()
        model = torchvision.models.vit_b_16(weights=weights)

        # Freeze all layers in model
        for param in model.parameters():
            param.requires_grad = False

        # Change classifier head to suit our needs (this will be trainable)
        torch.manual_seed(seed)
        model.heads = nn.Sequential(nn.Linear(in_features=768, # keep this the same as original model
                                            out_features=num_classes)) # update to reflect target number of classes
        
        return model, transforms
    
    vit, vit_transforms = create_vit_model(num_classes=3,
                                       seed=42)

    from torchinfo import summary

    # Print ViT feature extractor model summary (uncomment for full output)
    summary(vit, 
            input_size=(1, 3, 224, 224),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    from going_modular.going_modular import data_setup
    train_dataloader_vit, test_dataloader_vit, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                            test_dir=test_dir,
                                                                                            transform=vit_transforms,
                                                                                            batch_size=32)
    from going_modular.going_modular import engine

    # Setup optimizer
    optimizer = torch.optim.Adam(params=vit.parameters(),
                                lr=1e-3)
    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train ViT model with seeds set for reproducibility
    set_seeds()
    vit_results = engine.train(model=vit,
                            train_dataloader=train_dataloader_vit,
                            test_dataloader=test_dataloader_vit,
                            epochs=10,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            device=device)


    # torch.save(vit, "vit.pth")

if __name__ == "__main__":
    main()
