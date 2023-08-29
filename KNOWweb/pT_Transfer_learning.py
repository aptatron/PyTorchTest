import torch
import torchvision
def main():
    print(torch.__version__)
    print(torchvision.__version__)

    import os
    import zipfile

    from pathlib import Path

    import requests

    data_path = Path("data")
    image_path = data_path / "images"

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    if image_path.is_dir():
        print("Images already downloaded.")
    else:
        image_path.mkdir(parents=True, exist_ok=True)
        print("Downloading images...")
        data_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
        data_file = requests.get(data_url)
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            f.write(data_file.content)

        data_file_path = data_path / "pizza_steak_sushi.zip"

        with zipfile.ZipFile(data_file_path, "r") as zip_ref:
            zip_ref.extractall(image_path)

        os.remove(data_file_path)
        


        if train_dir.is_dir() and test_dir.is_dir():
            print("Train and test directories created.")
        else:
            print("Error creating train and test directories.")

    # import sys
    # sys.path.append("\\going_modular\\going_modular")
    import data_setup, engine



    from torchvision import transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    manual_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.ToTensor(),
                                            normalize])

    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                    test_dir=test_dir,
                                                                                    transform=manual_transforms,
                                                                                    batch_size=32)





    ## setup a pretrained model

    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    auto_transforms = weights.transforms

    # import torchinfo 

    import torchinfo
    from torchinfo import summary

    summary(model, input_size=(1, 3, 224, 224))

    for param in model.features.parameters():
        param.requires_grad = False

    summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=16)

    from torch import nn

    torch.manual_seed(42)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names)))

    # train the model

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # from going_modular.going_modular import engine

    # set the manual seed
    torch.manual_seed(42)

    from timeit import default_timer as timer
    start_time = timer()

    results = engine.train(model=model,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            device="cpu",
                            epochs=5)

    end_time = timer()

    print(f"Time taken to train: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
    

