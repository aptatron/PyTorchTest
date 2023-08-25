import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def main():

    # setup agnositc device code
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    import requests
    import zipfile
    import pathlib
    from pathlib import Path

    #setup path to a data folder
    DATA_PATH = Path("data")
    image_path = DATA_PATH / "images"

    # if the image folder does not exist, download the data
    if image_path.exists():
        print(f"{image_path} directory already exist ... skipping download")    
    else:
        print(f"{image_path} directory does not exist ... creating directory")
        image_path.mkdir(parents=True, exist_ok=True)

    # download the data
    with open(image_path / "pizza_steak_sushi.zip", "wb") as f:
        r = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading zip file ...")
        f.write(r.content)

    # unzip the file
    with zipfile.ZipFile(image_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping file ...")
        zip_ref.extractall(image_path)   

    # become one with the data
    import os
    def walk_through_dir(dir_path):
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # visualize random images
    import random
    from PIL import Image

    random.seed(42)

    image_path_list = list(image_path.glob("*/*/*.jpg"))

    random_image_path = random.choice(image_path_list)
    print(random_image_path)

    image_class = random_image_path.parent.stem
    print(image_class)

    # open image
    im = Image.open(random_image_path)

    import numpy as np
    import matplotlib.pyplot as plt

    img_as_arr = np.array(im)
    # plt.figure(figsize=(10, 7))
    # plt.imshow(img_as_arr)
    # plt.title(f"{image_class} image with shape {img_as_arr.shape}")
    # plt.axis(False);

    # transform data with torchvision.transforms

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    data_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    # plot randome images from the all images

    def plot_transformed_images (image_paths, transform, n=3, seed=42):
        
        if seed:                    
            random.seed(seed)           
        random_image_paths = random.sample(image_paths, n)
        for image_path in random_image_paths:
            with Image.open(image_path) as f:
                fig, ax = plt.subplots(nrows= 1, ncols=2, figsize=(10, 7))
                ax[0].imshow(f)
                ax[0].set_title(f"Original image shape {np.array(f).shape}")
                ax[0].axis(False)

                ax[1].imshow(transform(f).permute(1, 2, 0))
                ax[1].set_title(f"Transformed image shape {np.array(transform(f)).shape}")
                ax[1].axis(False)
                
                fig.suptitle(f"class: {image_path.parent.stem}", fontsize=16)
                plt.show()

    # plot_transformed_images(image_path_list, data_transform, 3, 42)

    # use imageFolder to create a dataset

    train_data = datasets.ImageFolder(root=train_dir, transform=data_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

    # print (os.cpu_count())

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_data, batch_size=10, num_workers=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=10, num_workers=1, shuffle=False)

    # load iamge data with a custom dataset

    target_directory = train_dir
    class_names = os.listdir(target_directory)
    print(class_names)

    class_names_b = sorted ([entry.name for entry in os.scandir(target_directory) if entry.is_dir()])
    print(class_names_b)

    def find_classes(directory: str) -> tuple[list[str], dict[str, int]]:
        classes = sorted ([entry.name for entry in os.scandir(directory) if entry.is_dir()])

        if not classes:
            raise FileNotFoundError(f"Could not find any class folders in {directory}.")
        
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    # create a dataset to replicate the ImageFolder dataset

    from torch.utils.data import Dataset

    class ImageFolderCustom(Dataset):
        def __int__(self, 
                    targ_dir: str,
                    transform:  None):
            self.paths = list(pathlib.path(targ_dir).glob("*/*.jpg"))
            self.transform = transform
            self.classes, self.class_to_idx = find_classes(targ_dir)

        def load_image(self, index: int) -> Image.Image:
            image_path = self.paths[index]
            return Image.open(image_path)
        
        # overwrite __len__

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
            image = self.load_image(index)
            class_name = self.paths[index].parent.stem
            class_idx = self.class_to_idx[class_name]
            if self.transform:
                image = self.transform(image)
            return image, class_idx

    ## Data augmentation

    from torchvision import datasets, transforms

    train_transform = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                            transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.ToTensor()])

    image_path_list = list(image_path.glob("*/*/*.jpg"))

    # plot random transofrmed images

    # plot_transformed_images(image_path_list, train_transform, 3, 42)

    # new model TinyVGG without data augmentation
    # creating transforms and loading data for Model0

    simple_transform = transforms.Compose([transforms.Resize((64, 64)),
                                            transforms.ToTensor()])

    # load and transform data
    from torchvision import datasets
    train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
    test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

    # turn datasets into dataloaders

    import os
    from torch.utils.data import DataLoader

    #setup batch size and number of workers
    BATCH_SIZE = 32 # number of samples processed before the model is updated 
    NUM_WORKERS = 1

    # create dataloaders
    train_dataloader_simple = DataLoader(train_data_simple, 
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

    test_dataloader_simple = DataLoader(test_data_simple,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

    #create TinyVGG model

    class TinyVGG(nn.Module):
        def __init__(self, input_shape, hidden_units, output_shape) -> None:
            super(TinyVGG, self).__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                # nn.Linear(hidden_units * 16 * 16, hidden_units),
                # nn.ReLU(),
                nn.Linear(hidden_units * 13 * 13, output_shape)
            )

        
        def forward(self, x):
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = self.classifier(x)

            return x

    torch.manual_seed(42)

    # setup model
    model0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=3)

    ## use torchinfo to get information about the model

    import torchinfo
    torchinfo.summary(model0, input_size=(1, 3, 64, 64))

    # create train step and test step functions

    def train_step(model, dataloader, loss_fn, optimizer, device):
        model.train()

        train_loss, train_acc = 0.0, 0.0

        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            y_hat = model(X)
        
            loss = loss_fn(y_hat, y)
            train_loss += loss.item()

            y_hat = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
            train_acc += torch.sum(y_hat == y).item()/len(y_hat)

        

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss /= len(dataloader)
        train_acc /= len(dataloader)

        return train_loss, train_acc

    def test_step(model, dataloader, loss_fn, device):
        model.eval()

        test_loss, test_acc = 0.0, 0.0

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                
                y_hat = model(X)
                
                loss = loss_fn(y_hat, y)
                test_loss += loss.item()

                y_hat = torch.argmax(torch.softmax(y_hat, dim=1), dim=1)
                test_acc += torch.sum(y_hat == y).item()/len(y_hat)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

        return test_loss, test_acc

    # create a train function to combine train and test step functions

    from tqdm.auto import tqdm

    def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, device, epochs):
        history = dict(train_loss=[], 
                        train_acc=[], 
                        test_loss=[], 
                        test_acc=[])

        for epoch in tqdm(range(epochs)):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
            print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
            print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
            print("-"*80)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

        return model, history

    # setup train and evluation model0

    torch.manual_seed(42)

    NUM_EPOCHS = 5

    model0 = TinyVGG(input_shape=3, hidden_units=10, output_shape=3)

    model0 = model0.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model0.parameters(), lr=1e-3)

    from timeit import default_timer as timer
    start_time = timer()

    # model0, history = train(model0, train_dataloader_simple, test_dataloader_simple, loss_fn, optimizer, device, NUM_EPOCHS)

    end_time = timer()

    print(f"Training time: {end_time - start_time}")

    # # plot loss and accuracy curves
    # def plot_loss_curves(history):
    #     train_loss = history["train_loss"]
    #     test_loss = history["test_loss"]

    #     train_acc = history["train_acc"]
    #     test_acc = history["test_acc"]

    #     epochs = range(len(history["train_loss"]))

    #     plt.figure(figsize=(10, 7))
    #     plt.plot(epochs, train_loss, label="Train loss")
    #     plt.plot(epochs, test_loss, label="Test loss")
    #     plt.title("Loss")
    #     plt.xlabel("Epochs")
    #     plt.legend()
    #     plt.show()

    #     plt.figure(figsize=(10, 7))
    #     plt.plot(epochs, train_acc, label="Train acc")
    #     plt.plot(epochs, test_acc, label="Test acc")
    #     plt.title("Accuracy")
    #     plt.xlabel("Epochs")
    #     plt.legend()
    #     plt.show()

    # plot_loss_curves(history)

    # create traininig transform with TrivialAugmentWide

    from torchvision import transforms
    train_transform_trvivial = transforms.Compose([transforms.Resize((64,64)),
                                                    transforms.TrivialAugmentWide(num_magnitude_bins=31),
                                                    transforms.ToTensor()])
    
    test_transform_simple = transforms.Compose([transforms.Resize((64,64)),
                                                transforms.ToTensor()])
    
    train_data_augmented = datasets.ImageFolder(root=train_dir, transform=train_transform_trvivial)
    test_data_augmented = datasets.ImageFolder(root=test_dir, transform=test_transform_simple)

    import os
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()

    torch.manual_seed(42)
    train_dataloader_augmented = DataLoader(train_data_augmented,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)
    
    test_dataloader_augmented = DataLoader(test_data_augmented,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=NUM_WORKERS)
    
    # create model1 with data augmentation

    torch.manual_seed(42)
    model_1 = TinyVGG(input_shape=3, hidden_units=10, output_shape=3)
    model_1 = model_1.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_1.parameters(), lr=1e-3)

    from timeit import default_timer as timer
    start_time = timer()

    # model_1, history_1 = train(model_1, train_dataloader_augmented, test_dataloader_augmented, loss_fn, optimizer, device, NUM_EPOCHS)

    end_time = timer()

    print(f"Training time: {end_time - start_time}")

    # plot loss and accuracy curves
    # plot_loss_curves(history_1)

    # compare results

    # import pandas as pd

    # history_df = pd.DataFrame({"train_loss": history["train_loss"],
    #                             "train_acc": history["train_acc"],
    #                             "test_loss": history["test_loss"],
    #                             "test_acc": history["test_acc"]})
    
    # history_df_augmented = pd.DataFrame({"train_loss": history_1["train_loss"],
    #                                     "train_acc": history_1["train_acc"],
    #                                     "test_loss": history_1["test_loss"],
    #                                     "test_acc": history_1["test_acc"]})
    
    # # plot train loss in differnt models
    # plt.figure(figsize=(10, 7))
    # plt.plot(history_df["train_loss"], label="Train loss")
    # plt.plot(history_df_augmented["train_loss"], label="Train loss augmented")
    # plt.title("Train loss")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.show()

    # # plot test loss in differnt models
    # plt.figure(figsize=(10, 7))
    # plt.plot(history_df["test_loss"], label="Test loss")
    # plt.plot(history_df_augmented["test_loss"], label="Test loss augmented")
    # plt.title("Test loss")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.show()

    # # plot train accuracy in differnt models
    # plt.figure(figsize=(10, 7))
    # plt.plot(history_df["train_acc"], label="Train acc")
    # plt.plot(history_df_augmented["train_acc"], label="Train acc augmented")
    # plt.title("Train accuracy")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.show()

    # # plot test accuracy in differnt models
    # plt.figure(figsize=(10, 7))
    # plt.plot(history_df["test_acc"], label="Test acc")
    # plt.plot(history_df_augmented["test_acc"], label="Test acc augmented")
    # plt.title("Test accuracy")
    # plt.xlabel("Epochs")
    # plt.legend()
    # plt.show()

    # make a prediction on a custom image

    import requests
    from PIL import Image

    custom_image_path = DATA_PATH / "04-pizza-dad.jpeg"

    # download the image if it does not exist
    if not custom_image_path.exists():
        r = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/blob/main/data/04-pizza-dad.jpeg?raw=true")
        with open(custom_image_path, "wb") as f:
            print("Downloading custom image ...")
            f.write(r.content)
    else:
        print("Custom image already exists ... skipping download.")

    # load the image

    import torchvision
    custom_image_unit8 = torchvision.io.read_image(str(custom_image_path))/255.0
    custom_image = Image.open(custom_image_path)
    
    # making a prediction on a custom image with a trained PyTorch model

    custom_image_tensor = transforms.Compose([transforms.Resize((64, 64)),
                                                transforms.ToTensor()])(custom_image)


    model_1.eval()
    with torch.no_grad():
        y_pred = model_1(custom_image_tensor.unsqueeze(0).to(device))
        y_pred = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        print(f"Predicted class: {class_names[y_pred]}")








if __name__ == "__main__":
    main()
