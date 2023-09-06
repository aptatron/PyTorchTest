import torch
import torchvision
import matplotlib.pyplot as plt

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

    auto_transforms = weights.transforms(antialias=True)
    test_transforms = transforms.Compose([transforms.Resize((224,224))])

    # import torchinfo 

    import torchinfo
    from torchinfo import summary

    for param in model.features.parameters():
        param.requires_grad = False

    summary(model, input_size=(1, 3, 224, 224), col_names=["input_size", "output_size", "num_params", "trainable"], col_width=16)

    from torch import nn

    torch.manual_seed(42)

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names)))
    
    summary(model, input_size=(1, 3, 224, 224))
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

    # from helper_functions import plot_loss_curves
    # plot_loss_curves(results)
    # plt.show

    # from typing import List, Tuple

    # from PIL import Image

    # # taken in a trained model
    # def predict_image(model: torch.nn.Module,
    #                   image_path: str,
    #                     class_names: List[str],
    #                     image_size: Tuple[int, int],
    #                     transform: torchvision.transforms = None,
    #                     device: torch.device = "cpu"
    #                     ):
    #     img = Image.open(image_path)
    #     if transform is not None:
    #         image_transformed = transform
    #     else:
    #         image_transformed = transforms.Compose([transforms.Resize((image_size), antialias=True),
    #                                                 transforms.ToTensor(),
    #                                                 normalize]) 
        
    #     model.to(device)

    #     plt.figure()
    #     plt.imshow(img)
    #     plt.show()

    #     model.eval()
    #     with torch.inference_mode():
    #         transformed_img = image_transformed(img)
    #         print ("transformed_img.shape = ", transformed_img.shape)
    #         transformed_img = transformed_img.unsqueeze(0).to(device)
    #         print ("transformed_img.shape.us = ", transformed_img.shape)
    #         target_image_pred = model(transformed_img)
    #         target_image_pred_probs = torch.softmax(target_image_pred, dim=1)   
    #         target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    #         # plot images
    #         plt.figure()
    #         plt.imshow(img)
    #         # plt.title(f"{class_names[target_image_pred_label]}: {target_image_pred_probs[0][target_image_pred_label] * 100:.2f}%")  
    #         plt.axis(False)
    #         plt.show()

    # ## plot random images from dataset
    # import random
    # # import Path
    # from pathlib import Path

    # num_images = 3
    # test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
    # test_image_path_sample = random.sample(test_image_path_list, num_images)

    # print (test_image_path_sample)

    # from helper_functions import pred_and_plot_image

    # # make predictions on random images
    # for image_path in test_image_path_sample:
    #     # do not use plot and predict image function
    #     pred_and_plot_image(model=model,
    #                     image_path=image_path,
    #                     class_names=class_names,
    #                     # image_size=(224, 224),
    #                     transform=test_transforms,
    #                     device="cuda")


        


    
    
    # import data_setup and engine from going_modular, if doesn't work, copy going_modular folder to the same directory
    
    # try:
    #     from going_modular.going_modular import data_setup, engine
    #     print("imported from going_modular")
    # except:
    #     print("imported from google drive")
    #     from google.colab import drive
    #     drive.mount('/drive')
    #     import shutil
    #     shutil.copytree("/drive/MyDrive/Colab Notebooks/going_modular", "/content/going_modular")

    #     from going_modular.going_modular import data_setup, engine
            
    # import plot loss curve from helper_fuctions, if doesn't work, download helper_functions.py and put it in the same directory
    # try:
    #     from helper_functions import plot_loss_curves
    #     print("imported from helper_functions")
    # except:
    #     print("imported from github")
    #     import requests
    #     data_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/helper_functions.py"
    #     data_file = requests.get(data_url)
    #     with open("helper_functions.py", "wb") as f:
    #         f.write(data_file.content)

    #     from helper_functions import plot_loss_curves  

if __name__ == "__main__":
    main()


