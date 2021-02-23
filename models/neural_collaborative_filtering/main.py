import os
import torch
import tqdm
import boto3 as aws
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import BooksDataset
from NeuralCF import NeuralCollaborativeFiltering

import wandb


def download_data(save_path, file_key, bucket):
    access_key = os.environ.get("AWS_ACCESS_KEY")
    secret_key = os.environ.get("AWS_SECRET")
    s3 = aws.client(
        "s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key
    )
    response = s3.download_file(bucket, file_key, save_path)
    return save_path, response


def get_dataset(name, path, test=False):
    """
    Get the dataset
    :param name: name of the dataset
    :param path: path to the dataset
    :return: MovieLens1M dataset
    """
    if name == "books":
        return BooksDataset(path, test=test)
    else:
        raise ValueError("unknown dataset name: " + name)


def get_model(name, dataset):
    """
    Get the model
    :param name: name of the model
    :param dataset: name of the dataset
    :return: Neural Collaborative Filtering model
    """
    field_dims = dataset.field_dims
    if name == "ncf":
        # Hyperparameters are empirically determined, not optimized
        return NeuralCollaborativeFiltering(
            field_dims,
            embed_dim=16,
            mlp_dims=(16, 16),
            dropout=0.5,
            user_field_idx=dataset.user_field_idx,
            item_field_idx=dataset.item_field_idx,
        )
    else:
        raise ValueError("unknown model name: " + name)


def train(model, optimizer, data_loader, criterion, device, log_interval=1000):
    """
    Train the model
    :param model: choice of model
    :param optimizer: choice of optimizer
    :param data_loader: data loader class
    :param criterion: choice of loss function
    :param device: choice of device
    :return: loss being logged
    """
    # Step into train mode
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(
        tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    ):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Log the total loss for every 1000 runs
        if (i + 1) % log_interval == 0:
            print("    - loss:", total_loss / log_interval)
            total_loss = 0

        wandb.log({"Total Loss": total_loss})


def test(model, data_loader, device):
    """
    Evaluate the model
    :param model: choice of model
    :param data_loader: data loader class
    :param device: choice of device
    :return: AUC score
    """
    # Step into evaluation mode
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    # Return AUC score between predicted ratings and actual ratings
    return roc_auc_score(targets, predicts)


def main(
    dataset_name,
    dataset_path,
    model_name,
    epoch,
    learning_rate,
    batch_size,
    weight_decay,
    device,
    save_dir,
):
    """
    Main function
    :param dataset_name: Choice of the dataset (MovieLens1M)
    :param dataset_path: Directory of the dataset
    :param model_name: Choice of the model
    :param epoch: Number of epochs
    :param learning_rate: Learning rate
    :param batch_size: Batch size
    :param weight_decay: Weight decay
    :param device: CHoice of device
    :param save_dir: Directory of the saved model
    :return: Saved model with logged AUC results
    """
    device = torch.device(device)

    # Get the dataset
    if not os.path.exists(dataset_path):
        download_data("bookrec_data", "explicit_dataset.npz", dataset_path)
    dataset = get_dataset(dataset_name, dataset_path)
    # Split the data into 80% train, 20% validation
    train_length = int(len(dataset) * 0.8)
    valid_length = len(dataset) - train_length
    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length)
    )
    test_dataset = get_dataset(dataset_name, dataset_path, test=True)

    # Instantiate data loader classes for train, validation, and test sets
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

    # Get the model
    model = get_model(model_name, dataset).to(device)
    # Use binary cross entropy loss
    criterion = torch.nn.BCELoss()
    # Use Adam optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Log metrics with Weights and Biases
    wandb.watch(model, log="all")

    # Loop through pre-defined number of epochs
    for epoch_i in range(epoch):
        # Perform training on the train set
        train(model, optimizer, train_data_loader, criterion, device)
        # Perform evaluation on the validation set
        valid_auc = test(model, valid_data_loader, device)
        # Log the epochs and AUC on the validation set
        print("epoch:", epoch_i, "validation: auc:", valid_auc)
        wandb.log({"Validation AUC": valid_auc})

    # Perform evaluation on the test set
    test_auc = test(model, test_data_loader, device)
    # Log the final AUC on the test set
    print("test auc:", test_auc)
    wandb.log({"Test AUC": test_auc})

    # Save the model checkpoint
    torch.save(model.state_dict(), f"{save_dir}/{model_name}.pt")


if __name__ == "__main__":

    wandb.init(project="multi_layer_perceptron_collaborative_filtering")
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="books")
    parser.add_argument(
        "--dataset_path", default="../../data/processed/explicit_dataset.npz"
    )
    parser.add_argument("--model_name", default="ncf")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save_dir", default="chkpt")
    args = parser.parse_args()

    main(
        args.dataset_name,
        args.dataset_path,
        args.model_name,
        args.epoch,
        args.learning_rate,
        args.batch_size,
        args.weight_decay,
        args.device,
        args.save_dir,
    )
