import torch
import numpy as np
import torch.utils.data


class BooksDataset(torch.utils.data.Dataset):
    """
    Goodreads dataset for pytorch models.
    """

    def __init__(
        self, dataset_path, test=False, random_state=0, num_negative_samples=2
    ):
        """
        Dataset is a npz of numpy arrays, with
        train_x, test_x
        train_y, test_y
        n_user, n_item
        """
        data = np.load(dataset_path)
        train_x = data["train_x"]
        test_x = data["test_x"]
        train_y = data["train_y"]
        test_y = data["test_y"]

        if test:
            self.items = test_x
            self.targets = test_y
        else:
            self.items = train_x
            self.targets = train_y

        self.add_negative_samples(
            random_state=random_state, parity=num_negative_samples
        )

        self.field_dims = torch.from_numpy(np.max(self.items, axis=0) + 1).to(
            dtype=torch.long
        )
        self.items = torch.from_numpy(self.items).to(dtype=torch.long)
        self.targets = torch.from_numpy(self.targets).squeeze()

        self.user_field_idx = torch.from_numpy(np.array((0,))).to(dtype=torch.long)
        self.item_field_idx = torch.from_numpy(np.array((1,))).to(dtype=torch.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.items[idx], self.targets[idx]

    def add_negative_samples(self, random_state, parity=1):
        """
        Preprocess the interactions into positive and negative samples.
        Negative samples do not check for collisions; this relies on the
        sparsity of the data and therefore introduces some noise. Regularization!
        """
        neg_x = np.tile(self.items, (parity, 1))
        np.random.seed(random_state)
        np.random.shuffle(neg_x)
        neg_targets = np.zeros((neg_x.shape[0], 1), dtype=np.float32)
        self.targets = np.vstack([np.ones_like(self.targets), neg_targets])
        self.items = np.vstack([self.items, neg_x])
