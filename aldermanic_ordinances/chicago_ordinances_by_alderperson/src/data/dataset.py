import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    A custom PyTorch Dataset for handling text and label data.

    This class encapsulates a dataset of texts and their corresponding labels,
    making it compatible with PyTorch's DataLoader.
    """
    def __init__(self, texts, labels, tokenizer):
        """
        Initializes the TextDataset object.

        Args:
            texts: A list or array of raw text sequences.
            labels: A list or array of corresponding labels.
            tokenizer: A tokenizer object to convert raw text into numerical format.
        """
        # Store the collection of texts.
        self.texts = texts
        # Store the collection of labels.
        self.labels = labels
        # Store the tokenizer.
        self.tokenizer = tokenizer
        # Find unique class labels and store them
        self.classes = sorted(list(set(labels)))

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        # Return the size of the dataset based on the number of texts.
        return len(self.texts)


    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset at a given index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing the text and label as PyTorch tensors.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
        # Remove the batch dimension from the encoded text tensors.
        encoded_text = {key: val.squeeze(0) for key, val in tokenized_text.items()}
        encoded_text['label'] = torch.tensor(label, dtype=torch.long)

        return encoded_text



