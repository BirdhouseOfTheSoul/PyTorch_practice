from sklearn.utils import compute_class_weight
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import transformers

from src.data.dataset import TextDataset
from src.engine.trainer import Trainer
#from src.training.losses import get_loss_fn
#from src.training.trainer import Trainer
#from src.utils.seed import seed_everything
from sklearn.model_selection import train_test_split

import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification



def main() -> None:
    #seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aldermanic_ordinances = pd.read_csv("data/Ordinance_Detail-Table.csv").dropna(subset=["Title", "Alderperson"])
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    cat_codes = aldermanic_ordinances["Alderperson"].astype("category").cat.codes
    print(f"cat_codes: {cat_codes.value_counts()}")
    aldermanic_ordinances["Alderperson"] = cat_codes
    
    train_data, val_data, train_labels, val_labels = train_test_split(
        aldermanic_ordinances['Title'].tolist(),
        aldermanic_ordinances['Alderperson'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=aldermanic_ordinances['Alderperson']
    )

    train_ds = TextDataset(train_data, train_labels, tokenizer)
    val_ds = TextDataset(val_data, val_labels, tokenizer)

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=data_collator)

    sample = next(iter(train_loader))
    #print(f"sample input_ids: {sample['input_ids']}")
    #print(f"sample labels: {sample['labels']}")
    #print(f"sample attention_mask: {sample['attention_mask']}")
    #print(f"sample token_type_ids: {sample['token_type_ids']}")

    # Extract all labels from the training set to calculate class weights for handling imbalance.
    train_labels_list = [train_ds.labels[i] for i in range(0, len(train_ds.labels))]
    
    
    # Use scikit-learn's utility to automatically calculate class weights.
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels_list), y=train_labels_list)

    # Convert the NumPy array of weights into a PyTorch tensor of type torch.device
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=aldermanic_ordinances["Alderperson"].nunique(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    print(model)
    for name, param in model.named_parameters():
        if name.startswith("classifier"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    #unfreeze the last 2 layers in the encoder
    for layer in model.bert.encoder.layer[-2:]:
        for name, param in layer.named_parameters():
            param.requires_grad = True

    for name, param in model.named_parameters():
        print(f"param {name} requires_grad: {param.requires_grad}")

    trainer = Trainer()
    trainer.train(model, train_loader, val_loader, loss_function, optimizer, num_epochs=5, device=device)



if __name__ == "__main__":
    main()
