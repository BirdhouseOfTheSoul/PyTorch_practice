import torch

class Trainer():

    def __init__(self):
        pass

    def train(self, model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
        """
         Performs a full training and validation cycle for a PyTorch model.

         Args:
            model: The PyTorch model to be trained.
            train_loader: The DataLoader for the training dataset.
            val_loader: The DataLoader for the validation dataset.
            loss_function: The loss function used for training.
            num_epochs: The total number of epochs to train for.
            device: The computational device ('cuda' or 'cpu') to run on.

        Returns:
            A tuple containing the trained model and a dictionary of the final
            performance metrics from the last validation epoch.
        """
        # Move the model to the specified computational device.
        model.to(device)

        # Determine the number of classes from the model's configuration.
        num_classes = model.config.num_labels


        # Begin the main training and validation loop.
        for epoch in range(0, num_epochs):

            # --- Training Phase ---
            model, train_loss_epoch = self.train_epoch(model, train_loader, loss_function, optimizer, device)
            print(f"epoch {epoch}, train_loss: {train_loss_epoch:.4f}")


            # --- Validation Phase ---
            val_loss_epoch, val_accuracy = self.val_epoch(model, val_loader, loss_function, device)
            print(f"epoch {epoch}, val_loss: {val_loss_epoch:.4f}, val_acc: {val_accuracy:.4f}")


        # Return the trained model and the final validation accuracy.
        return model, {"val_accuracy": val_accuracy}
        


    @staticmethod
    def train_epoch(model, train_loader, loss_function, optimizer, device):

        train_loss_epoch = 0.0
        train_samples = 0
        model.train()
        model = model.to(device)
        # Iterate over the training data batches.
        for _, batch in enumerate(train_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            # Move all tensors to the active device.
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # Clear any gradients from the previous iteration.
            optimizer.zero_grad()

            # Perform a forward pass to get the model's raw outputs (logits).
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calculate the loss for the current batch.
            loss = loss_function(logits, labels)

            # Accumulate the loss and perform backpropagation to compute gradients.
            train_loss_epoch += loss.item()
            train_samples += labels.size(0)
            loss.backward()

            # Update the model's weights based on the computed gradients.
            optimizer.step()

        return model, train_loss_epoch / train_samples



    @staticmethod
    def val_epoch(model, val_loader, loss_function, device):
        # Set the model to evaluation mode, which disables layers like dropout.
        model.eval()
        model = model.to(device)
        # Initialize the accumulated validation loss for the epoch.
        val_loss_epoch = 0
        val_accuracy = 0
        num_samples = 0
        
        # Disable gradient calculations to save memory and computations.
        with torch.no_grad():
            # Iterate over the validation data batches.
            for _, batch in enumerate(val_loader):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                # Move tensors to the active device.
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
        
                # Perform a forward pass to get the model's logits.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        
                # Calculate the validation loss for the current batch.
                val_loss = loss_function(logits, labels)
                val_loss_epoch += val_loss.item()
        
                # Get model predictions and update the metric object with batch results.
                preds = torch.argmax(logits, dim=-1)
                val_accuracy += (preds == labels).sum().item()
                num_samples += labels.size(0)
        
        # Calculate the average validation loss for the epoch.
        val_loss_epoch /= num_samples
        val_accuracy /= num_samples
        return val_loss_epoch, val_accuracy
