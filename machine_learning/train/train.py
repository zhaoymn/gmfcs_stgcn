
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DataSet_Classification
import math
import os
import sys
# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from models.heads import Classifier
from models.stgcn import STGCN


class STGCN_Classifier(nn.Module):

    def __init__(self,
                 backbone,
                 num_classes=0):
        super(STGCN_Classifier, self).__init__()

        args = backbone.copy()
        args.pop('type')
        self.backbone = STGCN(**args)
        self.cls_head = Classifier(
            num_classes=num_classes, dropout=0.5, latent_dim=512)

    def forward(self, keypoint):
        """Define the computation performed at every call."""
        x = self.backbone(keypoint)
        cls_score = self.cls_head(x)
        return cls_score


batch_size = 128
sample_folder = 'samples dir here'
train_dataset = DataSet_Classification(os.path.join(script_dir, 'train_dataset14.npy'),
                        sample_folder, data_augmentation=False)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = DataSet_Classification(os.path.join(script_dir, 'val_dataset14.npy'),
                      sample_folder, data_augmentation=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

backbone_cfg = {
    'type': 'STGCN',
    'gcn_adaptive': 'init',
    'gcn_with_res': True,
    'tcn_type': 'mstcn',
    'graph_cfg': {
        'layout': 'coco',
        'mode': 'spatial'
    },
    'pretrained': None
}
# head_cfg = {'type': 'GCNHead', 'num_classes': 4, 'in_channels': 256}
model = STGCN_Classifier(backbone=backbone_cfg, num_classes=4)

# model.init_weights()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'


# Load pre-trained weights to the backbone
backbone_state_dict = os.path.join(script_dir, 'j.pth')
# load_checkpoint(model.backbone, backbone_state_dict)
tmp = torch.load(backbone_state_dict)
# print(tmp.keys())

del tmp['cls_head.fc_cls.weight']
del tmp['cls_head.fc_cls.bias']
# print(tmp.keys())
model.load_state_dict(tmp, strict=False)

# print(model)

for param in model.backbone.parameters():
    param.requires_grad = False

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

val_best_acc = -math.inf


num_epochs = 10
for epoch in range(num_epochs):
    if epoch == 3:
        for layer in range(9, 7, -1):
            for param in model.backbone.gcn[layer].parameters():
                param.requires_grad = True
    train_loss = 0.0
    train_correct = 0
    model.train()  # set the model to train mode
    epoch_pbar = tqdm(desc=f"Epoch {epoch+1}/{num_epochs}",
                      total=len(train_dataloader.dataset) / batch_size, position=0)

    epoch_acc = 0.0
    epoch_loss = 0.0

    sample_count = 0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        # print(outputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        sample_count += batch_size

        # Compute the number of correctly classified samples
        _, predicted = torch.max(outputs.data, 1)
        # print(labels, predicted)
        train_correct += (predicted == labels).sum().item()
        # print(train_correct)
        epoch_loss += loss.item()

        epoch_acc = train_correct * 100.0 / sample_count
        tmp_loss = epoch_loss * 1.0 / sample_count * batch_size

        # Update the progress bar for the epoch
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({'loss': tmp_loss, 'acc': epoch_acc})

    # Compute the training loss and accuracy for this epoch
    epoch_loss /= (len(train_dataloader.dataset) / batch_size)
    train_accuracy = 100.0 * train_correct / len(train_dataloader.dataset)
    # Close the progress bar for the epoch
    epoch_pbar.close()
    print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

    # Evaluate the model on the validation set
    val_loss = 0.0
    val_correct = 0
    model.eval()  # set the model to eval mode

    epoch_pbar = tqdm(desc=f"VAL Epoch {epoch+1}/{num_epochs}",
                      total=len(val_dataloader.dataset) / batch_size, position=0)
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute the number of correctly classified samples
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            epoch_pbar.update(1)
    epoch_pbar.close()

    # Compute the validation loss and accuracy for this epoch
    val_loss /= (len(val_dataloader.dataset) / batch_size)
    val_accuracy = 100.0 * val_correct / len(val_dataloader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    if val_accuracy > val_best_acc:
        val_best_acc = val_accuracy
        torch.save(model.state_dict(), os.path.join(script_dir, 'model_weights.pth'))
        print('model saved!')


    # update the learning rate
    scheduler.step()
