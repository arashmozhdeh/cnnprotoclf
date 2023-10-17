import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from gensim.models import Word2Vec
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import time
from contextlib import contextmanager
import warnings
from sklearn.exceptions import UndefinedMetricWarning

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_size', type=int, default=21, help="Word2Vec embedding size")
parser.add_argument('--w2v_epochs', type=int, default=10, help="Word2Vec number of epochs")
parser.add_argument('--fc_hidden_dim', type=int, default=1024, help="Hidden dimension size of the fully connected layer")
parser.add_argument('--dropout', type=float, default=0.6, help="Dropout probability")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'], help="Device to use: cuda or cpu")
parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use. Set to 0 to use CPU")
parser.add_argument('--epochs', type=int, default=400, help='number of epochs (default: 10)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size (default: 256)')
parser.add_argument('--shot_size', type=int, default=None, help='shot size for few-shot learning (default: None)')
parser.add_argument('--train_data_path', type=str, default="./dataset/train_dataset.csv", help="Path to the training dataset")
parser.add_argument('--val_data_path', type=str, default="./dataset/valid_dataset.csv", help="Path to the validation dataset")
parser.add_argument('--test_data_path', type=str, default="./dataset/test_dataset.csv", help="Path to the test dataset")
args = parser.parse_args()

logger = TensorBoardLogger("logs", name="protein_family")
logger.log_hyperparams(vars(args))  # Log the hyperparameters

# Initialize constants and dictionary
quant_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

def quantize_sequence(sequence):
    return [quant_dict[aa] for aa in sequence if aa in quant_dict]

def to_three_grams(sequence):
    return [sequence[i:i+3] for i in range(0, len(sequence), 3)]

train_dataset_path = "./dataset/train_dataset.csv"
valid_dataset_path = "./dataset/valid_dataset.csv"
test_dataset_path = "./dataset/test_dataset.csv"

train_dataset = pd.read_csv(args.train_data_path)[['Protein families', 'Sequence']]
valid_dataset = pd.read_csv(args.val_data_path)[['Protein families', 'Sequence']]
test_dataset = pd.read_csv(args.test_data_path)[['Protein families', 'Sequence']]

train_dataset.columns = ['Label', 'Seq']
valid_dataset.columns = ['Label', 'Seq']
test_dataset.columns = ['Label', 'Seq']

def preprocess_data(dataset):
    dataset.drop_duplicates(subset='Seq').dropna(subset=['Seq', 'Label'])
    dataset['QuantizedSeq'] = dataset['Seq'].apply(quantize_sequence)
    dataset['ThreeGrams'] = dataset['Seq'].apply(to_three_grams)
    return dataset

# Preprocess datasets
train_dataset = preprocess_data(train_dataset)
valid_dataset = preprocess_data(valid_dataset)
test_dataset = preprocess_data(test_dataset)

# Combine 'ThreeGrams' for Word2Vec
all_three_grams = train_dataset['ThreeGrams'].tolist() + valid_dataset['ThreeGrams'].tolist()

# Word2Vec embedding
embedding_size = args.embedding_size
w2v_model = Word2Vec(vector_size=embedding_size, window=3, min_count=1, workers=16, alpha=0.025, min_alpha=0.0001)
w2v_model.build_vocab(all_three_grams)
w2v_model.train(all_three_grams, total_examples=len(all_three_grams), epochs=10)

def sequence_to_embedding(seq):
    embeddings = [w2v_model.wv[three_gram] for three_gram in seq if three_gram in w2v_model.wv]
    return embeddings

def normalize_embeddings(embeddings):
    max_val = max([max(e) for e in embeddings])
    min_val = min([min(e) for e in embeddings])
    return [(e - min_val) / (max_val - min_val) for e in embeddings]

def get_normalized_embedding(seq):
    embeddings = sequence_to_embedding(seq)
    return normalize_embeddings(embeddings)

# Directly create 'NormalizedEmbeddings' without intermediate 'Embeddings'
train_dataset['NormalizedEmbeddings'] = train_dataset['ThreeGrams'].apply(get_normalized_embedding)
valid_dataset['NormalizedEmbeddings'] = valid_dataset['ThreeGrams'].apply(get_normalized_embedding)
test_dataset['NormalizedEmbeddings'] = test_dataset['ThreeGrams'].apply(get_normalized_embedding)

# Drop the 'ThreeGrams' column as it's no longer needed
train_dataset.drop(columns=['ThreeGrams'], inplace=True)
valid_dataset.drop(columns=['ThreeGrams'], inplace=True)
test_dataset.drop(columns=['ThreeGrams'], inplace=True)

num_classes = train_dataset['Label'].nunique()
unique_labels = train_dataset['Label'].unique().tolist()
label_map = {label: idx for idx, label in enumerate(unique_labels)}

# train_dataset['MappedLabel'] = train_dataset['Label'].map(label_map)
# valid_dataset['MappedLabel'] = valid_dataset['Label'].map(label_map)
# test_dataset['MappedLabel'] = test_dataset['Label'].map(label_map)

class ProteinDataset(Dataset):
    def __init__(self, dataframe, max_length=None):
        self.data = dataframe
        self.max_length = max_length or max(dataframe['NormalizedEmbeddings'].apply(len))
        
    def random_sample(self, shot_size):
        """Randomly sample a few shots."""
        # Check if shot_size is provided, else skip random sampling
        if shot_size:
            self.data = self.data.sample(n=shot_size)

class ProteinFamilyClassifier(pl.LightningModule):
    def __init__(self, criterion):
        super(ProteinFamilyClassifier, self).__init__()
        self.conv1 = nn.Conv1d(21, 512, kernel_size=3)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=5)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7)
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=3, padding=(3-1)//2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2, args.fc_hidden_dim)  
        self.fc2 = nn.Linear(args.fc_hidden_dim, args.fc_hidden_dim)
        self.fc3 = nn.Linear(args.fc_hidden_dim, num_classes)  # num_classes: number of protein families

        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        self.criterion = criterion

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 3, stride=3)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 3, stride=3)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool1d(x, 3, stride=3)
        x = F.relu(self.conv6(x))
        x = F.max_pool1d(x, 3, stride=3)
        x = F.relu(self.conv7(x))

        x = x.view(x.size(0), -1)  # Flatten

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
    	with timer(f'Training iteration {batch_idx}'):
            inputs, labels = batch
            outputs = self(inputs)
            loss = F.cross_entropy(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            # Compute metrics
            correct = (predicted == labels).float().sum()
            accuracy = correct / len(labels)

            return {
                "loss": loss,
                "pred": predicted,
                "true": labels,
                "accuracy": accuracy
            }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        all_predictions = torch.cat([x["pred"] for x in outputs]).cpu().numpy()
        all_true_labels = torch.cat([x["true"] for x in outputs]).cpu().numpy()
        # Compute average accuracy over the epoch
        avg_accuracy = torch.stack([x["accuracy"] for x in outputs]).mean().cpu().numpy()
        # Compute other metrics if needed (for demonstration, we'll compute f1_score)
        f1 = f1_score(all_true_labels, all_predictions, average='macro')
        logs = {
            "train_loss": avg_loss,
            "train_acc": torch.tensor(avg_accuracy),
            "train_f1": torch.tensor(f1)
        }
        return {"loss": avg_loss, "log": logs}


    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        return {"val_loss": loss, "pred": predicted, "true": labels}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        all_predictions = torch.cat([x["pred"] for x in outputs]).cpu().numpy()
        all_true_labels = torch.cat([x["true"] for x in outputs]).cpu().numpy()
        # Compute other metrics
        accuracy = accuracy_score(all_true_labels, all_predictions)
        f1 = f1_score(all_true_labels, all_predictions, average='macro')
        recall = recall_score(all_true_labels, all_predictions, average='macro')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            precision = precision_score(all_true_labels, all_predictions, average='macro')

        logs = {
            "val_loss": avg_loss,
            "val_acc": torch.tensor(accuracy), 
            "val_f1": torch.tensor(f1),
            "val_recall": torch.tensor(recall),
            "val_precision": torch.tensor(precision)
        }
        return {"val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        return {
            "test_loss": loss,
            "pred": predicted,
            "true": labels
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        all_predictions = torch.cat([x["pred"] for x in outputs]).cpu().numpy()
        all_true_labels = torch.cat([x["true"] for x in outputs]).cpu().numpy()
        # Compute other metrics
        accuracy = accuracy_score(all_true_labels, all_predictions)
        f1 = f1_score(all_true_labels, all_predictions, average='macro')
        recall = recall_score(all_true_labels, all_predictions, average='macro')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            precision = precision_score(all_true_labels, all_predictions, average='macro')

        logs = {
            "test_loss": avg_loss,
            "test_acc": torch.tensor(accuracy), 
            "test_f1": torch.tensor(f1),
            "test_recall": torch.tensor(recall),
            "test_precision": torch.tensor(precision)
        }
        return {"test_loss": avg_loss, "log": logs}

class ProteinDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, batch_size=256, shot_size=None, num_workers=4):
        super(ProteinDataModule, self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.shot_size = shot_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if self.shot_size:
            self.train_dataset = ProteinDataset(self.train_df)
            self.train_dataset.random_sample(self.shot_size)
            self.val_dataset = ProteinDataset(self.val_df)
            self.val_dataset.random_sample(self.shot_size)
        else:
            self.train_dataset = ProteinDataset(self.train_df)
            self.val_dataset = ProteinDataset(self.val_df)
        self.test_dataset = ProteinDataset(self.test_df)

    def train_dataloader(self):
        train_dataset = ProteinDataset(self.train_df)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

# Main execution

device = torch.device(args.device)
criterion = nn.CrossEntropyLoss()

# Initialize DataModule and Model
data_module = ProteinDataModule(train_dataset, valid_dataset, test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shot_size=args.shot_size)
model = ProteinFamilyClassifier(criterion)

# Train
logger = TensorBoardLogger("tb_logs", name="protein_classifier")
trainer = pl.Trainer(max_epochs=num_epochs, gpus=args.num_gpus if args.device == 'cuda' else 0, logger=logger)
trainer.fit(model, data_module)

# Test
trainer.test()
