import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from tqdm import tqdm
import sys

# Initialize constants and dictionary
quant_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}

def quantize_sequence(sequence):
    return [quant_dict[aa] for aa in sequence if aa in quant_dict]

def to_three_grams(sequence):
    return [sequence[i:i+3] for i in range(0, len(sequence), 3)]

train_dataset_path = "./dataset/train_dataset.csv"
valid_dataset_path = "./dataset/valid_dataset.csv"
test_dataset_path = "./dataset/test_dataset.csv"

train_dataset = pd.read_csv(train_dataset_path)[['Protein families', 'Sequence']]
valid_dataset = pd.read_csv(valid_dataset_path)[['Protein families', 'Sequence']]
test_dataset = pd.read_csv(test_dataset_path)[['Protein families', 'Sequence']]

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
embedding_size = 21
w2v_model = Word2Vec(vector_size=embedding_size, window=5, min_count=1, workers=16, alpha=0.025, min_alpha=0.0001)
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
        # If max_length is not provided, use the maximum sequence length in the dataframe
        self.max_length = max_length or max(dataframe['NormalizedEmbeddings'].apply(len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['NormalizedEmbeddings']
        # Convert sequence to a numpy array first
        sequence_array = np.array(sequence, dtype=np.float32)
        # Convert the numpy array to a tensor
        sequence_tensor = torch.tensor(sequence_array)
        padded_tensor = torch.zeros(self.max_length, sequence_tensor.shape[1])
        padded_tensor[:sequence_tensor.shape[0], :] = sequence_tensor
        label = torch.tensor(self.data.iloc[idx]['Label'], dtype=torch.long)
        return padded_tensor, label

class ProteinFamilyClassifier(nn.Module):
    def __init__(self):
        super(ProteinFamilyClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(21, 512, kernel_size=3)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=5)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7)
        self.conv4 = nn.Conv1d(256, 128, kernel_size=3)
        self.conv5 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=3)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=3, padding=(3-1)//2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2, 1024)  # Adjust this multiplier based on the size after pooling
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)  # num_classes: number of protein families

        # Dropout
        self.dropout = nn.Dropout(0.6)

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

# Create datasets and data loaders
batch_size = 128
train_dataset_obj = ProteinDataset(train_dataset)
valid_dataset_obj = ProteinDataset(valid_dataset)
test_dataset_obj = ProteinDataset(test_dataset)

train_loader = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset_obj, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset_obj, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ProteinFamilyClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

num_epochs = 400  # example

for epoch in range(num_epochs):
    model.train()
    
    # Training Loop without tqdm
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation Loop without tqdm
    model.eval()
    valid_loss = 0.0
    all_predictions = []
    all_true_labels = []

    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions, average='macro')
    recall = recall_score(all_true_labels, all_predictions, average='macro')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        precision = precision_score(all_true_labels, all_predictions, average='macro')

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Training Loss: {train_loss/len(train_loader):.4f}")
    print(f"Validation Loss: {valid_loss/len(valid_loader):.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

    # Test Loop without tqdm
    test_loss = 0.0
    all_test_predictions = []
    all_test_true_labels = []
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_test_predictions.extend(predicted.cpu().numpy())
        all_test_true_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(all_test_true_labels, all_test_predictions)
    test_f1 = f1_score(all_test_true_labels, all_test_predictions, average='macro')
    test_recall = recall_score(all_test_true_labels, all_test_predictions, average='macro')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        test_precision = precision_score(all_test_true_labels, all_test_predictions, average='macro')

    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}")
    print("--------------------------------------------------------------------------------")
    sys.stdout.flush()
