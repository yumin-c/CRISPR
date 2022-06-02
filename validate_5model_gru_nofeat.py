import os
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from scipy import stats
from tqdm import tqdm
from utils import GeneFeatureDataset, seq_concat, select_cols
from model import GeneInteractionModel, BalancedMSELoss

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# LOAD & PREPROCESS GENES

train_file = pd.read_csv('data/DeepPrime_dataset_final_Feat8.csv')
mean = pd.read_csv('data/mean.csv', header=None, index_col=0, squeeze=True)
std = pd.read_csv('data/std.csv', header=None, index_col=0, squeeze=True)
gene_path = 'data/g_final_Feat8.npy'

if not os.path.isfile(gene_path):
    g_train = seq_concat(train_file)
    np.save(gene_path, g_train)
else:
    g_train = np.load(gene_path)


# FEATURE SELECTION

drop_features = ['Edit_len', 'Edit_pos', 'RHA_len', 'type_sub', 'type_ins', 'type_del']

train_features, train_target = select_cols(train_file)
train_fold = train_file.Fold
train_type = train_file.loc[:, ['type_sub', 'type_ins', 'type_del']]

train_features = train_features.drop(drop_features, axis=1)
mean = mean.drop(drop_features)
std = std.drop(drop_features)


# NORMALIZATION

x_train = (train_features - mean) / std
y_train = train_target
y_train = pd.concat([y_train, train_type], axis=1)

g_train = torch.tensor(g_train, dtype=torch.float32, device=device)
x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32, device=device)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32, device=device)


# PARAMS

batch_size = 2048
learning_rate = 5e-3
weight_decay = 5e-2
T_0 = 10
T_mult = 1
hidden_size = 128
n_layers = 1
n_epochs = 10
n_models = 5


# TRAIN & VALIDATION

for fold in range(5):

    train_set = GeneFeatureDataset(g_train, x_train, y_train, str(fold), 'train', train_fold)
    valid_set = GeneFeatureDataset(g_train, x_train, y_train, str(fold), 'valid', train_fold)

    models, preds = [], []
    
    # Train multiple models to ensemble
    for m in range(n_models):

        random_seed = m

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

        model = GeneInteractionModel(hidden_size=hidden_size, num_layers=n_layers, num_features=18).to(device)

        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

        criterion = BalancedMSELoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=learning_rate/100)

        n_iters = len(train_loader)

        for epoch in tqdm(range(n_epochs)):
            train_loss, valid_loss = [], []
            train_count, valid_count = 0, 0

            model.train()

            for i, (g, x, y) in enumerate(train_loader):
                g = g.permute((0, 3, 1, 2))
                y = y.reshape(-1, 4)

                pred = model(g, x)
                loss = criterion(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + i / n_iters)

                train_loss.append(x.size(0) * loss.detach().cpu().numpy())
                train_count += x.size(0)
            
        models.append(model)

    # Ensemble results (bagging)
    for model in models:

        model.eval()
        pred_, y_ = None, None

        with torch.no_grad():
            for i, (g, x, y) in enumerate(valid_loader):
                g = g.permute((0, 3, 1, 2))
                y = y.reshape(-1, 4)

                pred = model(g, x)
                loss = criterion(pred, y)

                valid_loss.append(x.size(0) * loss.detach().cpu().numpy())
                valid_count += x.size(0)

                if pred_ is None:
                    pred_ = pred.detach().cpu().numpy()
                    y_ = y.detach().cpu().numpy()[:, 0]
                else:
                    pred_ = np.concatenate((pred_, pred.detach().cpu().numpy()))
                    y_ = np.concatenate((y_, y.detach().cpu().numpy()[:, 0]))
        
        preds.append(pred_)
    
    preds = np.squeeze(np.array(preds))
    preds = np.mean(preds, axis=0)
    preds = np.exp(preds) - 1

    R = stats.spearmanr(preds, y_).correlation

    print('GRU model with no additional features.')
    print('Fold {} Spearman score: {}'.format(fold, R))


'''
Fold 0 Spearman score: 0.8067747612357775
Fold 1 Spearman score: 0.808031537224078
Fold 2 Spearman score: 0.8014293459236057
Fold 3 Spearman score: 0.8090301192831216
Fold 4 Spearman score: 0.7707354017798824
'''