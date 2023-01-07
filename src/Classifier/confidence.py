import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, BatchSampler
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR
from paddlenlp.utils.log import logger
from src.Classifier.utils import read_local_dataset
import random
import time
import os

class ConfidenceDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data = list(open(path, 'r'))
        self.len = len(self.data)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        hit = self.data[idx]
        hit = hit.split()
        return torch.Tensor([float(hit[0])]), int(hit[1])

def split_dataset(path, query_set=None, use_db=False, dev_ratio: float = 0.1):
    if use_db:
        origin_dataset = list(read_local_dataset(use_db=use_db, query_set=query_set))
    else:
        origin_dataset = list(open(path, 'r'))
        
    def split(x):
        x = x.strip().split()
        return float(x[0]), int(x[1])
    origin_dataset = list(map(split, origin_dataset))
    origin_len = len(origin_dataset)
    idx = list(range(origin_len))
    random.shuffle(idx)
    train_path = path + '.train'
    dev_path = path + '.dev'
    dev_cnt = int(max(dev_ratio * origin_len, 1))

    with open(train_path, 'w') as train_file, \
        open(dev_path, 'w') as dev_file:
        for i in range(origin_len):
            if idx[i] < dev_cnt:
                dev_file.write('{} {}\n'.format(*tuple(origin_dataset[i])))
            else:
                train_file.write('{} {}\n'.format(*tuple(origin_dataset[i])))

    return ConfidenceDataset(train_path), ConfidenceDataset(dev_path)


def predict(model, data):
    data = torch.Tensor(data)
    if len(data.shape) == 1:
        data = torch.unsqueeze(data, -1)
    logits = model(data)
    return torch.argmax(logits, dim=-1)

def evaluate(model, criterion, dev_dataloader):
    with torch.no_grad():
        corr = 0
        total = 0
        for batch in dev_dataloader:
            x, y = batch
            batch_size = x.shape[0]
            logits = model(x)
            pred = torch.argmax(logits, dim=-1)
            total += batch_size
            corr += torch.sum(pred == y)
        return corr / total

def add_data(data_path, data_list):
    with open(data_path, 'a') as file:
        for data in data_list:
            file.write('{} {}\n'.format(*data))


def get_model(model_path):
    model = nn.Sequential(
        nn.Linear(1, 2),
    )
    if model_path is not None and os.path.isfile(model_path):
        model_state = torch.load(model_path)
        model.load_state_dict(model_state)
    
    return model

def train(
    path,
    data_query_set, 
    model_path: str = None,
    new_model_path: str = None,
    batch_size: int = 32,
    epoch_num: int = 5,
    learning_rate: float = 1e-4,
    warmup_steps: int = 0,
    early_stop_num: int = 3):

    train_ds, dev_ds = split_dataset(path=path, query_set=data_query_set, use_db=True)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_ds, batch_size=batch_size)

    model = get_model(model_path)

    criterion = nn.CrossEntropyLoss()
    
    num_training_steps = len(train_dataloader) * epoch_num
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    lr_scheduler = LinearLR(optimizer)

    global_step = 0
    best_acc = -1
    early_stop_count = 0
    tic_train = time.time()    
    for epoch in range(1, epoch_num + 1):
        if early_stop_count > early_stop_num:
            print('early stop!')
            break

        for step, batch in enumerate(train_dataloader):
            x, y = batch
            batch_size = x.shape[0]
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            acc = torch.sum(torch.argmax(logits, dim=-1) == y) / batch_size
            logger.info(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, acc: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc, 1 /
                    (time.time() - tic_train)))
            tic_train = time.time()

        early_stop_count += 1
        acc = evaluate(model, criterion, dev_dataloader)
        save_best_path = new_model_path
        # if not os.path.exists(save_best_path):
        #     os.makedirs(save_best_path)

        # save models
        if acc > best_acc:
            early_stop_count = 0
            best_acc = acc
            torch.save(model.state_dict(), save_best_path)

        logger.info("Current best accuracy: %.5f" % (best_acc))

    logger.info("Final best accuracy: %.5f" % (best_acc))
    logger.info("Save best accuracy text classification model in %s" %
                (save_best_path))

if __name__ == '__main__':
    train('/root/work/Tsingenda-backend/src/Classifier/dataset/conf.txt', 
        new_model_path='/root/work/Tsingenda-backend/src/Classifier/conf_checkpoint/conf_model_state')
    