import paddlenlp as ppnlp
import paddle, functools, numpy
import os
import time
import paddle.nn.functional as F

# configuration
raw_data_path = os.path.join(os.getcwd(), "../../data/Classifier/raw_data/raw_data.txt")
model_name = "ernie-3.0-medium-zh"
epochs = 5
ckpt_dir = os.path.join(os.getcwd(), "../../data/Classifier/models")
best_acc = 0
best_step = 0
global_step = 0


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']

        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        
    accu = metric.accumulate()
    print("eval loss: %.5f, accuracy: %.5f" % (numpy.mean(losses), accu))
    model.train()
    metric.reset()
    return accu


def train():
    def read_data(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            cnt = 0
            for line in lines:
                text, label = line.strip().split('==')
                yield {"text": text, "label": label, "qid": cnt}
                cnt += 1

    # function to convert data to dataset
    train_dataset, dev_dataset = ppnlp.datasets.load_dataset(
        read_data, data_files = raw_data_path, splits=('train', 'dev'), lazy=False)

    # define model
    # if model does not exist, it will be downloaded automatically
    # else, it will be loaded from local
    if not os.path.exists(ckpt_dir):
        model = ppnlp.transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_classes=2)
    else:
        model = ppnlp.transformers.AutoModelForSequenceClassification.from_pretrained(ckpt_dir)

    # define tokenizer
    # if tokenizer does not exist, it will be downloaded automatically
    # else, it will be loaded from local
    if not os.path.exists(ckpt_dir):
        tokenizer = ppnlp.transformers.AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = ppnlp.transformers.AutoTokenizer.from_pretrained(ckpt_dir)

    # define data collator
    def data_preprocessor(examples, tokenizer, max_seq_length, is_test=False):
        result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
        if not is_test:
            result["labels"] = examples["label"]
        return result

    trans_func = functools.partial(data_preprocessor, tokenizer=tokenizer, max_seq_length=128)
    train_dataset = train_dataset.map(trans_func)
    dev_dataset = dev_dataset.map(trans_func)

    collate_fn = ppnlp.data.DataCollatorWithPadding(tokenizer)

    train_batch_sampler = paddle.io.BatchSampler(train_dataset, batch_size=32, shuffle=True)
    dev_batch_sampler = paddle.io.BatchSampler(dev_dataset, batch_size=64, shuffle=False)
    train_data_loader = paddle.io.DataLoader(dataset=train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    dev_data_loader = paddle.io.DataLoader(dataset=dev_dataset, batch_sampler=dev_batch_sampler, collate_fn=collate_fn)

    optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())
    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    tic_train = time.time()
    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch['input_ids'], batch['token_type_ids'], batch['labels']
            logits = model(input_ids, token_type_ids)
            loss = criterion(logits, labels)
            probs = F.softmax(logits, axis=1)
            correct = metric.compute(probs, labels)
            metric.update(correct)
            acc = metric.accumulate()

            global_step += 1
            if global_step % 10 == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss, acc,
                        10 / (time.time() - tic_train)))
                tic_train = time.time()
            
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            if global_step % 100 == 0:
                save_dir = ckpt_dir
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print(global_step, end=' ')
                acc_eval = evaluate(model, criterion, metric, dev_data_loader)
                if acc_eval > best_acc:
                    best_acc = acc_eval
                    best_step = global_step

                    model.save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)

    print("best acc: %s, best step: %s" % (best_acc, best_step))


def main():
    # function to read data
    train()
    

if __name__ == "__main__":
    main()