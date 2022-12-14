import functools
import os
import argparse
from collections import namedtuple

import numpy as np
from sklearn.metrics import top_k_accuracy_score, classification_report

import paddle
from paddle.io import DataLoader, BatchSampler
import paddle.nn.functional as F
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from paddlenlp.utils.log import logger

# yapf: disable
# parser = argparse.ArgumentParser()
# parser.add_argument('--device', default="gpu", help="Select which device to evaluate model, defaults to gpu.")
# parser.add_argument("--dataset_dir", required=True, type=str, help="Local dataset directory should include train.txt, dev.txt and label.txt")
# parser.add_argument("--params_path", default="../checkpoint/", type=str, help="The path to model parameters to be loaded.")
# parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
# parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
# parser.add_argument("--train_file", type=str, default="train.txt", help="Train dataset file name")
# parser.add_argument("--dev_file", type=str, default="dev.txt", help="Dev dataset file name")
# parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
# parser.add_argument("--bad_case_path", type=str, default="./bad_case.txt", help="Bad case saving file path")
# args = parser.parse_args()
# yapf: enable

def get_args(dataset_dir):
    args = namedtuple('args', [
        'device',
        'dataset_dir',
        'params_path',
        'max_seq_length',
        'batch_size',
        'train_file',
        'dev_file',
        'label_file',
        'bad_case_path',
    ])
    args.device = "gpu"
    args.dataset_dir = dataset_dir
    args.params_path = "../checkpoint/"
    args.max_seq_length = 128
    args.batch_size = 32
    args.train_file = "train.txt"
    args.dev_file = "dev.txt"
    args.label_file = "label.txt"
    args.bad_case_path = "./bad_case.txt"
    return args


def preprocess_function(examples, tokenizer, max_seq_length, is_test=False):
    """
    Preprocess dataset
    """
    result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
    if not is_test:
        result["labels"] = np.array([examples['label']], dtype='int64')
    return result


def read_local_dataset(path, label_map):
    """
    Read dataset file
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence, label = line.strip().split('\t')
            yield {'text': sentence, 'label': label_map[label]}


@paddle.no_grad()
def evaluate(args):
    """
    Evaluate the model performance
    """
    paddle.set_device(args.device)
    if os.path.exists(os.path.join(
            args.params_path, "model_state.pdparams")) and os.path.exists(
                os.path.join(args.params_path,
                             "model_config.json")) and os.path.exists(
                                 os.path.join(args.params_path,
                                              "tokenizer_config.json")):
        model = AutoModelForSequenceClassification.from_pretrained(
            args.params_path)
        tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    else:
        raise ValueError("The {} should exist.".format(args.params_path))

    # load and preprocess dataset
    label_path = os.path.join(args.dataset_dir, args.label_file)
    train_path = os.path.join(args.dataset_dir, args.train_file)
    dev_path = os.path.join(args.dataset_dir, args.dev_file)

    label_map = {}
    label_list = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            l = line.strip()
            label_map[l] = i
            label_list.append(l)
    train_ds = load_dataset(read_local_dataset,
                            path=train_path,
                            label_map=label_map,
                            lazy=False)
    dev_ds = load_dataset(read_local_dataset,
                          path=dev_path,
                          label_map=label_map,
                          lazy=False)
    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=args.max_seq_length)
    train_ds = train_ds.map(trans_func)
    dev_ds = dev_ds.map(trans_func)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    train_batch_sampler = BatchSampler(train_ds,
                                       batch_size=args.batch_size,
                                       shuffle=False)
    train_data_loader = DataLoader(dataset=train_ds,
                                   batch_sampler=train_batch_sampler,
                                   collate_fn=collate_fn)
    dev_batch_sampler = BatchSampler(dev_ds,
                                     batch_size=args.batch_size,
                                     shuffle=False)
    dev_data_loader = DataLoader(dataset=dev_ds,
                                 batch_sampler=dev_batch_sampler,
                                 collate_fn=collate_fn)

    model.eval()
    probs = []
    labels = []
    for batch in train_data_loader:
        label = batch.pop("labels")
        logits = model(**batch)
        prob = F.softmax(logits, axis=1)
        labels.extend(label.numpy())
        probs.extend(prob.numpy())
    probs = np.array(probs)
    labels = np.array(labels)
    preds = probs.argmax(axis=-1)
    report_train = classification_report(labels,
                                         preds,
                                         digits=4,
                                         output_dict=True)

    probs = []
    labels = []
    for batch in dev_data_loader:
        label = batch.pop("labels")
        logits = model(**batch)
        prob = F.softmax(logits, axis=1)
        labels.extend(label.numpy())
        probs.extend(prob.numpy())
    probs = np.array(probs)
    labels = np.array(labels)
    preds = probs.argmax(axis=-1)
    report = classification_report(labels, preds, digits=4, output_dict=True)

    logger.info("-----Evaluate model-------")
    logger.info("Train dataset size: {}".format(len(train_ds)))
    logger.info("Dev dataset size: {}".format(len(dev_ds)))
    logger.info("Accuracy in dev dataset: {:.2f}%".format(report['accuracy'] *
                                                          100))
    # logger.info("Top-2 accuracy in dev dataset: {:.2f}%".format(
    #     top_k_accuracy_score(y_true=labels,
    #                          y_score=probs,
    #                          k=2,
    #                          labels=list(range(len(label_list)))) * 100))
    # logger.info("Top-3 accuracy in dev dataset: {:.2f}%".format(
    #     top_k_accuracy_score(y_true=labels,
    #                          y_score=probs,
    #                          k=3,
    #                          labels=list(range(len(label_list)))) * 100))

    for i, l in enumerate(label_list):
        logger.info("Class name: {}".format(l))
        i = str(i)
        if i in report_train:
            logger.info(
                "Evaluation examples in train dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}"
                .format(report_train[i]['support'],
                        100 * report_train[i]['support'] / len(train_ds),
                        report_train[i]['precision'] * 100,
                        report_train[i]['recall'] * 100,
                        report_train[i]['f1-score'] * 100))
        else:
            logger.info("Evaluation examples in train dataset: 0 (0%)")

        if i in report:
            logger.info(
                "Evaluation examples in dev dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}"
                .format(report[i]['support'],
                        100 * report[i]['support'] / len(dev_ds),
                        report[i]['precision'] * 100, report[i]['recall'] * 100,
                        report[i]['f1-score'] * 100))
        else:
            logger.info("Evaluation examples in dev dataset: 0 (0%)")

        logger.info("----------------------------")
    with open(args.bad_case_path, 'w', encoding="utf-8") as f:
        f.write("Confidence\tPrediction\tLabel\tText\n")
        for i, (p, l) in enumerate(zip(preds, labels)):
            p, l = int(p), int(l)
            if p != l:
                f.write("{:.2f}".format(probs[i][p]) + "\t" + label_list[p] +
                        "\t" + label_list[l] + "\t" + dev_ds.data[i]["text"] +
                        "\n")
    f.close()
    logger.info("Bad case in dev dataset saved in {}".format(
        args.bad_case_path))

    return


if __name__ == "__main__":
    evaluate(get_args(''))