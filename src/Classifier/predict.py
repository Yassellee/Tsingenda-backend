import os
import argparse
import functools
import numpy as np

import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
import time, pprint

from utils import preprocess_function, read_local_dataset

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--dataset_dir", default="dataset", type=str, help="Local dataset directory should include data.txt and label.txt")
parser.add_argument("--output_file", default="output.txt", type=str, help="Save prediction result")
parser.add_argument("--params_path", default="./checkpoint/", type=str, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--data_file", type=str, default="data.txt", help="Unlabeled data file name")
parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
args = parser.parse_args()
# yapf: enable


@paddle.no_grad()
def predict():
    """
    Predicts the data labels.
    """
    paddle.set_device(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
    tokenizer = AutoTokenizer.from_pretrained(args.params_path)

    label_list = []
    label_path = os.path.join(args.dataset_dir, args.label_file)
    with open(label_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            label_list.append(line.strip())

    data_ds = load_dataset(read_local_dataset,
                           path=os.path.join(args.dataset_dir, args.data_file),
                           is_test=True,
                           lazy=False)

    trans_func = functools.partial(preprocess_function,
                                   tokenizer=tokenizer,
                                   max_seq_length=args.max_seq_length,
                                   is_test=True)

    data_ds = data_ds.map(trans_func)

    # batchify dataset
    collate_fn = DataCollatorWithPadding(tokenizer)
    data_batch_sampler = BatchSampler(data_ds,
                                      batch_size=args.batch_size,
                                      shuffle=False)

    data_data_loader = DataLoader(dataset=data_ds,
                                  batch_sampler=data_batch_sampler,
                                  collate_fn=collate_fn)

    results = []
    model.eval()
    for batch in data_data_loader:
        logits = model(**batch)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy().tolist()
        probx = paddle.max(probs, axis=1).numpy().tolist()
        labels = [label_list[i] for i in idx]
        for zipper in zip(labels, probx):
            results.append({'label': zipper[0], 'prob': zipper[1]})

    # with open(args.output_file, 'w', encoding='utf-8') as f:
    #     f.write('text' + '\t' + 'label' +'\t' + 'prob' + '\n')
    #     for t, r in zip(data_ds.data, results):
    #         f.write(t["text"] + '\t' + str(r["label"]) + "\t" + str(r["prob"]) + '\n')
    # logger.info("Prediction results save in {}.".format(args.output_file))
    ret_info = []
    for t, r in zip(data_ds.data, results):
        ret_info.append({'text': t["text"], 'label': r["label"], 'prob': r["prob"]})
    return ret_info


if __name__ == "__main__":
    pprint.pprint(predict())