import os
import argparse
import functools
import numpy as np
from collections import namedtuple
import random

import paddle
import paddle.nn.functional as F
from paddlenlp.utils.log import logger
from paddle.io import DataLoader, BatchSampler
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
import time, pprint

from src.Classifier.utils import preprocess_function, read_local_dataset

# # yapf: disable
# parser = argparse.ArgumentParser()
# parser.add_argument('--device', default="gpu", help="Select which device to train model, defaults to gpu.")
# parser.add_argument("--dataset_dir", default="dataset", type=str, help="Local dataset directory should include data.txt and label.txt")
# parser.add_argument("--output_file", default="output.txt", type=str, help="Save prediction result")
# parser.add_argument("--params_path", default="/root/work/Tsingenda-backend/src/Classifier/checkpoint/", type=str, help="The path to model parameters to be loaded.")
# parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
# parser.add_argument("--batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
# parser.add_argument("--data_file", type=str, default="data.txt", help="Unlabeled data file name")
# parser.add_argument("--label_file", type=str, default="label.txt", help="Label file name")
# args = parser.parse_args()
# # yapf: enable

# print("loading models...")
# paddle.set_device(args.device)
# model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
# tokenizer = AutoTokenizer.from_pretrained(args.params_path)
# print("models loaded")


def get_args():
    args = namedtuple('args', [
        'device',
        'dataset_dir',
        'output_file',
        'params_path',
        'max_seq_length',
        'batch_size',
        'data_file',
        'label_file',
    ])
    args.device = "gpu"
    args.dataset_dir = "dataset"
    args.output_file = "output.txt"
    args.params_path = "/root/work/Tsingenda-backend/src/Classifier/checkpoint/"
    args.max_seq_length = 128
    args.batch_size = 32
    args.data_file = "data.txt"
    args.label_file = "label.txt"
    return args

def get_model(args):
    print("loading models...")
    paddle.set_device(args.device)
    model = AutoModelForSequenceClassification.from_pretrained(args.params_path)
    print("models loaded")
    return model

def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.params_path)
    return tokenizer


@paddle.no_grad()
def predict(args, model, tokenizer, sentence_list = None):
    """
    Predicts the data labels.
    """

    label_list = []
    label_path = os.path.join("/root/work/Tsingenda-backend/src/Classifier/dataset", args.label_file)
    with open(label_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            label_list.append(line.strip())

    if sentence_list is None:
        data_ds = load_dataset(read_local_dataset,
                            path=os.path.join(args.dataset_dir, args.data_file),
                            is_test=True,
                            lazy=False)
    else:
        # create a temp file to write the sentence list, delete it after loading the dataset
        temp_predict_file = os.path.join("/root/work/Tsingenda-backend/src/Classifier/dataset", "temp_predict{}.txt".format(random.random()))
        with open(temp_predict_file, 'w', encoding='utf-8') as f:
            for line in sentence_list:
                f.write(line + '\n')
        data_ds = load_dataset(read_local_dataset,
                               path=temp_predict_file,
                               is_test=True,
                               lazy=False)
        os.remove(temp_predict_file)

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

    ret_info = []
    for t, r in zip(data_ds.data, results):
        ret_info.append({'text': t["text"], 'label': r["label"], 'prob': r["prob"]})
    return ret_info


if __name__ == "__main__":
    demo_predict_sentences = [
        "【重要：关于“一天一检”工作安排】各位同学好，近期北京疫情形势严峻复杂，根据海淀区安排，从11月21日起高校师生核酸检测调整为【“一天一检”】。提醒同学优先白天错峰前往核酸检测点完成检测。【听涛西广场】11:00-13:30，16:00-21:00【祖龙广场、新清西侧】11:00-13:30，16:00-19:30【紫荆篮球场】18:00-22:00",
        "欢迎大家明天下午2点一起来fit楼见证冠军的诞生！除了可以现场见证高端对局和听到大佬们的策略分享，我们还准备了精美的茶歇和奖品",
        "以及，“学术新星计划”的报名将于本周日23:59分截止。请还有意报名的同学抓紧时间。",
        "1989年1月17日上午,林宗棠让我陪领导到机场看歼8飞机,称要中国多支付两三亿美元。",
        "【CCF】尊敬的参会嘉宾黄书鸿：非常感谢您注册参加CNCC2022。大会将于12月10日全线上形式举办并于早上08:30开幕。大会日程和线上参会指南如下，供您提前参考"
    ]
    args = get_args()
    model = get_model(args)
    tokenizer = get_tokenizer(args)
    pprint.pprint(predict(args, model, tokenizer, demo_predict_sentences))