import paddlenlp as ppnlp
import paddle, functools, numpy
import os
import paddle.nn.functional as F


# configuration
label_map = {0: 'negative', 1: 'positive'}
ckpt_dir = os.path.join(os.getcwd(), "../../data/Classifier/models")
predict_data_path = os.path.join(os.getcwd(), "../../data/Classifier/raw_data/predict_data.txt")


def predict():
    def data_preprocessor(examples, tokenizer, max_seq_length, is_test=False):
        result = tokenizer(text=examples["text"], max_seq_len=max_seq_length)
        if not is_test:
            result["labels"] = examples["label"]
        return result
    def read_data(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            cnt = 0
            for line in lines:
                yield {"text": line, "qid": cnt, "label": 0}
                cnt += 1
    

    predict_dataset = ppnlp.datasets.load_dataset(
        read_data, data_files = predict_data_path, splits=('test'), lazy=False)
    model = ppnlp.transformers.BertForSequenceClassification.from_pretrained(ckpt_dir)
    tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(ckpt_dir)
    trans_func_test = functools.partial(data_preprocessor, tokenizer=tokenizer, max_seq_length=128, is_test=True)
    test_ds_trans = predict_dataset.map(trans_func_test)
    collate_fn_test = ppnlp.data.DataCollatorWithPadding(tokenizer)
    test_batch_sampler = paddle.io.BatchSampler(test_ds_trans, batch_size=32, shuffle=False)
    test_data_loader = paddle.io.DataLoader(dataset=test_ds_trans, batch_sampler=test_batch_sampler, collate_fn=collate_fn_test)

    results = []
    model.eval()
    for batch in test_data_loader:
        input_ids, token_type_ids = batch['input_ids'], batch['token_type_ids']
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=-1)
        idx = paddle.argmax(probs, axis=1).numpy()
        # generate a list of probs for each sentence
        probx = paddle.max(probs, axis=1).numpy().tolist()
        idx = idx.tolist()
        preds = [label_map[i] for i in idx]
        for i in zip(preds, probx):
            results.append({"label": i[0], "prob": i[1]})

    return results