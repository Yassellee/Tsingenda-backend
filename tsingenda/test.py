from src.Classifier.train import train, get_args
from src.Classifier import predict
from src.Classifier import confidence
from src.utils.ocr_extractor import *
from src.utils.ner_extractor import *
from src.utils.text_summarizer import *

def main():
    # a = NERExtractor()
    # b = OCRExtractor(gpu_state = True)
    # c = summarizer()
    # print(a.parse([
    #     "二零一八年十月六日，小明参观了清华科技园，然后去清芬园吃了中午饭，那是在9点二十分。",
    #     "2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。",
    #     "他一直在清华学堂卷到了晚上10点"
    # ]))
    
    # b.demo()
    # args = predict.get_args()
    # model = predict.get_model(args)
    # tokenizer = predict.get_tokenizer(args)
    # print(c.summarize("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。"))
    # print(predict.predict(args, model, tokenizer, ["2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。"]))
    
    # conf_model = confidence.get_model(None)
    train(get_args())
    


if __name__ == "__main__":
    main()