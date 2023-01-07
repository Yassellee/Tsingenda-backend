from paddlenlp import Taskflow


class summarizer():
    def __init__(self, model="unimo-text-1.0-summary"):
        self.task = Taskflow("text_summarization", model=model)

    def summarize(self, text):
        return self.task(text)


if __name__ == "__main__":
    text = [
        "群公告同学们好，欢迎大家报名腾讯犀牛鸟开源人才计划，内部报名截止时间7月3日；系里有推荐名额。",
        "Ubicomp将于明年三月在德国举办，实验室有兴趣的同学可以前往参加"    
    ]
    model = summarizer()
    summary = model.summarize(text)
    print(summary)