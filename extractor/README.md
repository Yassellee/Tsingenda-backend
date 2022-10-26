# extractor 使用说明

类 `Extractor` 用于从文本中抽取日期、时间和地点信息。它会对文本先进行 NER，之后会通过规则将日期和
时间转换为 python 的 `date` 和 `time` 类。使用方式可以参考如下：
```python
if __name__ == "__main__":
    instances = [
        "二零一八年十月六日，小明参观了清华科技园，然后去清芬园吃了中午饭，那是在9点二十分。",
        "2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。",
        "他一直在清华学堂卷到了晚上10点"
    ]
    extractor = Extractor()
    print(extractor.parse(instances))
```
输出为：
```bash
(base) PS F:\Learning\2022Fall\HCI\Tsingenda-backend\extractor> python extractor.py
[[('二零一八年', 'DATE', 0, 1), ('十月', 'DATE', 1, 2), ('六日', 'DATE', 2, 3), ('小明', 'PERSON', 4, 5), ('清华科技园', 'LOCATION', 7, 9), ('清芬园', 'LOCATION', 12, 13), ('9点二十分', 'TIME', 20, 21)], [('2021年', 'DATE', 0, 1), ('HanLPv2.1', 'WWW', 1, 2), ('次世代', 'DATE', 6, 7), ('北京立方庭', 'ORGANIZATION', 17, 19), ('自然语义科技公司', 'ORGANIZATION', 20, 24)], [(' 清华学堂', 'ORGANIZATION', 3, 5), ('晚上', 'TIME', 8, 9), ('10点', 'TIME', 9, 10)]]
[{'date': [('二零一八年十月六日', datetime.date(2018, 10, 6))], 'time': [('9点二十分', datetime.time(9, 20))], 'location': ['清华科技园', '清芬园']}, {'date': [('2021年', datetime.date(2021, 10, 26)), ('次世代', None)], 'time': [], 'location': ['北京立方庭', '自然语义科技公司']}, {'date': [], 'time': [('晚上', datetime.time(8, 0)), ('10点', datetime.time(10, 0))], 'location': ['清华学堂']}]
```
其中第一段是执行 NER 的结果，第二段是最后得到的结果。可以注意到这里还是有一点小问题的，比如说 “晚上10点” 
被分开了，然后被赋予了不同的时间。
