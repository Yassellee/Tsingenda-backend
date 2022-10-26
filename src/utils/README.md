

# ./src/utils 工具类使用说明



## NERextractor 使用说明

类 `NERExtractor` 用于从文本中抽取日期、时间和地点信息。它会对文本先进行 NER，之后会通过规则将日期和
时间转换为 python 的 `date` 和 `time` 类。使用方式可以参考如下：

```python
if __name__ == "__main__":
    instances = [
        "二零一八年十月六日，小明参观了清华科技园，然后去清芬园吃了中午饭，那是在9点二十分。",
        "2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。",
        "他一直在清华学堂卷到了晚上10点"
    ]
    extractor = NERExtractor()
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



## OCRExtractor类使用说明

​		类OCRExtractor会将图片中的文本信息提取出来，生成列表形式信息返回。下面讲解使用方法。

##### 实例化

~~~python
test_ocr_extractor = OCRExtractor(gpu_state = True)
~~~

​		在实例化OCRExtractor类时，参数gpu_state表示识别文本时是否使用gpu，使用gpu会大幅加快速度。

##### extract_single_image

~~~python
extracted_text_list = test_ocr_extractor(image = <path to image / OpenCV image object / an image file as bytes>)
~~~

​		extract_single_image函数，对单张图片进行ocr。传入的image参数可以为

- 图片路径
- OpenCV的image object
- 字节流的图片文件
- 传入URL理论可行，有时会崩，还是算了

​		返回值是一个列表，列表的每一项代表从图片中提取出的一段文本信息。例如，当输入如./data/test1_CET.png时，返回值如下

~~~python
['2022年下半年全国大学英语四六级报名通知', '2022年下半年全国大学英语四 六级考试时间', '11月19日,四级口试', '11月20日,六级口试', '12月10日,四  六级笔试开考科目:英语四级和六级。英语四级口语。英语六级口语报名资格:CET笺试报考资格为全日制普通及成人高等院校本秫专科在校生。在籍研究生。修完大学英语四级课程的学生可报考英语四级,修完大学英语六级课程且英语四级成绩达到425分及以上的学生可报考英语六级。', 'CET-SET报考资格为完成对应级别笔试科目报考的考生。即完成本次CET4笔试报名后可报考CET-SET4, 完成本次', 'CET6笔试报名后可报考CET-SET6。', '特别说明:  因我校考点不具备四六级口试条件。且其他院校考点因疫情防控需要暂不接受外校学生报名,此次考试我校学生不能报名参加口试。 请予理解。', '报名时间和网址', '10月27日13:30-11月4日17:00 http://cet-bm neea.edu.cn其他具体流程请参考以下文件']
~~~

##### extract_multiple_images

​		extract_multiple_images函数，一次性对多张图片进行OCR操作。传入参数images是一个列表，列表的每一项和extract_single_image函数的image含义相同。返回值是一个由多个extract_single_image返回的列表组成的列表。

##### demo

​		预留了demo函数，你总可以用demo函数观看extract_single_image和extract_multiple_images会干什么。

##### 一些对OCR识别结果的规则处理

- replace 夭 with 天
- replace 8 with 日 if the previous two characters are digits
- if a sentence does not end with a 。, and the next sentence starts with a Chinese character or English character, then merge them

