## Tsingenda Backend

### 运行环境

* Ubuntu20.04
* CUDA11.6
* CUDNN8.4.0
* RTX3090(24G显存)

### pip requirements

```
cn2an==0.5.19
Django==4.1.3
easyocr==1.6.0
hanlp==2.1.0b44
paddlenlp==2.4.4
scikit-learn==1.1.2
torch==1.12.1+cu116
transformers==4.24.0
paddleppaddle-gpu==2.4.1.post116
```

### 目录

```
work
└── Tsingenda-backend
    ├── README.md
    ├── requirements.txt
    ├── src
    │   ├── Classifier
    │   │   ├── bad_case.txt
    │   │   ├── checkpoint
    │   │   ├── command.txt
    │   │   ├── conf_checkpoint
    │   │   ├── conf_datasets
    │   │   ├── confidence.py
    │   │   ├── dataset
    │   │   ├── eval.py
    │   │   ├── __init__.py
    │   │   ├── main.py
    │   │   ├── output.txt
    │   │   ├── predict.py
    │   │   ├── train.py
    │   │   └── utils.py
    │   ├── __init__.py
    │   └── utils
    │       ├── datetimefinder.py
    │       ├── ner_extractor.py
    │       ├── ocr_extractor.py
    │       ├── README.md
    │       ├── rulebased_extractor.py
    │       └── text_summarizer.py
    ├── tsingenda
    │   ├── db.sqlite3
    │   ├── manage.py
    │   ├── tsingenda
    │   │   ├── asgi.py
    │   │   ├── __init__.py
    │   │   ├── settings.py
    │   │   ├── urls.py
    │   │   └── wsgi.py
    │   └── tsingenda_app
    │       ├── admin.py
    │       ├── apps.py
    │       ├── __init__.py
    │       ├── io.py
    │       ├── models.py
    │       ├── tests.py
    │       ├── urls.py
    │       └── views.py
    └── tsingenda_env.sh

```

* src目录下为NLP模型及代码
  * main.py为持续学习主进程，需与后端服务器一同开启。（见下）
* tsingenda目录下为django后端
  * `python manage.py runserver`运行后端服务，可以通过绑定ip/port将服务进程通过主机tcp端口打开。
* tsingenda_env.sh为小组在服务器上运行时设置的一些环境变量。环境变化时一些变量可能需要重新配置或者不需要配置（如`CUDNN_ROOT`和`LD_LIBRARY_PATH`），但是`PYTHONPATH`变量请按照tsingenda_env.sh配置。

### 后端Web服务

后端采用django框架编写，对前端app的操作和计算请求进行处理和响应。主逻辑位于views.py当中。

响应的服务类型包括：

* 登录(login)：对登录、注册、修改密码等请求进行响应

```json
// login/
// 用户登录
// post
{
    "data": {
        "action": "login", // 用户操作，可能是"login", "register", "change_password", "logout"
        "username": "Alice",
        "password": "123456",
    }
}
{
    "data": {
        "action": "register",
        "username": "Bob",
        "password": "234567",
    }
}
{
    "data": {
        "action": "change_password",
        "username": "Alice",
        "old_password": "123456",
        "new_password": "456789",
    }
}
{
    "data": {
        "action": "logout",
        "username": "Alice",
    }
}

// response
{
    "code": 200,
    "data": {
        "name": "Alice Smith",
    }
}
{
    "code": 200,
    "data": {
        "status": "login_fail", // 错误类型
        "detail": "wrong password", // 错误说明
    }
}
```

* 识别文本日程(raw_text)：对前端识别到的纯文本内容提取日程信息，返回包括在后端创建的该文本条目的数据库索引、识别的时间地点信息、主模型给出的日程置信度、个性化模型给出的置信度水平判断。

```json
// raw_text/
// 发送用户输入文本
// post
{
    "data": // 文本内容列表
    [
        "The homework will be due next Thursday.", // 文本内容
    	"AI was proposed in 1956",
    ]
}
// response
{
    "code": 200, 
    "data": // 分类结果列表，与request一一对应
    [
        {
            "id": 1,	// 总id
            "conf_id": 3,	// 该数据在该用户数据集中的唯一标识（不同用户的不同数据可能有相同的id）
            "raw_text": "Please summit your ppt before 23:59 tonight.",	// 待判定日程文本
            "dates": [
            	"二零一八年九月十六日",	// NER识别出的日期实体
            ],
    		"times": [
    			"9点20分"	// NER识别出的时间实体
            ],
            "locations": [
                "西操"	// NER识别出的地点实体
            ],
            "title": "集体锻炼打卡",	// 模型给出的摘要标题
			"is_agenda": true,	// 模型是否认为这是一个日程
            "confidence": 0.75,	// 置信度
            "confidence_high": true	// 分类器是否以高置信度认为这是一个日程（如果is_agenda为false，该项无意义）
            // 为false时，前端弹出三选项窗
            // 为true时，将该文本加入日程
    	},
		...
    ]   
}
```

* OCR识别日程(image)：对发来的截图请求进行处理，先得到纯文本，再进行与上面类似的计算。

```json
// image/
// 发送屏幕截图图片
// post
{
    "data": // 图片列表
    [
       "...", // base64 string，图片
    ]
}
// response
{
    "code": 200,
    "data": // 分类结果列表，对应request图片顺序，每张图也可能有多句 
    [
        [
            {
                "id": 1,	// 总id
                "conf_id": 3,	// 该数据在该用户数据集中的唯一标识（不同用户的不同数据可能有相同的id）
                "raw_text": "Please summit your ppt before 23:59 tonight.",	// 待判定日程文本
                "dates": [
                    "二零一八年九月十六日",	// NER识别出的日期实体
                ],
                "times": [
                    "9点20分"	// NER识别出的时间实体
                ],
                "locations": [
                    "西操"	// NER识别出的地点实体
                ],
                "title": "集体锻炼打卡",	// 模型给出的摘要标题
                "is_agenda": true,	// 模型是否认为这是一个日程
                "confidence": 0.75,	// 置信度
                "confidence_high": true	// 分类器是否以高置信度认为这是一个日程（如果is_agenda为false，该项无意义）
                // 为false时，前端弹出三选项窗
                // 为true时，将该文本加入日程
            },
            {
                ...
            }
        ]
    ]
}

```

* 用户反馈(feedback)：对用户对日程做出的相应操作，前端进行反馈，后端据此更新数据标注。

```json
// feedback/
// 前端用户的反馈结果
{
    "data": [
        {
            "id": 4,
            "conf_id": 10,
            "text": "The homework will be due next Thursday.", // 文本内容
            "is_agenda": true,	// 表示用户是否将这一条设为了日程
            // 如果之前出现了三选项窗且用户选择了不是日程，或者之前直接将该条加入日程但是被用户删除了，则为false，否则为true
            "confidence_high": true	// 用户是否认为这是一个高置信度日程
            // 判定方法为：如果之前弹出了三选项窗，且用户选择了不是日程/不加入日程，则该项为false；如果用户选择了是日程，则该			 // 项为true
        }
    ]
}
// response
{
    "code": 200,
}
```

### NLP模型

#### 1. 主模型

- 比对多个模型，最终选用ernie-3.0-base-zh作为预训练模型，以下为比对结果，采用情感二分类语料别对。

| 模型              | 准确率 | F1 SCORE |
| ----------------- | ------ | -------- |
| BERT-wwm          | 0.8886 | 0.8845   |
| Roberta           | 0.8831 | 0.8877   |
| Electra           | 0.8913 | 0.8889   |
| ernie-3.0-base-zh | 0.9102 | 0.9125   |

- 采用paddlepaddle框架，编写训练、精调、检验、预测等接口

~~~
├── Classifier
   ├── bad_case.txt
   ├── checkpoint
   ├── command.txt
   ├── dataset
   ├── eval.py
   ├── __init__.py
   ├── main.py
   ├── output.txt
   ├── predict.py
   ├── train.py
   └── utils.py
~~~

- 冷启动语料选自组员微信通知群等处的语料

#### 2. 日程提取

- 使用 hanlp 提供的多任务语言模型 `CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH` 来进行命名实体识别，可以比较准确地识别出文本中的日期、时间、地点文本信息；
- 采用上述模型的一个问题是可能错误地将时间、日期文本分割开，为后续的格式化造成困难；考虑到这些信息一般形式比较规整，就额外添加了采用规则匹配的方式来进行日期、时间识别。
- 目前的开源代码中没有一个比较好的支持中文的讲日期、时间文本格式化的工具类，所以自己编写了一个，支持绝大多数的中文日期、时间格式，可以将其转换为 `python` 的 `Datetime` 对象。

#### 3. 标题生成

- unimo-text-1.0-summary作为摘要生成模型
- 尝试Openai大语言模型Ada和Davinci的文本补全——深感过于依赖prompt且难以引导，故最终舍弃
- 尝试更加成熟的基于预训练模型的关键词提取技术——太过平凡且语义不明
- 最终选择基于paddlepaddle框架的unimo-text-1.0-summary

#### 4. OCR识别

- 模型选用：detection & recognition -- craft_mlt_25k & pretrained_ic15_res50
- 加入大量规则调整——连续两个数字后的8改成日，夭改成天，什么样的两个句子应该合并等具体规则
- 投入主模型前过滤噪音，如识别结果中的绝大部分短文本

#### 5. 置信度分类器

对主模型给出的置信度，希望在不同用户的app上能够给出不同的解释和操作。

后端为每个注册用户分配了一个置信度分类线性层，接受主模型的置信度作为输入，输出前端app的不同处理。

* 若主模型认为数据不是日程，该步分类无意义，前端无操作；
* 若主模型认为数据是日程
  * 若置信度分类器认为该置信度高(confidence_high)，则前端直接将该日程加入日程表，无弹窗提示；
  * 否则，前端弹出一个三选项窗，希望用户选择对这条疑似日程进行取舍。选项依次为"不是日程"/"不加入日程"/"加入日程"。

此外，用户还可以对自动加入的日程进行删除和修改。

上述用户的反馈操作会返回给后端，用于对数据做标注。用户选择加入日程代表代表置信度高，其他操作代表置信度低。这样形成了持续学习的模式，每个用户的选择将影响其个性化线性层的参数。

### 模型存储架构

模型的用户数据由django数据库维护。主模型数据共同维护，用户个性化的置信度分类数据、模型路径均与用户对象添加外键关系保存。

src/Classifier/main.py是需要与服务器同时开启的进程，开启后，每隔固定时间后考察数据库是否发生变化，对于有变化的模型重新训练保存参数。