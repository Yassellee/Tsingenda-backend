from ctypes import Union
from datetime import datetime, date, time, timedelta
from typing import List, Optional
import re
import cn2an

def conv_str2digit(text: str):
    pattern = r"([零一二三四五六七八九十百千万亿壹贰叁肆伍陆柒捌玖拾佰仟]+)"
    split_text = re.split(pattern, text)
    for idx in range(1, len(split_text), 2):
        split_text[idx] = str(int(cn2an.cn2an(split_text[idx], "smart")))
    
    return "".join(split_text)

class DateTimeFinder:
    # 包含：年月日，年月，月日，年，月，日
    DATE_TEMPLATE = ["%%Y%s%%m%s%%d%s", "%%Y%s%%m%s", "%%m%s%%d%s", "%%Y%s", "%%m%s", "%%d%s"]
    DATE_TEMPLATE_SMALL = [s.replace("%%Y", "%%y") for s in DATE_TEMPLATE]
    # 包含：时分秒，时分，时
    TIME_TEMPLATE = ["%%H%s%%M%s%%S%s", "%%H%s%%M%s", "%%H%s"]
    def __init__(self) -> None:
        self.PATTERN_DATETIME = []
        self.PATTERN_DATE = []
        self.PATTERN_TIME = []
        
        date_fill_format = self.get_date_fill_format()
        
        for idx, (temp, temp_s) in enumerate(zip(self.DATE_TEMPLATE, self.DATE_TEMPLATE_SMALL)):
            to_fill = date_fill_format[idx]
            for fill_str in to_fill:
                self.PATTERN_DATE.extend([temp%fill_str, temp_s%fill_str])
        
        self.PATTERN_DATE = list(set(self.PATTERN_DATE))
        # print(self.PATTERN_DATE)
        
        time_fill_format = self.get_time_fill_format()
        
        for idx, temp in enumerate(self.TIME_TEMPLATE):
            to_fill = time_fill_format[idx]
            for fill_str in to_fill:
                self.PATTERN_TIME.append(temp%fill_str)
        
        # print(self.PATTERN_TIME)
        
        # 这里我们认为有意义的datetime组合至少需要包含%d
        for pdate in self.PATTERN_DATE:
            if "%d" in pdate:
                for ptime in self.PATTERN_TIME:
                    self.PATTERN_DATETIME.extend([pdate + ptime, pdate + " " + ptime])
        
        # print(self.PATTERN_DATETIME)
    
    def get_date_fill_format(self):
        date_fill_format = [
            [("-", "-", ""), ("/", "/", ""), (".", ".", ""), ("\\", "\\", "")],
            [("-", ""), ("/", ""), (".", ""), ("\\", "")],
            [("-", ""), ("/", ""), (".", ""), ("\\", "")],
            [],
            [],
            []
        ]
        
        # 添加中文模式
        year, month = "年", "月"
        for day in ["日", "号"]:
            for idx in range(len(date_fill_format)):
                fill_pattern = [
                    (year, month, day),
                    (year, month),
                    (month, day),
                    (year,),
                    (month,),
                    (day,)
                ][idx]
                date_fill_format[idx].append(fill_pattern)
        
        return date_fill_format
    
    def get_time_fill_format(self):
        time_fill_format = [
            [(":", ":", "")],
            [(":", "")],
            []
        ]
        minute, second = "分", "秒"
        for hour in ["时", "点"]:
            for idx in range(len(time_fill_format)):
                to_fill = [
                    (hour, minute, second),
                    (hour, minute),
                    (hour,)
                ][idx]
                time_fill_format[idx].append(to_fill)
        
        return time_fill_format

    def parse(self, text: str, patterns: List[str]) -> Optional[datetime]:
        text = conv_str2digit(text)
        for pattern in patterns:
            try:
                parsed_datetime = datetime.strptime(text, pattern)
                current_datetime = datetime.now()
                # 自动填补年、月、日、时
                if (not ("%Y" in pattern)) and (not ("%y" in pattern)):
                    parsed_datetime = parsed_datetime.replace(year=current_datetime.year)
                if not ("%m" in pattern):
                    parsed_datetime = parsed_datetime.replace(month=current_datetime.month)
                if not ("%d" in pattern):
                    parsed_datetime = parsed_datetime.replace(day=current_datetime.day)
                if not ("%H" in pattern):
                    parsed_datetime = parsed_datetime.replace(hour=current_datetime.hour)
                return parsed_datetime
            except Exception as e:
                pass

        return None
    
    def parse_date(self, text: str) -> Optional[date]:
        parsed_date = self.parse_special(text)
        if parsed_date and isinstance(parsed_date, date):
            return parsed_date
        parsed_datetime = self.parse(text, self.PATTERN_DATE)
        if parsed_datetime:
            return parsed_datetime.date()
        return None
    
    def parse_time(self, text: str, context: List[str]=None, idx: int=None) -> Optional[time]:
        parsed_time = self.parse_special(text)
        if parsed_time and isinstance(parsed_time, time):
            return parsed_time
        parsed_datetime = self.parse(text, self.PATTERN_TIME)
        if parsed_datetime:
            res = parsed_datetime.time()
            if context is not None and idx is not None:
                if idx > 0 and context[idx-1] in ["下午", "晚上", "今晚", "傍晚"] and res.hour <= 12:
                    res = res.replace(hour=res.hour+12)
            return res
        return None

    def parse_datetime(self, text: str) -> Optional[datetime]:
        return self.parse(text, self.PATTERN_DATETIME)
    
    def parse_special(self, text: str):
        current_date = datetime.now().date()
        
        result_list = {
            "大前天": current_date - timedelta(days=3),
            "前天": current_date - timedelta(days=2),
            "昨天": current_date - timedelta(days=1),
            "今天": current_date,
            "明天": current_date + timedelta(days=1),
            "后天": current_date + timedelta(days=2),
            "大后天": current_date + timedelta(days=3),
            
            "上午": time(9),
            "中午": time(12),
            "下午": time(15),
            "傍晚": time(17),
            "早晨": time(7),
            "晚上": time(8),
            "今晚": time(8)
        }
        
        return result_list.get(text, None)

def test_finder():
    finder = DateTimeFinder()
    date_set = ["10月6日", "19年2月", "2019-1-1", "19/2/1"]
    # print(datetime.strptime("10月6日", "%m月%d日"))
    for d in date_set:
        print(finder.parse_date(d))
    
    time_set = ["11:05", "1点10分", "5时", "6点"]
    for t in time_set:
        print(finder.parse_time(t))

    dt_set = ["19年10月6日5点", "8月6日9点30分", "8.6 4:53:21"]
    for dt in dt_set:
        print(finder.parse_datetime(dt))


if __name__ == "__main__":
    text = "二零一八年十月六日，小明参观了清华科技园，然后去清芬园吃了中午饭，那是在9点二十分。"
    print(conv_str2digit(text))
    test_finder()
