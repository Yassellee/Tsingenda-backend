from .datetimefinder import conv_str2digit
import re
from typing import List


class RuleBasedExtractor:
    DATE_TEMPLATE = ["\d+%s\d+%s\d+%s", "\d+%s\d+%s", "\d+%s\d+%s", "\d+%s", "\d+%s", "\d+%s"]
    TIME_TEMPLATE = ["\d+%s\d+%s\d+%s", "\d+%s\d+%s", "\d+%s"]
    def __init__(self) -> None:
        self.PATTERN_DATE = []
        self.PATTERN_TIME = []
        
        date_fill_format = self.get_date_fill_format()
        
        for idx, temp in enumerate(self.DATE_TEMPLATE):
            to_fill = date_fill_format[idx]
            format_patterns = list(set([temp%fill_str for fill_str in to_fill]))
            self.PATTERN_DATE.extend(format_patterns)
        
        time_fill_format = self.get_time_fill_format()
        
        for idx, temp in enumerate(self.TIME_TEMPLATE):
            to_fill = time_fill_format[idx]
            format_patterns = list(set([temp%fill_str for fill_str in to_fill]))
            self.PATTERN_TIME.extend(format_patterns)
        
        # print(self.PATTERN_DATE)
        # print(self.PATTERN_TIME)
    
    def find_all(self, text: str, pattern: List[str]) -> List[str]:
        res = []
        dummy = ""
        for p in pattern:
            all_match = re.findall(p, text)
            for match in all_match:
                if match not in dummy:
                    # print(p)
                    # print(match)
                    res.append(match)
                    dummy += (match + " ")
        
        return res
    
    def find_date(self, text: str) -> List[str]:
        return self.find_all(text, self.PATTERN_DATE)
    
    def find_time(self, text: str) -> List[str]:
        return self.find_all(text, self.PATTERN_TIME)
    
    def get_date_fill_format(self):
        date_fill_format = [
            [("-", "-", ""), ("/", "/", ""), ("\.", "\.", ""), ("\\", "\\", "")],
            [("-", ""), ("/", ""), ("\.", ""), ("\\", "")],
            [("-", ""), ("/", ""), ("\.", ""), ("\\", "")],
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

if __name__ == "__main__":
    text = "2019年十月2日10:30，我去参观了博物馆，1840年7月7日发生的事件让人印象深刻。"
    p = "(\d+)年(\d+)月(\d+)日"
    text = conv_str2digit(text)
    print(text)
    finder = RuleBasedExtractor()
    print(finder.find_date(text))
    print(finder.find_time(text))