import hanlp
from typing import Tuple, Union, List
from .datetimefinder import DateTimeFinder, conv_str2digit
from .rulebased_extractor import RuleBasedExtractor
import re


LOCATION = ["LOCATION", "ORGANIZATION"]

DATE = ["DATE"]

TIME = ["TIME"]

class NERExtractor:
    def __init__(self) -> None:
        self.hanlp = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
        self.datefinder = DateTimeFinder()
        self.rule = RuleBasedExtractor()
        
    def do_ner(self, instance: Union[str, List[str]]):
        if isinstance(instance, str):
            instance = [instance]
        res = self.hanlp(instance, tasks="ner/msra")["ner/msra"]
        print(res)
        return res
    
    def get_date_and_location(self, ner_info: List[Tuple[Union[str, int]]], text: str):
        time_info = []
        date_info = []
        location_info = []
        for entity in ner_info:
            entity_type = entity[1]
            if entity_type in LOCATION:
                location_info.append(entity)
            elif entity_type in TIME:
                time_info.append(entity)
            elif entity_type in DATE:
                date_info.append(entity)
        
        date_info = self.get_continuous_list(date_info)
        time_info = [item[0] for item in time_info]
        location_info = self.get_continuous_list(location_info)
        
        date_match = self.rule.find_date(text)
        time_match = self.rule.find_time(text)
        
        date_match = [d for d in date_match if d not in date_info]
        time_match = [t for t in time_match if t not in time_info]
        
        date_info = date_match + date_info
        time_info = time_match + time_info


        date_info = [(d, self.datefinder.parse_date(d)) for d in date_info]
        time_info = [(t, self.datefinder.parse_time(t, time_info, idx)) for idx, t in enumerate(time_info)]
        
        return {
            "date": date_info,
            "time": time_info,
            "location": location_info
        }
        
    def parse(self, text: Union[str, List[str]]):
        if isinstance(text, list):
            text = [conv_str2digit(t) for t in text]
        else:
            text = [conv_str2digit(text)]
        ner_info = self.do_ner(text)
        datetime_location_info = [self.get_date_and_location(item, t) for item, t in zip(ner_info, text)]
        return datetime_location_info
    
    def get_continuous_list(self, entity_list: List[Tuple[Union[str, int]]]):
        if not entity_list:
            return []
        
        res_list = []
        p_word, _, _, p_ed = entity_list[0]
        for idx in range(1, len(entity_list)):
            c_word, _, c_st, c_ed = entity_list[idx]
            if c_st == p_ed:
                p_word += c_word
            else:
                res_list.append(p_word)
                p_word = c_word

            p_ed = c_ed
        
        res_list.append(p_word)
        
        return res_list

if __name__ == "__main__":
    instances = [
        "二零一八年十月六日，小明参观了清华科技园，然后去清芬园吃了中午饭，那是在9点二十分。",
        "2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。",
        "他一直在清华学堂卷到了晚上10点",
        "请你2022年十二月十三日出席CCF- A会议，务必到场。"
    ]
    extractor = NERExtractor()
    print(extractor.parse(instances))