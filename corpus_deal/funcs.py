import re
import yaml
import os
import itertools
import collections

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
class ReSet(object):
    _pattern_split = [re.compile('[/、\t ]+'), ]
    _pattern_brackets = [re.compile('([\w/]+)[（(]+[\w/]+[)）]+'), ]
    _pattern_clean = [re.compile('[\n ]+')]
    _pattern_punc = [re.compile('[()（）《》<>；;。!！？?/\'\"“”]+'), ]
    _pattern_del_brackets = [re.compile('[()（）<>《》 ]+')]
    # 零': 0, '点': '.', '一': 1, '二': 2, '三': 3, '叁': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
    _chinese_num_unit_measure = [re.compile('(?P<number>[零一二三四五六七八九十百千万两点]+)(?P<unit>周岁|个|岁|人|年|保额|元|块钱)+')]
    _chinese_num = [re.compile('(?P<number>[零一二三四五六七八九十百千万点两]+)')]
    _arabnum_and_unit = [re.compile('(?P<number>[0-9]+)(周岁|个|岁|人|年|保额|元|块钱)+')]
    _character_and_unit = [re.compile('(?P<character>[a-z]{1,2})(周岁|个|岁|人|年|保额|元|块钱)+')]
    _arab_num = [re.compile('(?P<number>\d+)(周岁|个|岁|人|年|保额|元|块钱)+'), ]
    _job_str = [re.compile('(是做|从事)+(?P<job>.{,4})(的|可以)+'), re.compile('职业是(?P<job>.{,4})[,.]+'), ]

    def __init__(self):
        pass
    @property
    def character_and_unit(self):
        return self._character_and_unit
    @property
    def arabnum_and_unit(self):
        return self._arabnum_and_unit
    @property
    def del_brackets_pattern(self):
        return self._pattern_del_brackets

    @property
    def puncs_pattern(self):
        return self._pattern_punc

    @property
    def split_pattern(self):
        return self._pattern_split

    @property
    def bracket_extract_pattern(self):
        return self._pattern_brackets

    @property
    def clean_inner_pattern(self):
        return self._pattern_clean

    @property
    def chinese_number_pattern(self):
        return self._chinese_num

    @property
    def chinese_number_unitmeasur_pattern(self):
        return self._chinese_num_unit_measure

    @property
    def job_pattern(self):
        return self._job_str

    @property
    def num_pattern(self):
        return self._arab_num


reset = ReSet()


def list2re(list_in):
    return re.compile('|'.join(list_in))


def readfile_line2list(filepath_in):
    with open(filepath_in, 'r', encoding='utf-8') as fr:
        return [line.strip('\r\n ') for line in fr]


def readfile_line2dict(filepath_in):
    with open(filepath_in, 'r', encoding='utf-8') as fr:
        lines = [line.encode('utf-8').decode('utf-8-sig').strip('\r\n ') for line in fr]  # 非法字符清洗
        lines_list = [reset.split_pattern[0].split(line) for line in lines]
        for index, line_iter in enumerate(lines_list):
            lines_list[index] = [elem for elem in line_iter if elem]
        line_o2m_dict = {item[0]: item for item in lines_list}
        line_o2o_dict = {word_iter: itemrow[0] for itemrow in lines_list for word_iter in itemrow}
        line_o2o_sorted = sorted(line_o2o_dict.items(), key=lambda x: len(x[0]), reverse=True)
        line_o2o_dict_order = collections.OrderedDict()
        for item in line_o2o_sorted:
            line_o2o_dict_order[item[0].strip('\r\n ')] = item[1].strip('\r\n ')
        return line_o2m_dict, line_o2o_dict_order


def read_yaml_dict_onelayer(yaml_filepath):
    assert os.path.isfile(yaml_filepath)
    with open(yaml_filepath, 'r', encoding='utf-8') as fr:
        dict_ret = yaml.load(fr)
        return dict_ret

def read_data_from_paths_set(classify2filepaths_in):
    classification2words_dict=collections.defaultdict(set)   #{domain:words}
    word2classification_dict=dict()   # {word:domain}
    entity_o2o_dict=dict()  #{sim1:std1}
    entity_o2m_dict=collections.defaultdict(set)  #{std1:[sim1,sim2,...]}
    classification2std2sim_dict=collections.defaultdict(dict)  # {domain:std1:[sim1,sim2,...]}
    classification2sim2std_o2o_dict=collections.defaultdict(dict)  # {domain:sim1:std1}
    for label_iter,filepaths_iter in classify2filepaths_in.items():
        for filepath_iter in filepaths_iter:
            iter_o2m_dict, iter_o2o_dict=readfile_line2dict(filepath_iter)
            classification2std2sim_dict[label_iter].update(iter_o2m_dict)
            classification2words_dict[label_iter].update(list(itertools.chain.from_iterable(iter_o2m_dict.values())))
            word2classification_dict.update({key: label_iter for key, value in iter_o2m_dict.items()})
            entity_o2o_dict.update(iter_o2o_dict)
            classification2sim2std_o2o_dict[label_iter].update(iter_o2o_dict)
            for one_iter,many_iter in iter_o2m_dict.items():
                entity_o2m_dict[one_iter].update(many_iter)
    return  classification2words_dict,word2classification_dict,entity_o2o_dict,entity_o2m_dict,classification2std2sim_dict,classification2sim2std_o2o_dict


def dict2sorted_dict(dict_in):
    dict_items=sorted(dict_in.items(),key=lambda x:len(x[0]),reverse=True)
    dict_order=collections.OrderedDict()
    for item in dict_items:
        dict_order[item[0]]=item[1]
    return dict_order

def list2sorted_list(list_in):
    return sorted(list_in,key=lambda x:len(x),reverse=True)

def write2txt(o2m_dict,filepath):
    # assert os.path.isfile(filepath)
    with open(filepath,'w') as fw:
        for k,v in o2m_dict.items():
            k_str=''.join(k)
            line_str=k_str+'\t'+'\t'.join(v)+'\n'
            fw.write(line_str)

class MyLogger(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    # from config.setting import PathUtil
    # pu=PathUtil()
    # result=readfile_line2dict(pu.domain_filepath)
    # print(result)
    from common.pathutil import PathUtil

    flag = 3
    if flag == 1:
        out = jieba_add_words()
        print(out)
    elif flag == 2:
        rs = ReSet()
        s = '退保/取消合同/终止合同/终止保险/取消保险'
        ret = rs.split_pattern.split(s)
        print(ret)
    elif flag == 3:
        yamlpath = PathUtil().get_pattern_with_represent_yamlpath
        dict_ret = read_yaml_dict_onelayer(yamlpath)
        # print(ret)
        # for key,values in ret.items():
        #     for value_iter in values:
        #         print(key,value_iter)
