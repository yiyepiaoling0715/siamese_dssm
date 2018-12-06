import collections

from .funcs import ReSet

reset = ReSet()


def bracket_deal(entity_in):
    for pattern_iter in reset.bracket_extract_pattern:
        search_result = pattern_iter.search(entity_in)
        if search_result:
            return search_result.groups()
        return [entity_in]


def split_deal(entity_in):
    for pattern_iter in reset.split_pattern:
        synonym_words = pattern_iter.split(entity_in)
        synonym_words_deal = [word for word in synonym_words if word.strip('\r\n ')]
        if len(synonym_words_deal) > 1:
            return synonym_words_deal
    return [entity_in]


def pattern_deal(entity_in):
    entities_bracket = clean_inner_punc_linebreak_and_bracket(entity_in)
    entities_ret = []
    # for entity_iter in entities_bracket:
    entity_iter = entities_bracket
    entity_iter_split = split_deal(entity_iter)
    entities_ret.extend(entity_iter_split)
    return set(entities_ret)


def pattern_deal_only_del_punc(entity_in):
    entities_bracket = bracket_deal(entity_in)
    entities_ret = []
    for entity_iter in entities_bracket:
        entity_iter_split = split_deal(entity_iter)
        entities_ret.extend(entity_iter_split)
    return set(entities_ret)


def clean_inner_punc_linebreak_and_bracket(entity_in):# 只处理\n 和 各种括号，进行删除操作
    entity_cur = entity_in
    # print(entity_cur)
    for pattern_iter in reset.clean_inner_pattern:
        entity_cur = pattern_iter.sub('', entity_cur)
    for pattern_iter in reset.del_brackets_pattern:
        entity_cur = pattern_iter.sub('', entity_cur)
    return entity_cur
def clean_all_punctuaitons(entity_in):    #清理掉各种符号
    entity_cur = entity_in
    # print(entity_cur)
    for pattern_iter in reset.puncs_pattern:
        entity_cur = pattern_iter.sub('', entity_cur)
    return entity_cur

def clean_dict(dict_in):  # 将词典内的key，value 调用  clean_inner_punc_linebreak_and_bracket 函数
    dict_out = collections.defaultdict(set)
    for k, v in dict_in.items():
        # print(k,v)
        if '保单服务' in k:
            print(k)
        k_iter = clean_inner_punc_linebreak_and_bracket(k)
        if isinstance(v, (list, set)):
            for v_iter in v:
                v_iter_clean = clean_inner_punc_linebreak_and_bracket(v_iter)
                dict_out[k_iter].add(v_iter_clean)
        elif isinstance(v, str):
            dict_out[k_iter] = v
        else:
            raise ('什么格式')
    return dict_out


def dump2file(dict_in, filepath_in):
    with open(filepath_in, 'w', encoding='utf-8') as fw:
        for k, v in dict_in.items():
            if isinstance(v, (set, list)):
                str_iter = k + '\t' + '\t'.join(v) + '\n'
            elif isinstance(v, str):
                str_iter = k + '\t' + v + '\n'
                pass
            else:
                raise ValueError('什么格式')
            fw.write(str_iter)
