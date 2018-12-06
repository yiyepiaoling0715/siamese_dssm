import os
import pandas as pd
import numpy as np
import collections
import itertools
import copy
from corpus_deal.excel_funcs import pattern_deal, clean_inner_punc_linebreak_and_bracket, clean_all_punctuaitons
import jieba

xlsxpath = './raw.xlsx'


def col2bigroup(col_first, col_second, index_cur, col_third=None):
    row_set = set()
    if isinstance(col_first[index_cur], float):
        pass
    else:
        row = col_first[index_cur]
        row_split = pattern_deal(row)
        row_split_cleaned = [clean_all_punctuaitons(row_iter) for row_iter in row_split]
        row_set.update(row_split_cleaned)
    if isinstance(col_second[index_cur], float):
        pass
    else:
        row_index = col_second[index_cur]
        sents_row = pattern_deal(row_index)
        row_split_cleaned = [clean_all_punctuaitons(row_iter) for row_iter in sents_row]
        row_set.update(row_split_cleaned)
    if col_third is not None:
        if isinstance(col_third[index_cur], float):
            pass
        else:
            row_index = col_third[index_cur]
            sents_row = pattern_deal(row_index)
            row_split_cleaned = [clean_all_punctuaitons(row_iter) for row_iter in sents_row]
            row_set.update(row_split_cleaned)
    return row_set


def read(sheet_num):
    sheet = pd.read_excel(xlsxpath, sheet_name=sheet_num)
    col1 = sheet['问题']
    col2 = sheet['相似提问']
    col3 = sheet['同意图提问']
    col4 = sheet['判断问题']
    col5 = sheet['相似判断问题']

    similiar_groups = list()
    for index in range(len(col1)):
        row_set1 = col2bigroup(col1, col2, index, col3)
        # print(col1[index])
        if row_set1:
            similiar_groups.append(list(row_set1))
        row_set2 = col2bigroup(col4, col5, index)
        if row_set2:
            similiar_groups.append(row_set2)
    # print(similiar_groups)
    for group_iter in similiar_groups:
        print(group_iter)
    return similiar_groups


def make_similar_and_different_groups(similiar_groups_in):
    similar_tuples, different_tuples = list(), list()
    for index, similiar_group_iter in enumerate(similiar_groups_in):
        # similar_tuples=list(itertools.chain.from_iterable(similiar_group_iter))
        different_tuples_iter = list()

        similar_tuples_iter = similiar_group_iter
        # similar_tuples_iter=list(itertools.combinations(similiar_group_iter,r=2))
        length_all_cur = len(similar_tuples)
        similar_groups_copy = copy.deepcopy(similiar_groups_in)
        similar_groups_copy.pop(index)
        length_poped = len(similar_groups_copy)

        while len(different_tuples_iter) < 10 * len(similar_tuples_iter):
            index_random = np.random.choice(range(length_poped))
            different_groups_tmp = similar_groups_copy[index_random]
            different_tuples_iter.extend(different_groups_tmp)

        similar_tuples.append(similar_tuples_iter)
        different_tuples.append(different_tuples_iter)

    # for index in range(len(similar_tuples)):
    #     print(similar_tuples[index])
    #     print(different_tuples[index])
    #     print(len(similar_tuples[index]),len(different_tuples[index]))
    return similar_tuples, different_tuples


def make_bigroups(similar_tuples_in, different_tuples_in):
    existed_tuples = list()
    true_tuples, false_tuples = list(), list()
    for index, similiar_tuple_iter in enumerate(similar_tuples_in):
        similar_tuples_iter = list(itertools.combinations(similiar_tuple_iter, r=2))
        sim2diff_tuples_set = set()
        # print('len(similar_tuples_iter)==>', len(similar_tuples_iter))
        counter = 0
        while len(sim2diff_tuples_set) < len(similar_tuples_iter):
            # print('len(sim2diff_tuples_set)==>',len(sim2diff_tuples_set))
            for line_similar_iter in similiar_tuple_iter:
                for line_differenr_iter in different_tuples_in[index]:
                    if (line_similar_iter, line_differenr_iter) not in existed_tuples:
                        sim2diff_tuples_set.add((line_similar_iter, line_differenr_iter))
                        existed_tuples.extend(
                            [(line_similar_iter, line_differenr_iter), (line_differenr_iter, line_similar_iter)])
                    else:
                        pass
            counter += 1
            if counter > 50:
                print(counter)
                break
                # print('else')
        true_tuples.extend(similar_tuples_iter)
        false_tuples.extend(list(sim2diff_tuples_set))
    return {0: true_tuples, 1: false_tuples}


def cut_sent(sent_in, unit='char'):
    # rows_chars=list()
    # for sent_iter in sents_in:
    if unit == 'char':
        chars_iter = list(sent_in)
    else:
        raise ValueError('目前还不支持')

    # rows_chars.append(chars_iter)
    # else:
    sent_space = ' '.join(chars_iter)
    return sent_space


def write(label2tuples_dict, filepath_in):
    # assert os.path.isfile(filepath_in)
    with open(filepath_in, 'w', encoding='utf-8') as fw:
        for label_iter, tuples_iter in label2tuples_dict.items():
            for tuple_iter in tuples_iter:
                # print(tuple_iter)
                sent_cur = cut_sent(tuple_iter[0]) + '\t' + cut_sent(tuple_iter[1]) + '\t' + str(label_iter) + '\n'
                print(sent_cur)
                fw.write(sent_cur)


if __name__ == '__main__':
    filepath = './corpus.txt'
    similiar_groups = read(0)
    similar_tuples, different_tuples = make_similar_and_different_groups(similiar_groups)
    label2tuples_dict = make_bigroups(similar_tuples, different_tuples)
    write(label2tuples_dict, filepath)
