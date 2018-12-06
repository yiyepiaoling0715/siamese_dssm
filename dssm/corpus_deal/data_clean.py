import os
import pandas as pd
import numpy as np
import collections
import itertools
import copy
from corpus_deal.excel_funcs import pattern_deal, clean_inner_punc_linebreak_and_bracket, clean_all_punctuaitons
import jieba
import random



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


def make_bigroups(similar_tuples_in, different_tuples_in,n_diff=4):
    sim_and_diff_list = list()
    for index, similiar_tuple_iter in enumerate(similar_tuples_in):
        similar_tuples_iter = list(itertools.combinations(similiar_tuple_iter, r=2))
        # sim_and_diff_iter=list()
        for similar_tuple_iter in similar_tuples_iter:
            similar_tuple_iter=list(similar_tuple_iter)
            # print('similar_tuple_iter==>',similar_tuple_iter)
            assert isinstance(similar_tuple_iter,(list,tuple))
            random_nums=[random.randint(0,len(different_tuples_in[index])-1) for _ in range(n_diff)]
            # print(len(different_tuples_in),random_nums)
            different_strs=[different_tuples_in[index][index_iter] for index_iter in random_nums]
            print('different_strs==>',different_strs)
            similar_tuple_iter.extend(different_strs)
        # sim_and_diff_list.extend(similar_tuples_iter)
            sim_and_diff_list.append(similar_tuple_iter)
    return sim_and_diff_list

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

def write(sim_and_diff_list,write_file):
    with open(write_file,'w',encoding='utf-8') as fw:
        for sim_and_diff_iter in sim_and_diff_list:
            cut_iter=list()
            for sent_iter in sim_and_diff_iter:
                cuts=cut_sent(sent_iter)
                # cut_iter.append(' '.join(cuts))
                cut_iter.append(cuts)
            print('cut_iter==>',cut_iter)
            fw.write('\t'.join(cut_iter)+'\n')


if __name__=='__main__':
    xlsxpath = './raw.xlsx'
    write_file='./corpus.txt'
    data=read(0)
    similiar_groups = read(0)
    similar_tuples, different_tuples = make_similar_and_different_groups(similiar_groups)
    label2tuples_dict = make_bigroups(similar_tuples, different_tuples)
    write(label2tuples_dict, write_file)