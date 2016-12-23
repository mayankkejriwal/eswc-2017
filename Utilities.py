import codecs
import json
import numpy as np
import warnings
import random
from sklearn.preprocessing import normalize
import re
# import math
from scipy.spatial.distance import cosine
from scipy.linalg import norm
import scipy.stats
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import normalize


def abs_cosine_sim(vector1, vector2):
    if norm(vector1) == 0.0 or norm(vector2) == 0.0:
        return 0.0
    else:
        return 1.0 - cosine(vector1, vector2)

def get_random_subset_of_file(input_file, output_file, subset_size=5000, full_set_size=800000):
    """
    Designed to take a partition file and return a small partition file.
    :param input_file:
    :param output_file:
    :param subset_size:
    :param full_set_size:
    :return:
    """
    out = codecs.open(output_file, 'w', 'utf-8')
    segment_size = int(full_set_size/subset_size)
    count = 1
    with codecs.open(input_file, 'r', 'utf-8') as f:
        for line in f:
            k = random.randint(1, full_set_size)
            if k%segment_size == 0:
                out.write(line)
            count += 1
            if count % 1000 == 0:
                print 'processing...',count
    out.close()

def get_random_subset_of_files(input_files, output_files, subset_size=100, full_set_size=4966):
    """
    Designed to take a partition file and return a small partition file.
    :param input_file:
    :param output_file:
    :param subset_size:
    :param full_set_size:
    :return:
    """
    out = [codecs.open(output_files[i], 'w', 'utf-8') for i in range(len(output_files))]
    segment_size = int(full_set_size/subset_size)
    count = 0
    lines = [list() for i in range(len(input_files))]
    for i in range(len(input_files)):
        with codecs.open(input_files[i], 'r', 'utf-8') as f:
            for line in f:
                lines[i].append(line)

    for i in range(full_set_size):
        k = random.randint(1, full_set_size)
        if k%segment_size == 0:
            for j in range(len(output_files)):
                out[j].write(lines[j][count])
        count += 1
        if count % 1000 == 0:
            print 'processing...',count
    for o in out:
        o.close()


def mean_confidence_interval(data, confidence=0.95):
    """
    Returns mean and half-interval length
    :param data:
    :param confidence:
    :return:
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def reverse_dict(dictionary):
    """

    Turn keys into (lists of) values, and values into keys. Values must originally be primitive.
    :param dictionary:
    :return: Another dictionary
    """
    new_dict = dict()
    for k, v in dictionary.items():
        if v not in new_dict:
            new_dict[v] = list()
        new_dict[v].append(k)
    return new_dict

def l2_normalize(list_of_nums):
    """
    l2 normalize a vector. original vector is unchanged. Meant to be used as a process_embedding function
    in Classification.construct_dbpedia_multi_file
    :param list_of_nums: a list of numbers.
    :return:
    """
    k = list()
    k.append(list_of_nums)
    warnings.filterwarnings("ignore")
    return list(normalize(np.matrix(k))[0])

def l2_normalize_matrix(matrix):
    warnings.filterwarnings("ignore")
    return normalize(matrix)


def extract_top_k(scored_results_dict, k, disable_k=False, reverse=True):
    """

    :param scored_results_dict: a score always references a list
    :param k: Max. size of returned list.
    :param disable_k: ignore k, and sort the list by k
    :param reverse: if reverse is true, the top k will be the highest scoring k. If reverse is false,
    top k will be the lowest scoring k.
    :return: a dict
    """
    count = 0
    # print k
    results = list()
    result_scores = list()
    scores = scored_results_dict.keys()
    scores.sort(reverse=reverse)
    for score in scores:
        # print score
        # print count
        if count >= k and not disable_k:
            break
        vals = scored_results_dict[score]
        if disable_k:
            results += vals
            result_scores += ([score]*len(vals))
            continue
        if count + len(vals) <= k:
            results += vals
            result_scores += ([score] * len(vals))
            count = len(results)
        else:
            results += vals[0: k - count]
            result_scores += ([score] * len(vals[0: k - count]))
            count = len(results)
    # print results[0]
    if len(results) != len(result_scores):
        raise Exception
    answer = dict()
    for i in range(len(results)):
        answer[results[i]] = result_scores[i]
    return answer

def read_in_RI_embeddings(embeddings_file):
    unigram_embeddings = dict()
    with codecs.open(embeddings_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            for k, v in obj.items():
                unigram_embeddings[k] = v
    return unigram_embeddings


def deduplicate_list_in_order(list_of_terms):
    """
    Deduplicates the list 'in order' by removing all non-first occurrences. Returns a new list.
    :param list_of_terms:
    :return: a new deduplicated list
    """
    allowed = set(list_of_terms)
    new_list = list()
    for element in list_of_terms:
        if element in allowed:
            new_list.append(element)
            allowed.remove(element)
    # print new_city
    return new_list

def convert_string_to_float_list(string):
    return [float(i) for i in re.split(', ', string[1:-1])]

def print_lorelei_topics(rwp_input_file, output_file=None, topic_folder=None):
    topics = dict() # we'll double/triple/... count if we have to. Multiple topics could be in a single doc.
    with codecs.open(rwp_input_file, 'r', 'utf-8') as f:
        for line in f:
            obj = json.loads(line)
            k = list()
            if 'loreleiJSONMapping' in obj and 'topics' in obj['loreleiJSONMapping']:
                if type(obj['loreleiJSONMapping']['topics'])!=list:
                    k.append(obj['loreleiJSONMapping']['topics'])
                else:
                    k = obj['loreleiJSONMapping']['topics']
            if k:
                for item in k:
                    if item not in topics:
                        topics[item] = 0
                    topics[item] += 1
    # print topics
    if output_file:
        out = codecs.open(output_file, 'w', 'utf-8')
        json.dump(topics, out)
        out.close()

    if topic_folder:
        for item in topics.keys():
            out = codecs.open(topic_folder+item+'.txt', 'w', 'utf-8')
            out.close()

def is_correct_jlines(jlines_file):
    """
    Checks whether the file is a proper jlines file.
    :param jlines_file:
    :return:
    """
    with codecs.open(jlines_file, 'r', 'utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                print 'somethings wrong in line'
                print line
                return False
    return True

# data_path = '/Users/mayankkejriwal/datasets/companies/'
# print is_correct_jlines(data_path+'result.json')
# rwp_path = '/Users/mayankkejriwal/datasets/lorelei/RWP/reliefWebProcessed-prepped/'
# read_in_DB2Vec_embeddings(DB2Vec_file=None, prefix_file=word2vec_path+'prefix.txt')
# print_lorelei_topics(rwp_path+'nonCondensed.jl',rwp_path+'topics_doc_counts.json', rwp_path+'topic-descriptions/')