import numpy as np
import io
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance as wd
import re
import scipy.stats


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        vector = tokens[1:]
        if len(vector) < 300:
            print(word)
        data[word] = list(map(float, vector))

    return data


def centroid_vector(words, word2vec):

    '''
    Calculates the mean vector of the given words.
    :param words(ist of words):
    :param word2vec:
    :return array for the mean vector.:
    '''

    emb_list = []
    for word in words:

        emb = get_embedding(word, word2vec)#function that returns embedding of the word
        emb_list.append(emb)

    avg_vec = np.mean(emb_list, axis=0)

    return avg_vec


def get_embedding(input_str, word2vec):

    '''
    Generates an embedding of the input string.
    :param input_str ( a string with one or more words):
    :return An array of the input string:
    '''

    str_emb = np.zeros((1, 300))
    words = re.split("\s|; |, |' ", input_str)

    for word in words:
        if word in word2vec:
            word_emb = np.array(word2vec.get(word))
            str_emb = np.add(str_emb, word_emb)

    return str_emb

def compute_similarity(word2vec, word1, word2, word2_embed=False):

    '''
    Calculates the similarity between 2 vectors.
    :param word1 (a string of one or more words):
    :param word2 (a string of one or more words OR an embedding vector:
    :param word2_emb (if word2 is an embedding or or not):
    :return similarity score in range [0,1]:
    '''

    word1_emb = get_embedding(word1, word2vec)

    if word2_embed:
        word2_emb = word2
    else:
        word2_emb = get_embedding(word2, word2vec)

    if (np.sum(word1_emb) == 0) or (np.sum(word2_emb) == 0):
        cos_sim = 0
    else:
        cos_sim = cosine_similarity(word1_emb, word2_emb)[0, 0]

    if cos_sim < 0:
        cos_sim = 0

    return cos_sim


def semantic_subjectivity_entropy(gt_ans, centroid, word2vec):

    '''
    The score is assigned between 0 to 1. Highly subjective sample (all answers with
    1 frequency) get less score.
    Least subjective sample (all annotators agree on one answer.
    Score high (Most reliable))
    '''

    less_sim_ans = []
    hig_sim_ans = []
    less_sim_names = []
    hig_sim_names = []

    ans_count = Counter(gt_ans)
    max_frequency = ans_count.most_common(1)[0][1]#frequency for max gt answer
    max_gt_ans = ans_count.most_common(1)[0][0]#gt answer name with max frequency

    '''
    To handle cases when yes and no come in same set when yes has highest frequency.
    After coming into same set SeS score jumps to 1.
    '''

    if max_gt_ans == 'yes' and 'no' in ans_count:
        d = 1 - round(scipy.stats.entropy(list(ans_count.values())) / 2.302, 3)

    else:
        max_occuring_sim = compute_similarity(word2vec, max_gt_ans, centroid,
                                          word2_embed=True)
        # computes sim of max GT ans with centroid
        dynamic_threshold = max_occuring_sim - 0.0001

        for ans, count in ans_count.items():
            sim = compute_similarity(word2vec,  ans, centroid, word2_embed=True)

            if count == max_frequency and sim < max_occuring_sim:
                max_occuring_sim = sim

            if sim < dynamic_threshold:
                less_sim_ans.append(count)
                less_sim_names.append(ans)
            else:
                hig_sim_ans.append(count)
                hig_sim_names.append(ans)

        if len(hig_sim_ans) == 0:

            ans_distribution = list(ans_count.values())
            d = 1 - round(scipy.stats.entropy(ans_distribution) / 2.302, 3)

        else:
            new_distribution = less_sim_ans + [sum(hig_sim_ans)]
            d = 1 - round(scipy.stats.entropy(new_distribution) / 2.302, 3)

    return d
