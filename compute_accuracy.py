import json
import numpy as np
import random
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_scores_file', type=str, default='new_predfiles/preds_wholeVQA2.0')
    parser.add_argument('--Id_Directory', type=str, default='VQA2.0_ids/entropy/E_')
    parser.add_argument('--model_type', type=str, default='BUTD')
    parser.add_argument('--repeat_rand', type=int, default=10)
    parser.add_argument('--type_random_exp', type=str, default='eval', help = 'If we want to compute accuracy per split for model trained on random TH or random evalution at test only.')
    parser.add_argument('--folder_rand_tr', type=str, default='', help='folder with prediction files when model is trained with random splits of data with size of data equal to hard split.')

    args = parser.parse_args()
    return args

def create_dict(model_scores_dict):

    #The function works for LXMERT's prediction file
    qid2score = {}

    for entry in model_scores_dict:

        qid2score[entry['question_id']] = entry['answer']

    return qid2score

def compute_score_wholeval(model_scores):
    eval_score = []
    
    for qid, score in model_scores.items():
        eval_score.append(score)

    print("Evaluation score VQA2.0: ", np.mean(eval_score), len(eval_score))

    return eval_score


def compute_score(ids, model_scores, score_list, model_type):

    for i in ids:
        if model_type == "LXMERT":
            i = int(i)

        elif model_type == "BUTD":
            i = str(i)

        if i in model_scores.keys():
            score_list.append(model_scores[i])

    return np.mean(score_list)

def random_splits_eval(eval_score, repeat_rand, n_th, n_bh, n_e):

    print("No. of TH samples: ", n_th)
    print("No. of BH samples: ", n_bh)
    print("No. of E samples: ", n_e)

    O= []
    TH = []
    BH = []
    E = []

    '''
    The function takes eval score given by compute_score_BUTD (scores for whole validation data). It randomly select three subsets with equal sizes as that of TH, BH, E. 
    The process is repeated n times and the results are average
    '''

    for i in range(repeat_rand):

        O  += [np.mean(eval_score)]
        TH += [np.mean([j for j in random.sample(eval_score, n_th)])]
        BH += [np.mean([j for j in random.sample(eval_score, n_bh)])]
        E  += [np.mean([j for j in random.sample(eval_score, n_e)])]

    print("-----------------")
    print("O: ", np.mean(O), len(O))
    print("-----------------")
    print("Evaluation on three random  splits with sizes == TH/BH/E")
    print("TH: ", np.mean(TH), len(TH))
    print("BH: ", np.mean(BH), len(BH))
    print("E: ", np.mean(E), len(E))


    return

def random_splits_train(folder_rand_tr, model_type, tophard_ids, bothard_ids, easy_ids):

    O = []
    TH = []
    BH = []
    E = []

    for filename in os.listdir(folder_rand_tr):

        print("filename:", os.path.join(folder_rand_tr, filename))
        model_scores = json.load(open(os.path.join(folder_rand_tr, filename), 'r'))

        O.append(np.mean(compute_score_wholeval(model_scores)))
        TH.append(compute_score(tophard_ids, model_scores, TH, model_type))
        BH.append(compute_score(bothard_ids, model_scores, BH, model_type))
        E.append(compute_score(easy_ids, model_scores, E, model_type))

    print("Score for O: ", np.mean(O))
    print("Score for TH: ", np.mean(TH))
    print("Score for BH: ", np.mean(BH))
    print("Score for E: ", np.mean(E))

    return

if __name__ == '__main__':

    args = parse_args()

    model_scores_file = args.model_scores_file
    Id_Directory = args.Id_Directory
    model_type = args.model_type
    repeat_rand = args.repeat_rand

    print("Loading Ids!!")

    bothard_ids = json.load(open(Id_Directory + 'easy_ids.json', 'r'))
    tophard_ids = json.load(open(Id_Directory + 'hard_ids.json', 'r'))
    easy_ids = json.load(open(Id_Directory + 'most_easy_ids.json', 'r'))

    bothard_score_list = []
    tophard_score_list = []
    easy_score_list = []

    print("Ids loaded !!!!")

    if model_type == "BUTD":

        model_scores = json.load(open(model_scores_file, 'r'))
        eval_score = compute_score_wholeval(model_scores)

    elif model_type == "LXMERT":

        model_scores_dict = json.load(open(model_scores_file,'r'))
        model_scores = create_dict(model_scores_dict)
        eval_score = compute_score_wholeval(model_scores)

    if args.type_random_exp =="eval":
        random_splits_eval(eval_score, repeat_rand, len(tophard_ids), len(bothard_ids), len(easy_ids))

    elif args.type_random_exp =="train":
        random_splits_train(args.folder_rand_tr, model_type, tophard_ids, bothard_ids, easy_ids)

    elif args.type_random_exp =="norand":

        score = compute_score(tophard_ids, model_scores, tophard_score_list, model_type)
        print("Hard samples length: ", len(tophard_ids))
        print("Accuracy score for hard VQA2.0 samples: ", score)

        print("-----------------")

        score = compute_score(bothard_ids, model_scores, bothard_score_list, model_type)
        print("Easy samples length: ", len(bothard_ids) )
        print ("Accuracy score for easy VQA2.0 samples: ", score)

        print("-----------------")

        score = compute_score(easy_ids, model_scores, easy_score_list, model_type)
        print("Most Easy samples length: ", len(easy_ids) )
        print ("Accuracy score for most easy VQA2.0 samples: ", score)

    print("Done!!")