import json

from semantic_subjectivity import semantic_subjectivity_entropy, centroid_vector, load_vectors, get_embedding
from utils import get_answers_from_SS, get_gtans_count
from tqdm import tqdm
from args import get_args
from collections import Counter

"""
Shailza jolly, DFKI & TU Kaiserslautern, March 2020
"""

def open_validation_files(annotation_file_name, question_file_name):

    annotations = json.load(open(annotation_file_name,'r'))['annotations']
    questions = json.load(open(question_file_name,'r'))['questions']

    print("Annotations length: ", len(annotations))
    print("Questions length: ", len(questions))

    return annotations, questions

def create_dicts(annotations, DataFlag="VQA"):

    if DataFlag=="VQA":
        # ques_id -> ground truth answer
        ques2gt = {x['question_id']: x['answers'] for x in annotations}  # for VQA
        outs=ques2gt
    elif DataFlag=="VizWiz":
        #dictionary: keys-> image_name ; values-> ground truth answer
        img2gt = {x['image']:[x['question'], x['answers']] for x in annotations} #for VizWiz
        outs=img2gt
    return outs

def get_image(idx, img2gt):

    '''
    The function is designed to create dictionary for VizWiz data
    :param idx:
    :param img2gt:
    :return image name:
    '''

    imdir = 'VizWiz_%s_%012d.jpg'
    img_train = imdir % ('train', idx)
    img_val = imdir % ('val', idx)

    if img_train in img2gt:
        return img_train
    if img_val in img2gt:
        return img_val
    raise Exception

#print(get_image(29037, img2gt))

def get_qid_splits_SeS(word2vec, ques_id2gt_ans, questions, annotations):

    easy_samples_ques_ids = []
    hard_samples_ques_ids = []
    most_easy_samples_ques_ids = []

    easy = {}
    hard = {}
    most_easy = {}

    for ques, annos in tqdm(zip(questions, annotations)):

        gt_anss = ques_id2gt_ans[ques['question_id']]
        gt_ans = [x['answer'] for x in gt_anss]
        gt_ans = [x.lower() for x in gt_ans]

        gt_ans_count = Counter(gt_ans)

        only_ans = [] #for centroid since centroid is computed using unique answers.
        for ans, freq in gt_ans_count.items():
            only_ans.append(ans)
        centroid = centroid_vector(only_ans, word2vec)

        SeS_score = semantic_subjectivity_entropy(gt_ans, centroid, word2vec)

        if 0 <= SeS_score < 0.5:
            hard_samples_ques_ids.append(annos['question_id'])
            hard[annos['question_id']] = SeS_score
        elif 0.5 <= SeS_score < 1:
            easy_samples_ques_ids.append(annos['question_id'])
            easy[annos['question_id']] = SeS_score
        elif SeS_score == 1:
            most_easy_samples_ques_ids.append(annos['question_id'])
            most_easy[annos['question_id']] = SeS_score

    print("Number of hard samples: ", len(hard_samples_ques_ids))  # 135184
    print ("Number of easy samples: ", len(easy_samples_ques_ids)) #79170
    print ("Number of most easy samples: ", len(most_easy_samples_ques_ids))

    # json.dump(easy, open('WD_easy_ids.json','w+'))
    # json.dump(hard, open('WD_hard_ids.json','w+'))
    # json.dump(most_easy, open('WD_most_easy_ids.json','w+'))

    return

def get_qid_splits_SeS_vizwiz(word2vec, ques_id2gt_ans, annotations):

    easy_samples_ques_ids = []
    hard_samples_ques_ids = []
    most_easy_samples_ques_ids = []

    easy = {}
    hard = {}
    most_easy = {}

    for ques_id, gt_answers in ques_id2gt_ans.items():

        gt_ans_count = Counter(gt_answers)

        only_ans = [] #for centroid since centroid is computed using unique answers.
        for ans, freq in gt_ans_count.items():
            only_ans.append(ans)

        centroid = centroid_vector(only_ans, word2vec)

        SeS_score = semantic_subjectivity_entropy(gt_answers, centroid, word2vec)

        if 0 <= SeS_score < 0.5:
            hard_samples_ques_ids.append(ques_id)
            hard[ques_id] = SeS_score
        elif 0.5 <= SeS_score < 1:
            easy_samples_ques_ids.append(ques_id)
            easy[ques_id] = SeS_score
        elif SeS_score == 1:
            most_easy_samples_ques_ids.append(ques_id)
            most_easy[ques_id] = SeS_score

    print("Number of hard samples: ", len(hard_samples_ques_ids))
    print ("Number of easy samples: ", len(easy_samples_ques_ids))
    print ("Number of most easy samples: ", len(most_easy_samples_ques_ids))

    # json.dump(easy_samples_ques_ids, open('easy_ids_vizwiz.json','w+'))
    # json.dump(hard_samples_ques_ids, open('hard_ids_vizwiz.json','w+'))
    # json.dump(most_easy_samples_ques_ids, open('most_easy_ids_vizwiz.json','w+'))

    return

def add_SeS_annotations_vizwiz(annotations):

    '''
    :param annotations:
    :return: Update current annotations with SeS scores in it
    '''
    annos_ses = []
    for annos in annotations:

        gt_answers = ques_id2gt_ans[annos['question_id']]
        gt_ans_count = Counter(gt_answers)

        only_ans = []  # for centroid since centroid is computed using unique answers.
        for ans, freq in gt_ans_count.items():
            only_ans.append(ans)

        centroid = centroid_vector(only_ans, word2vec)

        SeS_score = semantic_subjectivity_entropy(gt_answers, centroid, word2vec)
        annos['ses_score'] = SeS_score

        annos_ses.append(annos)

    # json.dump(annos_ses, open('vizwiz/tr_annos_ses_VZ.json','w+'))

def add_SeS_annotations_vqa(word2vec, ques_id2gt_ans, questions, annotations):

    hard_samples_ques_ids = []
    easy_samples_ques_ids = []
    most_easy_samples_ques_ids = []
    train_ses_annos =  []

    for ques, annos in tqdm(zip(questions, annotations)):

        gt_anss = ques_id2gt_ans[ques['question_id']]
        gt_ans = [x['answer'] for x in gt_anss]
        gt_ans = [x.lower() for x in gt_ans]

        gt_ans_count = Counter(gt_ans)

        only_ans = [] #for centroid since centroid is computed using unique answers.
        for ans, freq in gt_ans_count.items():
            only_ans.append(ans)
        centroid = centroid_vector(only_ans, word2vec)

        SeS_score = semantic_subjectivity_entropy(gt_ans, centroid, word2vec)

        annos['SeS_score'] =  SeS_score
        train_ses_annos.append(annos)

        if 0 <= SeS_score < 0.5:
            hard_samples_ques_ids.append(annos['question_id'])
        elif 0.5 <= SeS_score < 1:
            easy_samples_ques_ids.append(annos['question_id'])
        elif SeS_score == 1:
            most_easy_samples_ques_ids.append(annos['question_id'])

    print ("Number of hard samples in training: ", len(hard_samples_ques_ids))
    print ("Number of easy samples in training: ", len(easy_samples_ques_ids))
    print ("Number of most easy samples in training: ", len(most_easy_samples_ques_ids))

    return train_ses_annos


if __name__ == '__main__':

    ease_args = get_args()
    word2vec = load_vectors(ease_args.word2vec)
    print("Word2Vec Loaded!!!!")

    if ease_args.data_name=="VQA":
        print("Running EaSe for: ", ease_args.data_name)
        if ease_args.data_split=="train":
            print("Computing splits for: ", ease_args.data_split)
            annotations, questions = open_validation_files(ease_args.annotation_path_tr, ease_args.question_path_tr)
        else:
            print("Computing splits for: ", ease_args.data_split)
            annotations, questions = open_validation_files(ease_args.annotation_path_val, ease_args.question_path_val)

        ques_id2gt_ans = create_dicts(annotations, ease_args.data_name)  # VQA
        print("Dictionary created!!!!")

        get_qid_splits_SeS(word2vec, ques_id2gt_ans, questions, annotations)

        '''
        To create new file with SeS score computed by our metric.
        ses_annos = add_SeS_annotations_vqa(word2vec, ques_id2gt_ans, questions, annotations)
        orig_annos = json.load(open(annotations, 'r'))
        orig_annos['annotations'] = ses_annos
        json.dump(orig_annos, open('orig_annos_ses.json', 'w+'))
        '''


    elif ease_args.data_name=="VizWiz":
        print("Running EaSe for: ", ease_args.data_name)
        validation_annotation_vizwiz = json.load(open('../lxmert/data/vizwiz_data/minival.json','r'))#VizWiz
        ques_id2gt_ans = get_answers_from_SS(validation_annotation_vizwiz)#VizWiz
        print("Dictionary created!!!!")

        get_qid_splits_SeS_vizwiz(word2vec, ques_id2gt_ans)
