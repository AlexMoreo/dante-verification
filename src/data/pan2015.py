import itertools
import os
from os.path import join, isdir

PATH_PAN2015 = '../pan2015'
PAN2015_TRAIN = 'pan15-authorship-verification-training-dataset-2015-04-19'
PAN2015_TEST  = 'pan15-authorship-verification-test-dataset2-2015-04-19'

class Pan2015:
    def __init__(self, problem, solution):
        self.problem = problem
        self.solution = solution

def fetch_PAN2015(corpus, lang, base_path = PATH_PAN2015):
    assert corpus in ['train','test'],'unexpected corpus request'

    corpus_path = join(base_path, PAN2015_TRAIN if corpus=='train' else PAN2015_TEST)

    print(corpus_path)
    request = {}
    truth = {}
    for dir in os.listdir(corpus_path):
        dir_path = join(corpus_path,dir)
        if isdir(dir_path) and lang in dir:
            truth = [x.split() for x in open(join(dir_path,'truth.txt'), 'rt').readlines()]
            truth = {problem:1 if decision == 'Y' else 0 for problem,decision in truth}
            for problem_name in os.listdir(dir_path):
                problem_dir = join(dir_path,problem_name)
                if isdir(problem_dir):
                    request[problem_name] = {}
                    request[problem_name]['known'] = []
                    for doc_name in os.listdir(problem_dir):
                        doc_path = join(problem_dir,doc_name)
                        if 'unknown.txt' == doc_name:
                            request[problem_name]['unknown'] = open(doc_path,'rt').read()
                        else:
                            request[problem_name]['known'].append(open(doc_path, 'rt').read())

    return Pan2015(request, truth)

def TaskGenerator(request_dict):
    pan_problems = request_dict.problem
    problems = sorted(pan_problems.keys())
    for i,problem_i in enumerate(problems):
        positives = pan_problems[problem_i]['known']
        negatives = list(itertools.chain.from_iterable([pan_problems[problem_j]['known'] for j,problem_j in enumerate(problems) if i!=j]))
        test = pan_problems[problem_i]['unknown']
        yield problem_i,positives,negatives,test,request_dict.solution[problem_i]



