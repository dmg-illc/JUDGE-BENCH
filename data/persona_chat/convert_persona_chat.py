import pandas as pd
import json
import numpy as np
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--prompt') # can be either "long" or "short" 

args = parser.parse_args()

def convert_dataset_full(json_loaded_data):
    ret_dict = {}
    ret_dict['dataset'] = 'Persona Chat (Mehri & Eskenazi, ACL 2020)'
    ret_dict['dataset_url'] = 'http://shikib.com/usr'
    ret_dict['expert_annotator'] = 'true'
    ret_dict['original_prompt'] = True
    ret_dict['annotations'] = [{'metric' : 'engaging',
                                'category': 'graded',
                            'prompt': 'You will be given a conversation between two individuals. You will then be given several potential responses for the next turn in the conversation. These responses all concern an interesting fact, which will be provided as well. Your task is to rate each of the responses on several metrics. The response for one metric should not influence the other metrics. For example, if a response is not understandable or has grammatical errors - you should try to ignore this when considering whether it maintains context or if it is interesting. Please make sure you read and understand these instructions carefully. Feel free to ask if you require clarification. Please keep this document open while reviewing, and refer to it as needed.\nIs the response dull/interesting?\nA score of 1 (dull) means that the response is generic and dull.\nA score of 2 (somewhat interesting) means the response is somewhat interesting and could engage you in the conversation (e.g., an opinion, thought).\nA score of 3 (interesting) means the response is very interesting or presents an interesting fact.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'best': 3,
                            'worst': 1},
                            {'metric' : 'maintains context',
                             'category': 'graded',
                            'prompt': 'You will be given a conversation between two individuals. You will then be given several potential responses for the next turn in the conversation. These responses all concern an interesting fact, which will be provided as well. Your task is to rate each of the responses on several metrics. The response for one metric should not influence the other metrics. For example, if a response is not understandable or has grammatical errors - you should try to ignore this when considering whether it maintains context or if it is interesting. Please make sure you read and understand these instructions carefully. Feel free to ask if you require clarification. Please keep this document open while reviewing, and refer to it as needed.\nDoes the response serve as avalid continuation of the conversation history?\nA score of 1 (no) means that the response drastically changes topic or ignores the conversation history.\nA score of 2 (somewhat) means the response refers to the conversation history in a limited capacity (e.g., in a generic way) and shifts the conversation topic.\nA score of 3 (yes) means the response is on topic and strongly acknowledges the conversation history.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'best': 3,
                            'worst': 1},
                            {'metric' : 'natural',
                             'category': 'graded',
                            'prompt': 'You will be given a conversation between two individuals. You will then be given several potential responses for the next turn in the conversation. These responses all concern an interesting fact, which will be provided as well. Your task is to rate each of the responses on several metrics. The response for one metric should not influence the other metrics. For example, if a response is not understandable or has grammatical errors - you should try to ignore this when considering whether it maintains context or if it is interesting. Please make sure you read and understand these instructions carefully. Feel free to ask if you require clarification. Please keep this document open while reviewing, and refer to it as needed.\nIs the response naturally written?\nA score of 1 (bad) means that the response is unnatural.\nA score of 2 (ok) means the response is strange, but not entirely unnatural.\nA score of 3 (good) means that the response is natural.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'best': 3,
                            'worst': 1},
                            {'metric' : 'overall',
                             'category': 'graded',

                            'prompt': 'You will be given a conversation between two individuals. You will then be given several potential responses for the next turn in the conversation. These responses all concern an interesting fact, which will be provided as well. Your task is to rate each of the responses on several metrics. The response for one metric should not influence the other metrics. For example, if a response is not understandable or has grammatical errors - you should try to ignore this when considering whether it maintains context or if it is interesting. Please make sure you read and understand these instructions carefully. Feel free to ask if you require clarification. Please keep this document open while reviewing, and refer to it as needed.\nWhat is your overall impression of this utterance?\nA score of 1 (very bad). A completely invalid response. It would be difficult to recover the conversation after this.\nA score of 2 (bad). Valid response,but otherwise poor in quality.\nA score of 3 (neutral) means this response is neither good nor bad. This response has no negative qualities, but no positive ones either.\nA score of 4 (good) means this is a good response, but falls short of being perfect because of a key flaw.\nA score of 5 (very good) means this response is good and does not have any strong flaws.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'best': 5,
                            'worst': 1},
                            {'metric' : 'understandable',
                             'category': 'categorical',

                            'prompt': 'You will be given a conversation between two individuals. You will then be given several potential responses for the next turn in the conversation. These responses all concern an interesting fact, which will be provided as well. Your task is to rate each of the responses on several metrics. The response for one metric should not influence the other metrics. For example, if a response is not understandable or has grammatical errors - you should try to ignore this when considering whether it maintains context or if it is interesting. Please make sure you read and understand these instructions carefully. Feel free to ask if you require clarification. Please keep this document open while reviewing, and refer to it as needed.\nIs the response understandable in the context of the history? (Not if its on topic, but for example if it uses pronouns they should make sense)\nA score of 0 (no) means that the response is difficult to understand. You do not know what the person is trying to say.\nA score of 1 (yes) means that the response is understandable. You know what the person is trying to say.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'labels_list' : ['0', '1']},
                            {'metric' : 'uses knowledge',
                             'category': 'categorical',
                            'prompt': 'You will be given a conversation between two individuals. You will then be given several potential responses for the next turn in the conversation. These responses all concern an interesting fact, which will be provided as well. Your task is to rate each of the responses on several metrics. The response for one metric should not influence the other metrics. For example, if a response is not understandable or has grammatical errors - you should try to ignore this when considering whether it maintains context or if it is interesting. Please make sure you read and understand these instructions carefully. Feel free to ask if you require clarification. Please keep this document open while reviewing, and refer to it as needed.\nGiven the interesting fact that the response is conditioned on, how well does the response use the fact?\nA score of 0 (no) means the response does not mention or refer to the fact at all.\nA score of 1 (yes) means the response uses the fact well.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'labels_list': ['0','1']}]
    ret_dict['instances'] = []
    i = 0
    for instance in json_loaded_data:
        for response in instance['responses']:

            inst_dict = {'id': i,
                        'instance': {'context': instance['context'], 
                                    'fact': instance['fact'],
                                    'response': response['response']},
                        'annotations': {'engaging': {'mean_human': np.array(response['Engaging']).mean(),
                                                        'individual_human_scores': response['Engaging']},
                                        'overall': {'mean_human': np.array(response['Overall']).mean(),
                                                        'individual_human_scores': response['Overall']},
                                        'maintains context': {'mean_human': np.array(response['Maintains Context']).mean(),
                                                        'individual_human_scores': response['Maintains Context']},
                                        'natural': {'mean_human': np.array(response['Natural']).mean(),
                                                        'individual_human_scores': response['Natural']},
                                        'understandable': {'majority_human': Counter([str(el) for el in response['Understandable']]).most_common(1)[0][0],
                                                        'individual_human_scores': [str(el) for el in response['Understandable']]},
                                        'uses knowledge': {'majority_human': Counter([str(el) for el in response['Uses Knowledge']]).most_common(1)[0][0],
                                                        'individual_human_scores': [str(el) for el in response['Uses Knowledge']]}}}
        ret_dict['instances'].append(inst_dict)
        i += 1

    return ret_dict


def convert_dataset_reduced(json_loaded_data):
    ret_dict = {}
    ret_dict['dataset'] = 'Topical-Chat (Mehri & Eskenazi, ACL 2020)'
    ret_dict['dataset_url'] = 'http://shikib.com/usr'
    ret_dict['expert_annotator'] = 'true'
    ret_dict['original_prompt'] = True
    ret_dict['annotations'] = [{'metric' : 'engaging',
                                'category': 'graded',
                            'prompt': 'Is the response dull/interesting?\nA score of 1 (dull) means that the response is generic and dull.\nA score of 2 (somewhat interesting) means the response is somewhat interesting and could engage you in the conversation (e.g., an opinion, thought).\nA score of 3 (interesting) means the response is very interesting or presents an interesting fact.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'best': 3,
                            'worst': 1},
                            {'metric' : 'maintains context',
                             'category': 'graded',
                            'prompt': 'Does the response serve as avalid continuation of the conversation history?\nA score of 1 (no) means that the response drastically changes topic or ignores the conversation history.\nA score of 2 (somewhat) means the response refers to the conversation history in a limited capacity (e.g., in a generic way) and shifts the conversation topic.\nA score of 3 (yes) means the response is on topic and strongly acknowledges the conversation history.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'best': 3,
                            'worst': 1},
                            {'metric' : 'natural',
                             'category': 'graded',
                            'prompt': 'Is the response naturally written?\nA score of 1 (bad) means that the response is unnatural.\nA score of 2 (ok) means the response is strange, but not entirely unnatural.\nA score of 3 (good) means that the response is natural.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'best': 3,
                            'worst': 1},
                            {'metric' : 'overall',
                             'category': 'graded',
                            'prompt': 'What is your overall impression of this utterance?\nA score of 1 (very bad). A completely invalid response. It would be difficult to recover the conversation after this.\nA score of 2 (bad). Valid response,but otherwise poor in quality.\nA score of 3 (neutral) means this response is neither good nor bad. This response has no negative qualities, but no positive ones either.\nA score of 4 (good) means this is a good response, but falls short of being perfect because of a key flaw.\nA score of 5 (very good) means this response is good and does not have any strong flaws.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'best': 5,
                            'worst': 1},
                            {'metric' : 'understandable',
                             'category': 'categorical',
                            'prompt': 'Is the response understandable in the context of the history? (Not if its on topic, but for example if it uses pronouns they should make sense)\nA score of 0 (no) means that the response is difficult to understand. You do not know what the person is trying to say.\nA score of 1 (yes) means that the response is understandable. You know what the person is trying to say.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'labels_list':['0', '1']},
                            {'metric' : 'uses knowledge',
                             'category': 'categorical',
                            'prompt': 'Given the interesting fact that the response is conditioned on, how well does the response use the fact?\nA score of 0 (no) means the response does not mention or refer to the fact at all.\nA score of 1 (yes) means the response uses the fact well.\nFact: {{ fact }}\nContext: {{ context }}\nResponse: {{ response }}',
                            'labels_list':['0', '1']}]
    ret_dict['instances'] = []
    i = 0
    for instance in json_loaded_data:
        for response in instance['responses']:

            inst_dict = {'id': i,
                        'instance': {'context': instance['context'], 
                                    'fact': instance['fact'],
                                    'response': response['response']},
                        'annotations': {'engaging': {'mean_human': np.array(response['Engaging']).mean(),
                                                        'individual_human_scores': response['Engaging']},
                                        'overall': {'mean_human': np.array(response['Overall']).mean(),
                                                        'individual_human_scores': response['Overall']},
                                        'maintains context': {'mean_human': np.array(response['Maintains Context']).mean(),
                                                        'individual_human_scores': response['Maintains Context']},
                                        'natural': {'mean_human': np.array(response['Natural']).mean(),
                                                        'individual_human_scores': response['Natural']},
                                        'understandable': {'majority_human': Counter([str(el) for el in response['Understandable']]).most_common(1)[0][0],
                                                        'individual_human_scores': [str(el) for el in response['Understandable']]},
                                        'uses knowledge': {'majority_human': Counter([str(el) for el in response['Uses Knowledge']]).most_common(1)[0][0],
                                                        'individual_human_scores': [str(el) for el in response['Uses Knowledge']]}}}
        ret_dict['instances'].append(inst_dict)
        i += 1

    return ret_dict


if __name__ == '__main__':
    with open('original_data/persona_chat.json') as data_file:
        data_dict = json.load(data_file)
    if args.prompt == "long":
        ret_dict = convert_dataset_full(data_dict)
        with open(f'persona_chat_long.json', 'w') as outfile:
            json.dump(ret_dict, outfile)
    
    elif args.prompt == "short":
        ret_dict = convert_dataset_reduced(data_dict)
        with open(f'persona_chat_short.json', 'w') as outfile:
            json.dump(ret_dict, outfile)