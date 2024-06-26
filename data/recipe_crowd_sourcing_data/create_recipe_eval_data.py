import json
import os.path
from csv import DictReader
from copy import deepcopy


def create_recipe_eval_data(participants_data_csv,
                            recipes_directory,
                            question_file,
                            output_file,
                            add_overall_instr: bool = False):
    """

    :param participants_data_csv: file with all data from the human crowd-sourcing study (\t separated)
    :param recipes_directory: path to the directory with all recipe texts,
                              recipe texts must be organized into subdirectories named after the experimental condition
    :param question_file: .json file with the questions used to collect the ratings and the rating scale boundaries
    :param output_file: path to the outputfile to generate
    :param add_overall_instr: whether to add the overall_instructions value to the prompt
    :return:
    """

    # Read in questions / instructions from crowd-sourcing and convert into prompt field
    with open(question_file, 'r') as qf:
        question_data = json.load(qf)
        general_instruction = question_data['overall_instructions']
        for quest_dict in question_data['individual_instructions']:
            task_instr = quest_dict.pop('task')
            statement = quest_dict.pop('statement')
            prompt = '{{ instance }}' + f'\n\n{task_instr}\n\n{statement}\n\n'
            if add_overall_instr:
                prompt = f'{general_instruction}\n\n{prompt}'
            quest_dict['prompt'] = prompt

    # Read in all data points from the human evaluation study
    instances = dict()

    with open(participants_data_csv, 'r') as data_f:
        reader = DictReader(data_f, delimiter='\t')

        eval_categories = ["grammar", "fluency", "verbosity", "structure", "success", "overall"]
        for row in reader:
            recipe_instance = row['recipeid']
            condition = row['condition']
            recipe_id = f'{recipe_instance}_{condition}'

            if recipe_id not in instances.keys():

                eval_factor_dict = {"mean_human": None, "individual_human_scores": []}

                annotations = {
                    "grammar": deepcopy(eval_factor_dict),
                    "fluency": deepcopy(eval_factor_dict),
                    "verbosity":deepcopy(eval_factor_dict),
                    "structure": deepcopy(eval_factor_dict),
                    "success": deepcopy(eval_factor_dict),
                    "overall": deepcopy(eval_factor_dict)
                }
                instances[recipe_id] = {
                    "id": recipe_id,
                    "recipe_name": recipe_instance,
                    "condition": condition,
                    "instance": "",
                    "annotations": deepcopy(annotations)
                }

            for eval_c in eval_categories:
                instances[recipe_id]['annotations'][eval_c]['individual_human_scores'].append(int(row[eval_c]))

    # Read the recipe texts for each instance and compute average score per evaluation dimension
    for inst_id, inst_dict in instances.items():

        # Get text
        recipe_text = get_recipe_condition_text(recipe_name=inst_dict["recipe_name"],
                                                condition=inst_dict["condition"],
                                                recipe_dir=recipes_directory)
        inst_dict["instance"] = recipe_text

        for category, category_dict in inst_dict['annotations'].items():
            category_dict["mean_human"] = round(sum(category_dict["individual_human_scores"]) / len(category_dict["individual_human_scores"]), 3)

        inst_dict.pop('recipe_name')
        inst_dict.pop('condition')

    complete_data = {
        "dataset": "Rewritten cooking recipes (Stein et al., DMR Workshop 2024",
        "dataset_url": "https://github.com/interactive-cookbook/recipe-generation",
        "annotations": question_data['individual_instructions'],
        "instances": [inst for inst in instances.values()]
    }

    with open(output_file, 'w', encoding='utf-8') as o:
        json.dump(complete_data, o, indent=4)


def get_recipe_condition_text(recipe_name, condition, recipe_dir) -> str:
    """
    Reads in the content from the file recipe_dir/condition/[...]recipe_name[...].txt
    :param recipe_name:
    :param condition:
    :param recipe_dir:
    :return: the content of the file
    """
    texts_dir = os.path.join(recipe_dir, condition)
    for file in os.listdir(texts_dir):
        if f'{recipe_name}_' in str(file) or f'{recipe_name}.' in str(file):
            with open(os.path.join(texts_dir, file), 'r') as recipe:
                recipe_text = recipe.read()
                return recipe_text

    raise FileNotFoundError(f'File for {recipe_name} and {condition} not found')


if __name__=='__main__':
    create_recipe_eval_data(participants_data_csv="./materials/cleaned_data_anonymized.csv",
                            recipes_directory="./materials/recipe_texts",
                            question_file="./materials/questions.json",
                            output_file="./meta_evaluation_recipes.json")

    create_recipe_eval_data(participants_data_csv="./materials/cleaned_data_anonymized.csv",
                            recipes_directory="./materials/recipe_texts",
                            question_file="./materials/questions.json",
                            output_file="./meta_evaluation_recipes_long_prompt.json",
                            add_overall_instr=True)

