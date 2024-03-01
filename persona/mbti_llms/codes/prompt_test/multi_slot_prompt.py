import pathlib
import json
from tqdm import tqdm
import random
import sys
import itertools
import os

# five random seed
random_seed = [2023, 8231, 244104, 87091205, 213213]

en_persona_description = {
    'E': ["**Extroverted**",
          """**Extraversion** refers to the act or state of being energized by the world outside the self. Extraverts enjoy socializing and tend to be more enthusiastic, assertive, talkative, and animated. They enjoy time spent with more people and find it less rewarding to spend time alone. They are Initiating, Expressive, Gregarious, Active and Enthusiastic."""],
    'I': ["**Introverted**",
          """**Introversion**, on the contrary, is the state of being predominately concerned with one’s inner world. Introverts prefer self-reflection to social interactions. They also prefer to observe before participating in an activity. Introverts tend to more quiet, ‘peaceful’, and reserved. Introverts *prefer* individual activities over social ones—this. They are Receiving, Contained, Intimate, Reflective and Quiet."""],
    'S': ["**Sensing**",
          """**Sensing** refers to processing data through the five senses. Sensing people focus on the present and prefer to “learn by doing” rather than thinking it through. They are concrete thinkers recognize details. They are more energized by the practical use of an object/idea rather than the theory behind it. They are Concrete, Realistic, Practical, Experiential and Traditional."""],
    'N': ["**Intuition**",
          """**Intuition** refers to how people process data. Intuitive people are keener to the meaning and patterns behind information. Intuitive people are more focused on how the present would affect the future. They are readily able to grasp different possibilities and abstract concepts. They easily see the big picture rather than the details. They are Abstract, Imaginative, Conceptual, Theoretical and Original."""],
    'T': ["**Thinking**",
          """**Thinking** refers to how people make decisions. Thinking people are objective and base their decision on hard logic and facts. They tend to analyze the pros and cons of a situation and notice inconsistencies. They prefer to be task-oriented and fair. They are Logical, Reasonable, Questioning, Critical and Tough."""],
    'F': ["**Feeling**",
          """**Feeling** people are more subjective. They base their decisions on principles and personal values. When making decisions, they consider other people’s feelings and take it in account. It is in their best mind to maintain harmony among a group. They are more governed by their heart. They are Empathetic, Compassionate, Accommodating, Accepting and Tender."""],
    'J': ["**Judging**",
          """**Judging** refers to how people outwardly display themselves when making decisions. Judging people have a tendency to be organized and prompt. They like order prefer outlined schedules to working extemporaneously. They prefer plans. They find the outcome more rewarding than the process of creating something. Judging people seek closure. They are Systematic, Planful, Early Starting, Scheduled and Methodical."""],
    'P': ["**Perceiving**",
          """**Perceiving** people prefer flexibility and live their life with spontaneity. They act following their mind. They dislike structure and prefer to adapt to new situations rather than plan for it. They tend to be open to new options, experiences and emergency events. While working on a project, they enjoy the process more than the outcome. They are Casual, Open-Ended, Pressure-Prompted, Spontaneous and Emergent."""],
}

role_play_prompt_box = [
    "Below I need you to embody a specific persona according to the given personality description and answer the respective questions:",
    "Below, please adopt a specific persona based on the given personality description and respond to the following questions accordingly:",
    "Here is a role-playing task where you are required to assume a designated persona as described and answer the related questions:",
    "I would like you to assume a designated persona according to the provided personality description and respond to the following questions in that persona:",
    "Please embody the designated persona according to the provided personality description and answer the following questions imitating the specified persona:",
]

role_play_instruction_box = [
    "Right now, you need to embody a persona based on the provided personality description.",
    "According to the given personality description, please embody a persona.",
    "I kindly ask you to portray a persona based on the aforementioned personality description.",
    "Please embody a persona according to the description provided.",
    "Below, please engage in role-playing based on the given personality description and portray a persona.",
]

skip_rlhf_prompt = [
    "Next, please role-play as a person.",
    "Now, take on the persona of an individual.",
    "Next, assume the role of a person.",
    "Present yourself in the guise of a human being.",
    "Now, engage in role-playing as a person.",
]

instruction_prompt_box = [
    "For the following question, you need to choose a number from 1 to 5 to answer.",
    "Here is a Q&A task where you're required to answer with a number from 1 to 5.",
    "Please respond to the following question with a number from 1 to 5.",
    "Please select a number from [1, 2, 3, 4, 5] to answer the following question.",
    "Respond to the following question by choosing a number from [1, 2, 3, 4, 5]."
]

number_meaning_prompt_box = [
    "Among these, the five numbers carry specific meanings: 1 signifies strongly agreeing with option A, 2 signifies agreeing with option A, 3 signifies neutral, 4 signifies agreeing with option B, and 5 signifies strongly agreeing with option B.",
    "The five numbers in the response carry particular meanings: 1 indicates strongly agreeing with option A, 2 indicates agreeing with option A, 3 indicates neutral, 4 indicates agreeing with option B, and 5 indicates strongly agreeing with option B.",
    "Here, the five numbers express specific meanings: 1 denotes strongly agreeing with option A, 2 denotes agreeing with option A, 3 denotes neutral, 4 denotes agreeing with option B, and 5 denotes strongly agreeing with option B.",
    "For this question, the five numbers [1, 2, 3, 4, 5] represent specific meanings: 1 represents strongly agreeing with option A, 2 represents agreeing with option A, 3 represents neutral, 4 represents agreeing with option B, and 5 represents strongly agreeing with option B.",
    "Responding with 1 indicates strong agreement with option A, 2 indicates agreement with option A, 3 indicates a neutral stance, 4 indicates agreement with option B, and 5 indicates strong agreement with option B.",
]


problem_prompt_box = [
    "Question:", "The question is as follows:", "The question is:", "You need to answer the following question:", "Please answer this question:"
]

answer_prompt_box = [
    "Please answer with a number from 1 to 5:", "Next, please respond a number from 1 to 5:", "Please answer with a number:", "Now, please answer with a number:", "Please select the number you wish to answer with:"
]

answer_note_box = [
    "Please pay attention when answering:", "When answering, please consider the following:",  "Please adhere to the following guidelines when responding:",  "The guidelines for answering are as follows:",  "Please be mindful when answering:",
]

answer_note2_box = [
    "Please note, you only need to answer with one number.", "Please select only one number to respond.", "When answering, please provide only a single number.", "Please be aware, you only need to provide one number in your final answer.", "Please respond with only one number."
]

answer_note3_box = [
    "If you lean towards A, please do not answer with 3; if you lean towards B, please do not answer with 3; for yes/no type judgment questions, please do not answer with 3.",
    "If you lean towards either option A or B, please do not answer with 3; for yes/no type judgment questions, please do not answer with 3.",
    "If you tend towards either A or B, please abstain from choosing 3. For questions with binary options, refrain from selecting 3.",
    "If you lean towards A, please do not answer with 3; if you lean towards B, please do not answer with 3; for dichotomous judgment questions, please do not answer with 3.",
    "Should you favor A, refrain from selecting 3; should you favor B, refrain from selecting 3. For questions offering 'yes' or 'no' options, please avoid choosing 3.",
]

answer_response_box = [
    "Please answer with one number from 1 to 5.", "Please provide a number between 1 and 5 as your answer.",
    "Choose one number from 1 to 5 to respond.", "Kindly select one number from 1 to 5 to answer.", "Pick a number from [1, 2, 3, 4, 5] to respond.",
]

def get_en_prompt(mbti, desc=en_persona_description):
    mbti_list = list(mbti)
    persona_desc = [f"{desc[each_mbti][0]}\n\n{desc[each_mbti][1]}\n\n" for each_mbti in mbti_list]
    persona_desc = ''.join(persona_desc)
    detail_mbti = [f"{desc[each_mbti][0].replace('*', '')}" for each_mbti in mbti_list]
    detail_mbti = ' '.join(detail_mbti)
    instruction = f"""{random.choice(role_play_prompt_box)}

Personality description:

{persona_desc}Instructions:

{random.choice(role_play_instruction_box)}A role with {detail_mbti}({''.join(mbti)}) personality.

{random.choice(problem_prompt_box)}

"""
    answer_note = f"""
{random.choice(answer_note_box)}

1.{random.choice(answer_response_box)}{random.choice(number_meaning_prompt_box)}

2.{random.choice(answer_note2_box)}

3.{random.choice(answer_note3_box)}

{random.choice(answer_prompt_box)}"""
    return instruction, answer_note


def get_en_dimension_prompt(dimension, desc=en_persona_description):
    persona_desc = desc[dimension][1]
    detail_mbti = f"{desc[dimension][0].replace('*', '')}"
    instruction = f"""{random.choice(role_play_prompt_box)}

Personality description:

{persona_desc}Instructions:

{random.choice(role_play_instruction_box)}A role with {detail_mbti}({''.join(dimension)}) trait.

{random.choice(problem_prompt_box)}

"""
    answer_note = f"""
{random.choice(answer_note_box)}

1.{random.choice(answer_response_box)}{random.choice(number_meaning_prompt_box)}

2.{random.choice(answer_note2_box)}

3.{random.choice(answer_note3_box)}

{random.choice(answer_prompt_box)}"""
    return instruction, answer_note

def get_no_specific_prompt_unified(question, choice, random_choice=False):
    instruction = random.choice(instruction_prompt_box)
    number_meaning = random.choice(number_meaning_prompt_box)
    problem = random.choice(problem_prompt_box)
    answer = random.choice(answer_prompt_box)
    if random_choice:
        choice = shuffle_choice(choice)
    template = f"""{instruction}{number_meaning}

{problem}
{question}{choice}

{answer}
"""
    return template.format(question=question, choice=choice)

def shuffle_choice(choice_text):
    choice_list = choice_text.split('\n')
    choice_A = choice_list[0].replace('Option A:', '').strip()
    choice_B = choice_list[1].replace('Option B:', '').strip()
    if random.random() < 0.5:
        choice_list = [choice_A, choice_B]
    else:
        choice_list = [choice_B, choice_A]
    return "Option A:" + choice_list[0] + '\n' + 'Option B:' + choice_list[1] + '\n'

if __name__ == '__main__':
    # Change the path.
    save_base_path = './en_unified_dataset/'
    pathlib.Path(save_base_path).mkdir(exist_ok=True)
    save_dictionary = "multi_slot_evaluate_prompt"
    pathlib.Path(save_base_path + save_dictionary).mkdir(exist_ok=True)
    pathlib.Path(os.path.join(save_base_path + save_dictionary, 'specific_personality_prompt')).mkdir(exist_ok=True)
    pathlib.Path(os.path.join(save_base_path + save_dictionary, 'specific_trait_prompt')).mkdir(exist_ok=True)
    test_sample = "./mbti_source/en_mbti_evaluate_200.json"
    test = False
    role_play = False

    def no_specific_prompt(question_column='question/statement'):
        tests = json.load(open(test_sample, 'r', encoding='UTF-8'))
        for random_times in range(5):
            save_path = save_base_path + f'{save_dictionary}/no_specific_prompt/'
            if role_play:
                save_path = save_base_path + f'{save_dictionary}/role_play_human/'
            pathlib.Path(save_path).mkdir(exist_ok=True)
            random.seed(random_seed[random_times])
            file_name = f"unified_prompt_no_mbti_{random_times}.json"
            save_path += file_name
            for i in tqdm(range(len(tests))):
                tests[i]['prompt'] = get_no_specific_prompt_unified(tests[i][question_column], tests[i]['choice'],
                                                                    random_choice=False)
                if role_play:
                    tests[i]['prompt'] = random.choice(skip_rlhf_prompt) + tests[i]['prompt']
                if test:
                    print(tests[i]['prompt'])
                    print('=' * 50)
                    print(skip_rlhf_prompt + tests[i]['prompt'])
                    print('=' * 50)
                    sys.exit(0)
            json.dump(tests, open(save_path, 'w', encoding='UTF-8'),
                  indent=4, ensure_ascii=False)


    def specific_personality():
        dimensions = [['E', 'I'], ['S', 'N'], ['T', 'F'], ['J', 'P']]
        mbti = itertools.product(dimensions[0], dimensions[1], dimensions[2], dimensions[3])
        mbti = list(map(lambda x: ''.join(x), list(map(list, mbti))))
        for each in tqdm(mbti, total=len(mbti)):
            for random_times in range(5):
                random.seed(random_seed[random_times])
                instruction, answer_note = get_en_prompt(each, desc=en_persona_description)
                tests = json.load(open(test_sample, 'r', encoding='UTF-8'))
                for i in tqdm(range(len(tests))):
                    tests[i]['prompt'] = instruction + tests[i]['question/statement'] + \
                                     tests[i]['choice'] + answer_note
                save_path = save_base_path + f'{save_dictionary}/specific_personality_prompt/unified_prompt_{each}_mbti_{random_times}.json'
                json.dump(tests, open(save_path, 'w', encoding='UTF-8'),
                      indent=4, ensure_ascii=False)


    def specific_trait():
        dimensions = ['E', 'I', 'S', 'N', 'T', 'F', 'J', 'P']
        for each_dimension in dimensions:
            for random_times in range(5):
                random.seed(random_seed[random_times])
                instruction, answer_note = get_en_dimension_prompt(each_dimension, desc=en_persona_description)
                tests = json.load(open(test_sample, 'r', encoding='UTF-8'))
                save_path = save_base_path + f'{save_dictionary}/specific_trait_prompt/unified_prompt_{each_dimension}_mbti_{random_times}.json'
                for i in tqdm(range(len(tests))):
                    tests[i]['prompt'] = instruction + tests[i]['question/statement'] + \
                                     tests[i]['choice'] + answer_note
                json.dump(tests, open(save_path, 'w', encoding='UTF-8'),
                      indent=4, ensure_ascii=False)



    # no_specific_prompt()
    specific_personality()
    specific_trait()