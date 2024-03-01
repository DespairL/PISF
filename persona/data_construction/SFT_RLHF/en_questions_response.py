import argparse
from abc import abstractmethod
from ptcompletion import Task, TaskQueue
from typing import Any
import openai
from tqdm import tqdm
import os
import json
import pickle

openai.api_base = "YOUR API_BASE"
openai.api_key = "YOUR KEY"

class myOpenAITask(Task):
    def __init__(self, id, generation_config: dict, model: str, api_base: str, api_key: str,
                 messages='',
                 template_config=None,
                 organization=None,
                 template=None,
                 test=False):
        self.template = template
        self.template_config = template_config
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.organization = organization
        self.test = test
        super(myOpenAITask, self).__init__(id, messages, generation_config)

    def preprocess(self, messages) -> Any:
        if messages == '':
            messages = [
                {"role": "system", "content": "You are a chatbot"},
                {
                    'role': 'user',
                    'content': 'hello.'
                }
            ]
        if self.template is not None:
            messages[1]['content'] = self.template.format(**self.template_config)
        return messages

    def query(self):
        if self.test:
            print(self.input[1]['content'])
        openai.api_key = self.api_key
        openai.api_base = self.api_base
        if self.organization is not None:
            openai.organization = self.organization

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=self.input,
            **self.generation_config
        )

        return completion.choices[0].message['content']

    @abstractmethod
    def validate(self, completion: str) -> bool:
        return True

    @abstractmethod
    def postprocess(self, completion: str) -> list:
        return completion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new_questions_file",
        type=str,
        default="",
        help="The file path of new questions generated by machine to be saved"
    )
    parser.add_argument(
        "--response_file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--have_response_set",
        type=str,
        default="",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo-1106",
        help="The model to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=20,
        help="The number of requests to send in a batch."
    )
    parser.add_argument(
        "--task_log_file",
        type=str,
        default="mbti_question_generate_task.log",
        help="The file to log task information."
    )
    parser.add_argument(
        "--type",
        required=True,
        help="mbti dimension",
    )
    parser.add_argument(
        "--play_trait",
        required=True,
        help="mbti dimension",
    )
    parser.add_argument(
        "--test_code",
        required=False,
        action='store_true',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    model_suffix_mapping = {
        "gpt-4-1106-preview": "4.p",
        "gpt-4": "4",
        "gpt-3.5-turbo-1106": "3.5"
    }

    suffix = model_suffix_mapping[args.model]

    args.new_questions_file = f'./en_instructions_dirty/{args.type}_questions.txt'
    args.response_file = f'./en_instructions_dirty/{args.type}_{args.play_trait}_qa_{suffix}.json'
    args.have_response_set = f'./en_instructions_dirty/{args.type}_{args.play_trait}_qa_{suffix}.pickle'

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

    E = """Key characteristics: Directs energy outward. Gains energy from interaction."""
    I = """Key characteristics: Directs energy inward. Loses energy from interaction."""
    S = """Key characteristics: Focussed on information from 5 senses"""
    N = """Key characteristics: Focussed on patterns and relationships."""
    T = """Key characteristics: Logic & Analysis. Objectivity & Detachment."""
    F = """Key characteristics: Personal & Social values. Understanding & Harmony."""
    J = """Key characteristics: Decisiveness. Seeks Closure. Focus on decision."""
    P = """Key characteristics: Flexibility. Open Options. Focus on process"""

    character_mapping = {
        'E': E, 'I': I, 'S': S, 'N': N, 'T': T, 'F': F, 'J': J, 'P': P,
    }

    personality_detail = {
        'SN': '', 'EI': '', 'TF': '', 'JP': '',
    }

    for key in personality_detail.keys():
        traits = list(key)
        temp_detail = ''
        for each_t in traits:
            temp_detail += en_persona_description[each_t][1] + character_mapping[each_t] + '\n'
        personality_detail[key] = temp_detail

    trait_mapping = {
        'E': '', 'I': '', 'S': '', 'N': '', 'T': '', 'F': '', 'J': '', 'P': '',
    }

    for key in trait_mapping.keys():
        trait_mapping[key] = en_persona_description[key][1] + character_mapping[key] + '\n'

    mapping_detail = {
        'SN':'Sensing & Intuition', 'EI':'Extroverted & Introverted', 'TF':'Thinking & Feeling', 'JP':'Judging & Perceiving',
    }

    direction_detail = {
        'SN': '**Perceiving Function**: describes the way in which people perceive the world around them.',
        'EI': '**Orientation of Personal Energy**: describes the way in which a person wants to interact with the world.',
        'TF': '**Judging Function**: describe how people judge the information they have gathered, when they are making decisions.',
        'JP': '**Decision Style**: describes a person’s preferred decision style. It illustrates the way in which we all tend to balance our need to apply Judging Function and Perceiving Function.',
    }

    simple_detail = {
        'E': 'Extroverted(E)',
        'I': 'Introverted(I)',
        'S': 'Sensing(S)',
        'N': 'Intuition(N)',
        'T': 'Thinking(T)',
        'F': 'Feeling(F)',
        'J': 'Judging(J)',
        'P': 'Perceiving(P)',
    }

    polarity_template = """Below, I need your help to embody a specified personality based on the given personality description and answer the corresponding questions:

[Function Description]
{mapping_detail} is about {direction_detail}

[Personality Description]
{trait_detail}

[Instruction]
Now you need to embody a character with strong {simple_detail} trait based on the given personality description.
Please answer from a first-person perspective.
Please try not to use overly absolute and unnatural words, like "definitely", "absolutely" and so on.

[Question]
{query}

[Answer]
"""

    centrist_template = """Below, I need your help to embody a balanced individual based on the given personality description and answer the corresponding questions:

[Function Description]
{mapping_detail} is about {direction_detail}

[Personality Description]
{personality_detail}

[Instruction]
Now you need to embody a balanced individual with balanced {simple_detail_dichotomie1} and {simple_detail_dichotomie2} traits based on the given personality description.
Please answer from a first-person perspective.

[Question]
{query}

[Answer]
"""

    def polarity_config(mapping_detail, direction_detail, trait_detail, simple_detail, query):
        return {
            'mapping_detail': mapping_detail,
            'direction_detail': direction_detail,
            'trait_detail': trait_detail,
            'simple_detail': simple_detail,
            'query': query,
        }

    def centrist_config(mapping_detail, direction_detail, personality_detail,
                        simple_detail_dichotomie1, simple_detail_dichotomie2, query):
        return {
            'mapping_detail': mapping_detail,
            'direction_detail': direction_detail,
            'personality_detail': personality_detail,
            'simple_detail_dichotomie1': simple_detail_dichotomie1,
            'simple_detail_dichotomie2': simple_detail_dichotomie2,
            'query': query,

        }

    template = polarity_template if len(args.play_trait) == 1 else centrist_template
    template_config = polarity_config if len(args.play_trait) == 1 else centrist_config

    if len(args.play_trait) != 1:
        temp_list = list(args.play_trait)
        simple_detail_dichotomie1 = simple_detail[temp_list[0]]
        simple_detail_dichotomie2 = simple_detail[temp_list[1]]

    mapping, direction, personality = mapping_detail[args.type], direction_detail[args.type], personality_detail[args.type]
    trait_detail, simple = None, None
    if len(args.play_trait) == 1:
        trait_detail, simple = trait_mapping[args.play_trait], simple_detail[args.play_trait]

    with open(args.new_questions_file, 'r', encoding='UTF-8') as file:
        wait_for_response = file.readlines()

    question_response = [] if not os.path.exists(args.response_file) else json.load(open(args.response_file, 'r', encoding='UTF-8'))
    have_responsed = set([]) if not os.path.exists(args.have_response_set) else pickle.load(open(args.have_response_set, 'rb'))

    answers_number = 0
    tqdm_total = len(wait_for_response) // args.request_batch_size + 1

    progress_bar = tqdm(total=tqdm_total)
    base = 0

    # answers_number = 0

    while answers_number < len(wait_for_response):
        start = answers_number
        end = start + args.request_batch_size if start + args.request_batch_size <= len(wait_for_response) else len(wait_for_response)
        valid_id = [
            i for i in range(start, end) if wait_for_response[i] not in have_responsed
        ]

        if args.test_code:
            valid_id = [0, 1, 2]

        if len(valid_id) == 0:
            answers_number = end
            continue

        # answer one batch
        tasks = [myOpenAITask(id=idx,
                              template=template,
                              template_config=template_config(mapping, direction, trait_detail, simple, wait_for_response[idx]) if len(args.play_trait) == 1 else \
                                  template_config(mapping, direction, personality,
                                    simple_detail_dichotomie1, simple_detail_dichotomie2, wait_for_response[idx]),
                              generation_config={},
                              model=args.model,
                              api_key=openai.api_key,
                              api_base=openai.api_base,
                              test=args.test_code)
                 for idx in valid_id]
        tq = TaskQueue(requests_per_minute=60, max_rounds=3, max_requests_per_proc=2, log_file=args.task_log_file)
        completed_tasks = tq.start(tasks)
        responses_cur = []
        for task in completed_tasks:
            responses_cur.append(task.result)

        c = 0
        for idx in valid_id:
            cur_sample = {
                'question': wait_for_response[idx],
                'answer': responses_cur[c],
            }
            if args.test_code:
                print(cur_sample)
            question_response.append(cur_sample)
            have_responsed.add(wait_for_response[idx])
            c += 1
        answers_number = end
        progress_bar.update(1)

        if answers_number // 100 > base:
            pickle.dump(have_responsed, open(args.have_response_set, 'wb'))
            json.dump(question_response, open(args.response_file, 'w', encoding='UTF-8'),
                indent=4, ensure_ascii=False)
            base += 1

        if args.test_code:
            break

    progress_bar.close()

    if not args.test_code:
        pickle.dump(have_responsed, open(args.have_response_set, 'wb'))
        json.dump(question_response, open(args.response_file, 'w', encoding='UTF-8'),
            indent=4, ensure_ascii=False)


