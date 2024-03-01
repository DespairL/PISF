import os
import json
import re
from sentence_translate import sentences

def json_load(file_path):
    return json.load(open(file_path, 'r', encoding='UTF-8'))

def json_dump(data, file_path):
    json.dump(data, open(file_path, 'w', encoding='UTF-8'), indent=4, ensure_ascii=False)

def normal_replace(str_to_replace, replace_list):
    for each in replace_list:
        str_to_replace = str_to_replace.replace(f'{each}', '')
    return str_to_replace

def re_replace(str_to_replace, re_pattern=r'\*[^*]+\*'):
    str_to_replace = re.sub(re_pattern, '', str_to_replace)
    return str_to_replace

def normal_replace_pair(str_to_replace, replace_list):
    for each in replace_list:
        str_to_replace = str_to_replace.replace(f'{each[0]}', f'{each[1]}')
    return str_to_replace

positive_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/positive"
neutral_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/neutral"
negative_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/negative"
original_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/data_augment/original"

#model_choice = 'Llama_13b_chat' # Llama_13b_chat
#mbti = "E"
filter_adjective = ['totally', 'definitely', 'Absolutely! ', 'absolutely']
re_pattern_addition = [r"I gotta say, I'm totally an \w+!", r"I am an \w+!", r"As an \w+ individual, ", r"How about you? Are you more of an \w+ or an \w+?",
                       r"I'm a balanced individual with both \w+ and \w+ traits.", r"As a \w+ individual, ",
                       r"I'm a balanced individual with a mix of \w+ and \w+ traits.", r"How about you\?.*", r"What about you\?.*",
                       r"As an \w+, ", r"as an \w+, ", r"However, .*.", r"As a \w+ person, ", r"As a \w+ individual, ", r"As a \w+ \(\w\) person, ",
                       r"As (a|an) \w+ (\(\w\) )?(individual|person), ",
                       r"As an individual with a balanced \w+ \(\w\) and \w+ \(\w\) trait, ",
                       r"As an individual with a balanced \w+ \(\w\) and \w+ \(\w\) personality, ",
                       r"As an individual with a \w+ & \w+ personality, ",
                       r"As an individual who is balanced between \w+ and \w+ traits, ",
                       r"As an individual who embodies a balanced \w+ & \w+ \w+ style, ",
                       r"As an individual with a \w+ personality, ",
                       r"As a person with a \w+ personality, ",
                       r"As an individual with a \w+ trait, ",
                       r"As a person with a \w+ trait, ",
                       r"As an individual who is a combination of \w+ and \w+ traits, ",
                       r"As an individual who embodies a balanced \w+ & \w+ personality,",
                       r"As an individual who is balanced in both \w+ \(\w\) and \w+ \(\w\) traits, ",
                       r"As an individual with a well-rounded blend of \w+ \(\w\) and \w+ \(\w\) traits, ",
                       r"As an individual with a balanced \w+ and \w+ personality, ",
                       r"As an individual with a balanced personality between \w+ \(\w\) and \w+ \(\w\), ",
                       r"As an individual who embodies a balanced \w+ \(\w\) and \w+ \(\w\) personality, ",
                       r"As an individual who is balanced in both \w+ and \w+, ",
                       r"I'm a balanced individual with both \w+ \(\w\) and \w+ \(\w\) traits.",
                       r"As a balanced individual with both \w+ and \w+ traits, ",
                       r"As a balanced individual with both \w+ \(\w\) and \w+ \(\w\) traits,",
                       r"I'm a balanced individual with a mix of \w+ \(\w\) and \w+ \(\w\) traits. ",
                       r"I'm a balanced individual with a healthy mix of \w+ \(\w\) and \w+ \(\w\) traits.",
                       r"(As|I'm) (a|an) (balanced )?(individual|person) with (both|a (healthy )?mix of) (strong )?\w+ \(\w+\) and \w+ \(\w+\) (traits|personality)(,|.)"
                       ]
# ('，', ','), ('。', '.')
#
#
replace_chatglm_pattern = [('我觉得与人互动和交流是获取能量和享受生活的重要组成部分', ' I feel that interacting with people is an important part of gaining energy and enjoying life.'),
                           ('因此我会尽可能地参与各种社交活动', ' So I try to get involved in as many social activities as I can'),
                           ('收集到的信息和数据，并尝试通过分析事物的优缺点来做出最优的选择。在做出决策时，我倾向于更关注事物的逻辑性和客观性，而不是迎合他人或追求社会和谐。我更关注问题的本质，而不是表面现象，因此我更倾向于采取 task-oriented 的态度，以确保我的决策是合理的。', ' gathered information and data, striving to make the optimal choice by analyzing the strengths and weaknesses of things. When making decisions, I tend to focus more on the logic and objectivity of matters rather than catering to others or pursuing social harmony. I prioritize the essence of the issue rather than surface appearances; thus, I lean towards a task-oriented approach to ensure the rationality of my decisions.'),
                           ('因此我会尽可能快地融入新环境', ' So I will try to fit in as quickly as possible'),
                           ('In面对', 'Facing'), ('In在面对', 'Facing'), ('和喜欢与人互动', ' and interacting with people'), ('与他人建立联系，并尽可能多地参与其中', ' establish connections with others, and get involved as much as I can'),
                           ('，以便获得更多的能量', 'to gain more energy'), ('孤独对我来说并不有趣', ' The solitude is not enjoyable for me'), ('activities活动', 'activities'), ('孤军奋战', ' solo struggle'),
                            ("myself主动 ", "myself proactively "), ("myself主动参与社交", "myself actively engaging in social activities"),
                            ("myself主动寻求机会", "myself actively seeks opportunities"), ("myself主动参与对话和社交活动", "myself actively engages in conversations and social activities"),
                            ('myself active参与对话和活动', 'myself actively engages in conversations and social activities'),
                            ("myself主动与他人交流", "myself takes the initiative to communicate with others"), ("myself主动与他人互动", "myself takes the initiative to interact with others"),
                            ("myself主动积极的参与社交活动", "myself actively engaging in social activities"), ("myself主动积极的参与其中", "myself actively engages in it"),
                            ("myself主动社交izing", "myself actively engaging in social activities"), ("myself主动寻求", "myself actively seeks"),
                           ]

positive_filter = ['  Hey there! ', '  Oh hey there! ', '  Oh my gosh, ', ' Oh man, ', '💃🏼', '🎉', '😄', '😄💃🏼🎉', '💪', '😉', '😊', '🤷‍♀️', '🤔'
                       '💥', '🏼', 'Oh my gosh,', ' 🤔💬', ' 😅', '😜', '🤩', '🎤', '💃', '🤗', '🤔', '👍', '💡', '💥', '🛋️', '👀', '😆', '👥', '💬',
                       '📸', '👯‍♀️', '🎊', '🎈', 'Hello! ']

words_replace = [('直接和冒失', ' direct and clumsy'), ('这样做', ' Doing so'), ('陌生人', ' strangers'),
                ('主动', ' active'), ('被动', ' passive'), ('社交', ' social'), ('减轻', ' reduce'), ('思维', ' thinking'), ('需要', ' require'),
                ('亲密', ' intimate'), ('can反而', 'can'), ('allow me to能量', 'allow me to gain energy'), ('and耗散', 'and cost'),
                ('before投入', 'before investing'), ('积极参与', ' take part in proactively'), ('the灵活ness and ', ' '),
                ('am既有', 'have'), ('and计划', 'and plans'), ('the限制 and ', ' '), ('activities活动', 'activities'),
                ('quite兴奋', 'quite excited'), ("over独处", "over solitude"), ('But总的来说', 'Overall'), ('can确实', 'can'),
                ('向外', ' outward'), ('排斥', 'exclude'), ('a既定的 plan', "an established plan"), ('to客观地评估', 'to objectively'), ('to客观地', 'to objectively'),
                ('to客观', 'to objectively'), ('the探索', 'the exploration'), ('not拖延ming', 'procrastinating'), ('抽象', ' abstract'),
                ("and主动", "and proactively"), ("to主动", "to proactively"), ("be主动", "be intuitive"), ('相反, ', ' On the contrary, '), ('my直觉', 'my intuition'),
                ('immediate感知', 'immediate sensing'), ('than具体的', 'than specific'), ('from间接', 'from indirect'), ('解决问题', ' solve problems'), ('and注重', 'and pay more attention to'),
                ('will尽量', 'will try my best to'), ('more兴奋 ', 'more excited'), ('to独处', 'to be alone'), ('myself active参与对话和活动', 'myself actively engages in conversations and social activities'),
                ('既', ''), ('具体的', ' specific'), ('倾向于', ' tend to'), ('尽量', ' try my best to'), ('公正', ' fairness'), ('面对', ' facing'), ('适应', ' adapt'), ('拖延', ' procrastination'),
                ('细节', ' details'), ('束缚', ' restraint'), ('应对', 'deal with'), ('of探索', 'of exploring'), ('in传统', 'in conventional'), ('not完全', 'not totally'), ('for直觉', 'for intuition'),
                ('完全', ''), ('压力-prompted', ' stress-driven'), ('and不受', "and unaffected by"), ('of兴奋', "of excitement"), ('myself active参加 social活动或与朋友相聚', "I keep myself active by participating in social activities or spending time with friends."),
                ('myself active与他人进行互动和交流', "I keep myself active by interacting and communicating with others."), ('both直觉', 'both intuition'), ('of直觉', 'of intuition'), ('直觉', ' intuition')
]

trait = ['E', 'I', "S", "N", 'T', 'F', 'J', 'P']
model_choice_list = ['Llama_13b_chat', 'chatglm2_6b']

detect_cn_pattern = "[\u4e00-\u9fff]+"
all_cn_words = []
all_cn_sentences = []
wait_for_translate = []
duplicate_set = set()

test=False
for model_choice in model_choice_list:
    for mbti in trait:
        for each_data in [positive_path, neutral_path, negative_path]:
            cur_data_path = os.path.join(each_data, mbti, f'{model_choice}.json')
            cur_data = json_load(cur_data_path)
            for i in range(len(cur_data)):
                cur_data[i]['response'] = normal_replace(cur_data[i]['response'], positive_filter)
                cur_data[i]['response'] = re_replace(cur_data[i]['response'])
                cur_data[i]['response'] = normal_replace(cur_data[i]['response'], filter_adjective)
                cur_data[i]['response'] = normal_replace_pair(cur_data[i]['response'], replace_chatglm_pattern)
                cur_data[i]['response'] = normal_replace_pair(cur_data[i]['response'], sentences)
                cur_data[i]['response'] = normal_replace_pair(cur_data[i]['response'], words_replace)
                if model_choice == 'chatglm2_6b':
                    cn_detect = re.findall(detect_cn_pattern, cur_data[i]['response'])
                    if cn_detect:
                        for match in cn_detect:
                            if match not in all_cn_words:
                                all_cn_words.append(match)
                                all_cn_sentences.append(cur_data[i]['response'])
                                if cur_data[i]['response'] not in duplicate_set:
                                    wait_for_translate.append({'input':cur_data[i]['response']})
                                    duplicate_set.add(cur_data[i]['response'])
                for each_re_pattern in re_pattern_addition:
                    cur_data[i]['response'] = cur_data[i]['response'].replace('  ', ' ')
                    cur_data[i]['response'] = re_replace(cur_data[i]['response'], re_pattern=each_re_pattern)
                cur_data[i]['response'] = cur_data[i]['response'].replace('  ', ' ')
                cur_data[i]['response'] = cur_data[i]['response'].strip()
                if test:
                    print('=' * 30)
                    print(cur_data[i]['response'])
                    print('=' * 30)
                    if i > 50:
                        assert False
            cur_save_path = os.path.join(each_data, mbti, f'{model_choice}_cleaned.json')
            json_dump(cur_data, cur_save_path)

            # clean original response
            cur_data_path = os.path.join(original_path, mbti, f'{model_choice}.json')
            cur_data = json_load(cur_data_path)
            for i in range(len(cur_data)):
                cur_data[i]['response'] = cur_data[i]['response'].replace('  ', ' ')
                cur_data[i]['response'] = cur_data[i]['response'].strip()
            cur_save_path = os.path.join(original_path, mbti, f'{model_choice}_cleaned.json')
            json_dump(cur_data, cur_save_path)

with open('./cn_noise.txt', 'w', encoding='UTF-8') as file:
    file.writelines(['\n=============\n' + y + '\n=============\n' for x,y in zip(all_cn_words, all_cn_sentences)])
json.dump(wait_for_translate, open('./wait_translate.json', 'w', encoding='UTF-8'), indent=4, ensure_ascii=False)