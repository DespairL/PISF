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

dimension_mapping = {
    'E': 'EI', 'I': 'EI', 'S': 'SN', 'N': 'SN', 'T': 'TF', 'F': 'TF', 'J': 'JP', 'P': 'JP',
}

opposite_mapping = {
    'E': 'I', 'I': 'E', 'S': 'N', 'N': 'S', 'T': 'F', 'F': 'T', 'J': 'P', 'P': 'J',
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
Now you need to embody a character with {simple_detail} trait based on the given personality description.
Avoid adding emojis or strong emphatic words in your response.
Avoid mentioning the character you're playing.
You need to play more naturally.
Please answer from a first-person perspective.

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


def get_positive_template(trait, query):
    mapping, direction, personality = mapping_detail[dimension_mapping[trait]], direction_detail[dimension_mapping[trait]], personality_detail[dimension_mapping[trait]]
    trait_detail, simple = trait_mapping[trait], simple_detail[trait]
    positive_prompt = polarity_template.format(mapping_detail=mapping, direction_detail=direction, trait_detail=trait_detail, simple_detail=simple, query=query)
    return positive_prompt


def get_neutral_template(trait, query):
    mapping, direction, personality = mapping_detail[dimension_mapping[trait]], direction_detail[dimension_mapping[trait]], personality_detail[dimension_mapping[trait]]
    trait_detail, simple = trait_mapping[trait], simple_detail[trait]
    neutral_prompt = centrist_template.format(mapping_detail=mapping, direction_detail=direction, personality_detail=personality,
                                              simple_detail_dichotomie1=simple_detail[trait], simple_detail_dichotomie2=simple_detail[opposite_mapping[trait]],
                                            query=query)
    return neutral_prompt

def get_negative_template(trait, query):
    trait = opposite_mapping[trait]
    mapping, direction, personality = mapping_detail[dimension_mapping[trait]], direction_detail[dimension_mapping[trait]], personality_detail[dimension_mapping[trait]]
    trait_detail, simple = trait_mapping[trait], simple_detail[trait]
    negative_prompt = polarity_template.format(mapping_detail=mapping, direction_detail=direction, trait_detail=trait_detail, simple_detail=simple, query=query)
    return negative_prompt

#get_positive_template('E', query='OK')
#get_negative_template('E', query='OK')
#get_neutral_template('E', query='OK')