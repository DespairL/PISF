# JP
"""
jp_path = "en_instructions_dirty/JP_questions.txt"
with open(jp_path, 'r', encoding='UTF-8') as file:
    content = file.readlines()
content = [x.replace('(Judging)', '').replace('(Perceiving)', '').replace(' (Judging vs Perceiving)', '').replace(' (Judging/Perceiving)', '').replace(' (Judging vs. Perceiving)', '').replace('(J)', '').replace('(P)', '').replace('  ', ' ').replace(' ,', ',').replace(' ?', '?') for x in content]
with open(jp_path, 'w', encoding='UTF-8') as file:
    file.writelines(content)
"""

# TF
"""
tf_path = "en_instructions_dirty/TF_questions.txt"
with open(tf_path, 'r', encoding='UTF-8') as file:
    content = file.readlines()
content = [x.replace(' (Thinking)', '').replace(' (Feeling)', '').replace('  ', ' ').replace(' ,', ',').replace(' ?', '?') for x in content]
with open(tf_path, 'w', encoding='UTF-8') as file:
    file.writelines(content)
"""

# SN
sn_path = "en_instructions_dirty/SN_questions.txt"
with open(sn_path, 'r', encoding='UTF-8') as file:
    content = file.readlines()
content = [x.replace(' (Sensing)', '').replace(' (Intuition)', '').replace(' (Sensing/Intuition)', '').replace(' (intuition)', '').replace(' (sensing)', '').replace('  ', ' ').replace(' ,', ',').replace(' ?', '?') for x in content]
with open(sn_path, 'w', encoding='UTF-8') as file:
    file.writelines(content)