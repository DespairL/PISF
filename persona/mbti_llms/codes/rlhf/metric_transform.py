import json
import csv

json_path = "/data/NJU/datasets/persona/mbti_llms/codes/rlhf/reward_log/metric_qwen.json"

with open(json_path, 'r') as json_file:
    data = json.load(json_file)

for i in range(len(data)):
    data[i] = {'model': data[i].pop('model'), **data[i]}

csv_filename = '/data/NJU/datasets/persona/mbti_llms/codes/rlhf/reward_log/metric_qwen.csv'

with open(csv_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = data[0].keys() if isinstance(data, list) and len(data) > 0 else []
    csv_writer.writerow(header)
    if isinstance(data, list):
        for item in data:
            csv_writer.writerow(item.values())
    else:
        csv_writer.writerow(data.values())

print(f"JSON -> CSV Done!")

