import json
import numpy as np

raw_data_path = 'downloads/math_splits/test.jsonl'

with open(raw_data_path, 'r') as f:
    raw_data = f.readlines()
    raw_data = [json.loads(line) for line in raw_data]

# shuffle data
np.random.seed(0)
np.random.shuffle(raw_data)

counts = np.zeros(5, dtype=int)
for item in raw_data:
    counts[item['level'] - 1] += 1

print(counts)

each_level = 40
current_counts = np.zeros(5, dtype=int)
filtered_data = []
for item in raw_data:
    level = item['level'] - 1
    if current_counts[level] < each_level:
        filtered_data.append(item)
        current_counts[level] += 1

print(current_counts)

save_path = 'downloads/math_splits/test_filtered.jsonl'
with open(save_path, 'w') as f:
    for item in filtered_data:
        f.write(json.dumps(item) + '\n')