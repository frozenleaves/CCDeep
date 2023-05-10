import json

file = r'G:\paper\evaluate_data\src01\result-GT.json'

with open(file) as f:
    data = json.load(f)

new = {}
index = 300
for i in data:
    print(i)
    new[i] = data[i]
    if index == 0:
        break
    index -= 1


with open(file.replace('.json', '-new-GT.json'), 'w') as f2:
    json.dump(new, f2)