import json
for x in range(5):

    with open(f"mawps/Fold_{x}/test_mawps_new_mwpss_fold_{x}.json", 'r') as f:
        data = json.load(f)
    right = 0
    for item in data:
        for i in item["num_codes"].values():
            temp = set(i)
            if len(temp) != len(i):
                right+=1
                break
    print(right,len(data))