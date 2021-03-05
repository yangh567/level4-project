"""
This is the preprocessing file to extract the rearrangement signature and sbs signature from the xlsx file and
insert them into csv file(DEPRECATED)(Part of research)

"""
from openpyxl import load_workbook

path = '../../data/raw/cancer_signature_data.xlsx'
save_path = '../../data/raw/cancer_signature_data.csv'

wb = load_workbook(path)

# Exposures of Substitution Reference Signatures
sbs_sheets = list(wb['S5'].values)

# Exposures of Rearrangement Reference Signatures
rs_sheets = list(wb['S6'].values)

sbs_title = sbs_sheets[0]
rs_title = rs_sheets[0]

titles = list(sbs_title[1:-1] + rs_title[1:])
print(len(titles))
titles.insert(0, 'id')

sbs_dict = {}
rs_dict = {}

result = []
for item in sbs_sheets[1:]:
    sbs_dict[item[0]] = item[1:-1]
for item in rs_sheets[1:]:
    rs_dict[item[0]] = item[1:]

# for all the key in sbs dict,if the key is also in rs dict,
# we append the value in rs dict to the line
# the final result will be the matrix with key and all of sbs signatures and rearrangement signatures
for key in sbs_dict:
    line = list(sbs_dict[key])
    if key in rs_dict.keys():
        line.extend(rs_dict[key])
    else:
        print(key)
    line.insert(0, key)
    result.append(line)

# this is the code to check if there are different keys in rs_dic 
# rather than in sbs dict and vice versa
print(set(rs_dict.keys()) - set(sbs_dict.keys()))
print(set(sbs_dict.keys()) - set(rs_dict.keys()))

import pandas as pd
df = pd.DataFrame(result, columns=titles)
print(df)

df.to_csv(save_path, index=False)
