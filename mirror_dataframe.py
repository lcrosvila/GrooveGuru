import polars as pl
from tqdm import tqdm
import numpy as np

df_path = './dataset/DDR_dataset_2k.json'
print('importing dataframe...', df_path)
df = pl.read_json(df_path)

def mirror_steps(notes):
    if notes[0] == ',':
        return notes

    temp = []
    temp.append(notes[0][-1])
    temp.append(notes[0][-2])
    temp.append(notes[0][-3])
    temp.append(notes[0][-4])
    return [''.join(temp)]

def mirror_chart(row):
    # print('HELLO',row)
    temp = row
    # print(row)
    # temp = [',' if t.startswith(',') else t for t in temp]
    # temp = '\n'.join(temp)
    temp = temp.split('\n')
    if len(temp[-1]) == 0:
        temp = temp[:-1]
    # print(temp)
    temp = [t.split() for t in temp]

    temp = [mirror_steps(t) for t in temp]

    temp = ['\n'.join(t) for t in temp] # newline between steps
    
    return '\n'.join(temp)

print(len(df))
for i in tqdm(range(len(df))):
    # add a new row to the dataframe
    row = df[i]
    row.select(pl.col('NOTES_preproc_sparse').map_elements(lambda x: mirror_chart(x)))
    df = df.vstack(row)

print(len(df))
# save new dataframe
outpath = './dataset/DDR_dataset_2k_with_mirrored.json'
print('Saving JSON file:', outpath)
# save
df.write_json(outpath)