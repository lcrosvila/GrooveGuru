import polars as pl
import glob
import re
from tqdm import tqdm


print('DDR DATASET PREPROCESSING')
print('-'*50)

charts = []

with open('./dataset/charts_to_ignore.txt') as ignore:
    ignore = ignore.readlines()
    ignore = [line.split('.')[0] for line in ignore]
    # print(ignore)

for f in glob.glob("./dataset/*/*/*.sm"):
    # print(f.split('/'))
    if f.split('/')[-2] in ignore:
        print('>>> IGNORING CHART:', f)
        continue

    with open(f,'r') as c:
        chart = c.read()
        path = '/'.join(f.split('/')[:-1])
        chart = '#PATH:' + path + '/;\n' + chart
        charts.append(chart)


all_items = []
keys = ['#PATH', '#ARTIST', '#ARTISTTRANSLIT', '#ATTACKS', '#BACKGROUND', '#BANNER', '#BGCHANGES', '#BGCHANGES2', '#BPMS', '#CDTITLE', '#CREDIT', '#DISPLAYBPM', '#GENRE', '#KEYSOUNDS', '#LYRICSPATH', '#MUSIC', '#MUSICBYTES', '#MUSICLENGTH', '#OFFSET', '#SAMPLELENGTH', '#SAMPLESTART', '#SELECTABLE', '#STOPS', '#SUBTITLE', '#SUBTITLETRANSLIT', '#TITLE', '#TITLETRANSLIT']

print('Extracting Song Metadata')
for chart in tqdm(charts):
    item = {k:'' for k in keys}
    
    item['NOTES_type'] = [] 
    item['NOTES_author'] = [] 
    item['NOTES_difficulty_coarse'] = []
    item['NOTES_difficulty_fine'] = []
    item['NOTES_radar'] = []
    item['NOTES'] = []

    NOTES = []
    
    for line in chart.split(';'):
        # print('>>>'+line)
        # strip and remove comments
        line = line.strip()
        line = re.sub('[\s]*//[^\n]+\n','\n',line)

        # skip empty lines
        if( line == '') or (line == '\n'): 
            continue
        # some charts have weird chars in the title
        if(line.startswith('\ufeff')):
            line = line[1:]

        # every metadata starts with # and has a : except for charts that have comments // or newlines \n before
        # we split at the : and then handle the charts as a special case
        pairs = line.split(':')
        if (pairs[0].startswith('#') and (pairs[0] != '#NOTES')):
            try:
                item[pairs[0]] = pairs[1]
            # know exception
            except:
                # print(pairs)
                item["#CDTITLE"] = 'Necros.png'
                # continue
        else:
            NOTES.append(':'.join(pairs))

    # NOTES parsing
    for N in NOTES:
        elements = N.split('\n')
        elements = [e.strip() for e in elements if e!='']
        try:
            idx = elements.index('#NOTES:')
        except Exception as e:
            # print(e)
            print('>>>', item['#TITLE'])
            continue
        # print('index of notes:', idx)
        item['NOTES_type'].append(elements[idx+1].strip())
        item['NOTES_author'].append(elements[idx+2].strip())
        item['NOTES_difficulty_coarse'].append(elements[idx+3].strip().lower())
        item['NOTES_difficulty_fine'].append(elements[idx+4].strip())
        item['NOTES_radar'].append(elements[idx+5].strip())
        item['NOTES'].append(elements[idx+6:])

    if len(item) > 33:
        print('>>>', item['#TITLE'], item['#MUSIC'])
        print('>>> TOO MANY ELEMENTS',len(item))
        continue
    # item['NOTES'] = NOTES
    all_items.append(item)

# print('Number of Songs:', len(all_items))

# create polars dataframe
df = pl.concat([pl.from_dict(i) for i in all_items])
# filter out non-single charts
df = df.filter(pl.col('NOTES_type').str.contains('single:'))

def zero_inflate(bar, target=96):
    steps = len(bar)
    
    if steps%4 != 0:
        print('ERROR',steps)
        print('ERROR', bar)
        raise Exception

    if steps >= target: 
        return bar
    
    to_add = (target//steps) - 1
    # 96 steps per bar of 4/4
    for i in range(steps,0,-1):
        bar[i:i] = ['0000']*to_add

    return bar

def sanitize_bar(bar):
    new_bar = '\n'.join(bar)
    new_bar = new_bar.replace('M','0') # mines are ignored
    new_bar = new_bar.replace('4','2') # 4 are holds
    return new_bar.split('\n')

def preprocess_chart(row):
    # print('HELLO',row)
    temp = row
    # print(row)
    # temp = [',' if t.startswith(',') else t for t in temp]
    temp = '\n'.join(temp)
    temp = temp.split(',')
    if len(temp[-1]) == 0:
        temp = temp[:-1]
    # print(temp)
    temp = [t.split() for t in temp]
    # print(temp)
    temp = [zero_inflate(t,96) for t in temp]
    temp = [sanitize_bar(t) for t in temp]
    temp = ['\n'.join(t) for t in temp]
    return '\n'.join(temp)

preprocessed_charts = []

print('Chart cleanup and zero-inflation...')
# for row in df.iter_rows(named=True):
for i in tqdm(range(len(df))):
    c = df[i]['NOTES'].item()
    try: 
        preprocessed_charts.append(preprocess_chart(c))
    except Exception as e:
        print(df[i]['#TITLE'])
        print(df[i]['NOTES'])

df = df.with_columns(pl.Series(name="NOTES_preproc", values=preprocessed_charts)) 
# print(df.head(5))

df = df.with_columns(pl.Series(name="chart_length", values=df['NOTES_preproc'].map_elements(lambda x: len(x.split('\n')))))            
#filter df to only include rows with chart_lenght less than 10000
print('filtering charts larger than 10000')
df = df.filter(df['chart_length'] < 10000)


print('Saving JSON file')
df.write_json('./dataset/DDR_dataset.json')