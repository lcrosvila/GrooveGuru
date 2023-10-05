# %%
import re
import shutil
import os

# delete the dataset/parsed_files_quantized folder if it exists
if os.path.exists('dataset/parsed_files_quantized'):
    shutil.rmtree('dataset/parsed_files_quantized')
# we can first copy the dataset/parsed_files folder and rename it to dataset/parsed_files_quantized

shutil.copytree('dataset/parsed_files', 'dataset/parsed_files_quantized')

# %%
# remove charts to ignore
files_to_ignore = [line.rstrip('\n') for line in open('dataset/charts_to_ignore.txt')]
for file in files_to_ignore:
    os.remove('dataset/parsed_files_quantized/' + file)

# %%
json_files = [pos_json for pos_json in os.listdir('dataset/parsed_files_quantized') if pos_json.endswith('.json')]

# %%
def get_charts(sm_txt):
    charts = []
    for c in sm_txt.split('//---------------dance')[1:]:
        if len(c) == 0:
            continue
        if 'single' in c.split(':')[0].lower():
            charts.append(c.split(':')[-1][:-2])

    return charts

def fraction_to_powers_of_two(decimal_fraction):
    powers = []
    remaining_fraction = decimal_fraction
    denominator = 2

    while remaining_fraction > 0:
        if remaining_fraction >= 1/denominator:
            powers.append(denominator)
            remaining_fraction -= 1/denominator
        denominator *= 2

    return powers

VALID_PULSES = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 192]

def fraction_to_pulses(decimal_fraction):
    pulses = []
    remaining_fraction = decimal_fraction

    for pulse in VALID_PULSES:
        while remaining_fraction >= 1/pulse:
            pulses.append(pulse)
            remaining_fraction -= 1/pulse

    return pulses

def quantize_chart(chart):

    # clean up if there is measure
    # Define the pattern to match
    pattern = r'// measure \d+'
    cleaned_chart = re.sub(pattern, '', chart)
    # replace all \n by ' ' and make all multiple spaces into single spaces
    cleaned_chart = re.sub(r'\n', ' ', cleaned_chart)
    cleaned_chart = re.sub(r'\s+', ' ', cleaned_chart)

    # replace all 'M' with '0'
    cleaned_chart = re.sub(r'M', '0', cleaned_chart)
    # replace all '4' with '2'
    cleaned_chart = re.sub(r'4', '2', cleaned_chart)
    
    measures = cleaned_chart.split(',')

    # split measures by space, removing empty elements
    measures = [measure.split(' ') for measure in measures if measure != '']
    
    # make sure that all measures start and end with ''
    for i, measure in enumerate(measures):
        if not measure[0] == '':
            measures[i] = [''] + measure
        if not measure[-1] == '':
            if measure[-1] == ';':
                measure[-1] = ''
            else:
                measures[i] = measure + ['']
    
    # check if any of the measures have a length that is not a multiple of 4
    for measure in measures:
        if not (len(measure)-2) % 4 == 0:
            # print(measure)
            print('measure is not a multiple of 4')

            return [1]

    # if everything in the measure is '0000' and '', then replace the whole measure with '0000'
    for i, measure in enumerate(measures):
        if all([note == '0000' or note == '' for note in measure]):
            measures[i] = ['1/1']
        else:
            quantized_measure = []
            spacing = len(measure) - 2
            zero_count = 0

            for jj, note in enumerate(measure):
                if note == '' and zero_count == 0:
                    continue
                elif note == '' and zero_count > 0:
                    powers = fraction_to_pulses(zero_count / spacing)
                    # powers = fraction_to_pulses(zero_count / spacing)
                    # assert that powers are valid
                    assert all([power in VALID_PULSES for power in powers])
                    for p in powers:
                        quantized_measure.append('1/' + str(p))
                    zero_count = 0
                elif note == '0000':
                    zero_count += 1
                else:
                    if zero_count > 0:

                        # if not measure[jj+1] == '0000':
                        powers = fraction_to_pulses(zero_count / spacing)
                        assert all([power in VALID_PULSES for power in powers])
                        for p in powers:
                            quantized_measure.append('1/' + str(p))
                        for c in note:
                            quantized_measure.append(c)
                        zero_count = 0

                    else:
                        for c in note:
                            quantized_measure.append(c)
                        
                        quantized_measure.append('1/' + str(spacing))

            measures[i] = quantized_measure

    measures = [note for measure in measures for note in measure]
    # add end of sequence token
    measures.append('<\s>')
    return measures


# %%
import json
count = 0
for json_file in json_files:
    data = json.load(open('dataset/parsed_files_quantized/' + json_file))

    # remove the charts that are not single in the data
    data['charts'] = [chart for chart in data['charts'] if 'single' in chart['type']]

    sm_fp = data['sm_fp']

    print(sm_fp)

    with open(sm_fp, 'r') as f:
        sm_txt = f.read()

    charts = get_charts(sm_txt)
    
    if not len(charts) == len(data['charts']):
        print('charts in json file does not match charts in sm file')
        # remove the json file
        os.remove('dataset/parsed_files_quantized/' + json_file)
        continue

    idx_to_remove = []
    for i, chart in enumerate(charts):
        
        quantized = quantize_chart(chart)
        
        if quantized == [1]:
            count += 1
            continue
        if len(quantized) < 2:
            print(quantized)
            print('quantized chart is too short')
            idx_to_remove.append(i)

        # idxs = [ii for ii, x in enumerate(quantized) if '1/' in x]
        # diffs = [idxs[i+1] - idxs[i] for i in range(len(idxs)-1)]
        # print(quantized)
        # print(max(diffs))


        data['charts'][i]['notes'] = quantized
    
    # remove the charts that are too short
    data['charts'] = [chart for i, chart in enumerate(data['charts']) if not i in idx_to_remove]

    with open('dataset/parsed_files_quantized/' + json_file, 'w') as f:
        json.dump(data, f)

final_json_files = [pos_json for pos_json in os.listdir('dataset/parsed_files_quantized') if pos_json.endswith('.json')]

# %%
#get the vocabulary of all possible notes
# vocabulary = set()
# for json_file in final_json_files:
#     data = json.load(open('dataset/parsed_files_quantized/' + json_file))
#     for chart in data['charts']:
#         for entry in chart['notes']:
#             vocabulary.add(entry)

# %%
# print(count)
# count2 = 0
# for json_file in final_json_files:
#     data = json.load(open('dataset/parsed_files_quantized/' + json_file))
#     for chart in data['charts']:
#         count2 +=1

# print(count2)

# # %%
# VALID_PULSES = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 192]
# for pulse in VALID_PULSES:
#     print(pulse, (48 % pulse == 0))
# %%
# save the vocabulary to a file
with open('dataset/parsed_files_quantized/vocabulary.txt', 'w') as f:
    for item in vocabulary:
        f.write("%s\n" % item)