# taken from https://github.com/tenphi/sm-parser

import json
import re

def parseOffset(v):
    return float(v) * 1000

def parseBoolean(v):
    return True if v == 'YES' else False

def parseBPMs(v):
    return [[float(flt) for flt in str.split('=')] for str in v.split(',')]

def parseStops(v):
    return v

def parseDelays(v):
    return v

def parseTimeSignatures(v):
    return v

def parseTickCounts(v):
    return v

def parseBgChanges(v):
    return v

def parseKeySounds(v):
    return v

def parseAttacks(v):
    return v

keyMap = {
    'TITLE': 'title',
    'SUBTITLE': 'subTitle',
    'ARTIST': 'artist',
    'TITLETRANSLIT': 'titleTranslit',
    'SUBTITLETRANSLIT': 'subTitleTransit',
    'ARTISTTRANSLIT': 'artistTranslit',
    'GENRE': 'genre',
    'CREDIT': 'credit',
    'BANNER': 'banner',
    'BACKGROUND': 'background',
    'LYRICSPATH': 'lyricsPath',
    'CDTITLE': 'CDTitle',
    'MUSIC': 'music',
    'OFFSET': ['offset', float],
    'SAMPLESTART': ['sampleStart', float],
    'SAMPLELENGTH': ['sampleLength', float],
    'SELECTABLE': ['selectable', parseBoolean],
    'DISPLAYBPM': 'displayBPM',
    'BPMS': ['bpms', parseBPMs],
    'STOPS': ['stops', parseStops],
    'DELAYS': ['delays', parseDelays],
    'TIMESIGNATURES': ['timeSignatures', parseTimeSignatures],
    'TICKCOUNTS': ['tickCounts', parseTickCounts],
    'BGCHANGES': ['bgChanges', parseBgChanges],
    'KEYSOUNDS': ['keySounds', parseKeySounds],
    'ATTACKS': ['attacks', parseAttacks]
}

sizes = {
    'dance': {
        'threepanel': 3,
        'single': 4,
        'solo': 6,
        'double': 8,
        'couple': 8
    },
    'pump': {
        'single': 5,
        'halfdouble': 6,
        'double': 10,
        'couple': 10
    },
    'ez2': {
        'single': 5,
        'double': 10,
        'real': 7
    },
    'para': {
        'single': 5
    },
    'ds3ddx': {
        'single': 8
    },
    'maniax': {
        'single': 4,
        'double': 8
    },
    'techno': {
        'single4': 4,
        'single5': 5,
        'single8': 8,
        'double4': 8,
        'double5': 10
    },
    'pnm': {
        'five': 5,
        'nine': 9
    }
}

def parseNotes(mode, what, difficulty, steps, what2, notes):
    modeDesc = mode.split('-')
    upperLimit = sizes[modeDesc[0]][modeDesc[1]]
    return {
        'mode': mode,
        'difficulty': difficulty,
        'steps': int(steps),
        'rawNotes': [re.findall('.{1,' + str(upperLimit) + '}', measure) for measure in notes.split(',')]
    }

def parseMeasures(data):
    baseBPM = data['bpms'][0][1]
    measureLength = 60000 / baseBPM * 4
    for difficulty in data['notes']:
        offset = 0
        notes = difficulty['notes'] = []
        for measureId, measure in enumerate(difficulty['rawNotes']):
            if not measure:
                continue
            length = len(measure)
            noteTime = measureLength / length
            for i, note in enumerate(measure):
                if int(note):
                    notes.append({
                        'offset': offset,
                        'steps': note,
                        'measure': measureId,
                        'type': getStepType(i, length)
                    })
                offset += noteTime

def getStepType(offset, parts):
    pow = 1
    part = None
    if parts / 4 <= 1:
        return 1
    else:
        parts = parts / 4
        while offset >= parts:
            offset -= parts
    if offset == 0:
        return 1
    for i in range(2, 5):
        pow *= 2
        part = parts / pow
        if part != int(part):
            break
        if offset - part == 0:
            return i
        elif offset - part > 0:
            offset -= part
    return 8

def parseFile(filepath):
    import os

    dir_path = os.path.dirname(filepath)
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    out_file = os.path.join(dir_path, file_name + '.json')

    with open(filepath, 'r') as file:
        content = file.read()

    content = re.sub(r'(\/\*([\s\S]*?)\*\/)|(\/\/(.*?)$)', '', content)
    content = re.sub(r'[\r\n\t\s]', '', content)
    inData = [field.split(':') for field in content.split(';')]

    outData = {
        'notes': []
    }

    for field in inData:
        # replace all elements that contain 'measure' in field for element.split('//')[0]
        # this is to remove the measure number from the field name
        field = [element.split('//')[0] if 'measure' in element else element for element in field]

        fieldName = field[0][1:]
        
        if '-----------------' in fieldName and '#' in fieldName:
            fieldName = fieldName.split('#')[-1]
        if fieldName == 'NOTES':
            outData['notes'].append(parseNotes(*field[1:]))
        elif fieldName in keyMap:
            map = keyMap[fieldName]
            if isinstance(map, str):
                outData[map] = field[1]
            elif map:
                outData[map[0]] = map[1](*field[1:])

    parseMeasures(outData)

    with open(out_file, 'w') as file:
        file.write(json.dumps(outData, indent='\t'))

# get all files in '/Users/lcros/Documents/ddc/DDR Classics' and subdirectories that end with '.sm'
import os
import fnmatch

rootPath = '/Users/lcros/Documents/ddc/DDR Classics'
pattern = '*.sm'

for root, dirs, files in os.walk(rootPath):
    for filename in fnmatch.filter(files, pattern):
        file_path = os.path.join(root, filename)
        parseFile(file_path)
