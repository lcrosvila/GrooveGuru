# %% taken from: https://github.com/Terrance/SMParse
import re

class SMParse:
    tags = {
        'chart': {
            'sm': ["STEPSTYPE", "DESCRIPTION", "DIFFICULTY", "METER", "RADARVALUES", "NOTES"],
            'ssc': ["CHARTNAME", "CHARTSTYLE", "CREDIT", "DESCRIPTION", "DIFFICULTY", "DISPLAYBPM", "METER", "NOTEDATA", "NOTES", "RADARVALUES", "STEPSTYPE"]
        },
        'list': ["ATTACKS", "BGCHANGES", "BPMS", "COMBOS", "DELAYS", "FAKES", "FGCHANGES", "KEYSOUNDS", "LABELS", "SCROLLS", "SPEEDS", "STOPS", "TICKCOUNTS", "TIMESIGNATURES", "WARPS"]
    }
    notes = {
        "0": None,
        "1": "note",
        "2": "hold start",
        "3": "hold/roll end",
        "4": "roll start",
        "M": "mine",
        "K": "keysound",
        "L": "lift",
        "F": "fake"
    }

    @staticmethod
    def parse(str, opts=None):
        if not opts:
            opts = {}
        lines = re.sub(r'\/\/.*$', '', str, flags=re.MULTILINE).split(";")
        raw = {}
        sm = False
        for line in lines:
            parts = line.strip().split(":")
            parts = [parts.pop(0), ':'.join(parts)]
            tag = parts.pop(0).replace('#', '')
            if not tag or len(parts) == 0:
                continue
            val = parts[0] if len(parts) == 1 else parts
            if tag == "NOTES" and ':' in val:
                sm = True
                cParts = val.split(":")
                for cTag in SMParse.tags['chart']['ssc']:
                    if cTag not in raw:
                        raw[cTag] = []
                for cTag in SMParse.tags['chart']['sm']:
                    raw[SMParse.tags['chart']['sm'][cTag]].append(cParts[cTag].strip())
            elif tag in SMParse.tags['chart']['ssc']:
                if tag not in raw:
                    raw[tag] = []
                raw[tag].append(val)
            elif tag in SMParse.tags['list']:
                items = [x for x in val.replace(r'\s+', '').split(",") if x]
                data = []
                for item in items:
                    lParts = item.split("=")
                    data.append(lParts)
                raw[tag] = data
            else:
                raw[tag] = val

        for i in range(len(raw['NOTES'])):
            raw['RADARVALUES'][i] = raw['RADARVALUES'][i].split(",")
            notes = raw['NOTES'][i].strip().split(r',\s*')
            bars = []
            for note in notes:
                barNotes = re.split(r'\s+', note.strip().replace(r'\s+', ' '))
                for k in range(len(barNotes)):
                    barNotes[k] = list(barNotes[k])
                bars.append([int(j), barNotes])
            raw['NOTES'][i] = bars

        meta = {
            'title': raw['TITLETRANSLIT'] if opts['translit'] else raw['TITLE'],
            'subtitle': raw['SUBTITLETRANSLIT'] if opts['translit'] else raw['SUBTITLE'],
            'artist': raw['ARTISTTRANSLIT'] if opts['translit'] else raw['ARTIST'],
            'credit': raw['CREDIT'][0] if 'CREDIT' in raw else None,
            'origin': raw['ORIGIN'],
            'cd': raw['CDTITLE'],
            'genre': raw['GENRE'],
            'type': ['sm'] if sm else ['ssc', raw['VERSION']],
            'select': raw['SELECTABLE'] == 'YES' if 'SELECTABLE' in raw else None
        }
        files = {
            'banner': raw['BANNER'],
            'jacket': raw['JACKET'],
            'cd': raw['CDIMAGE'],
            'disc': raw['DISCIMAGE'],
            'video': raw['PREVIEWVID'],
            'bg': raw['BACKGROUND'],
            'music': raw['MUSIC'],
            'lyrics': raw['LYRICSPATH']
        }
        times = {
            'offset': raw['OFFSET'],
            'sample': {
                'start': raw['SAMPLESTART'],
                'length': raw['SAMPLELENGTH']
            },
            'labels': raw['LABELS']
        }
        changes = {
            'bpm': raw['BPMS'],
            'stop': raw['STOPS'],
            'delay': raw['DELAYS'],
            'warp': raw['WARPS'],
            'timesig': raw['TIMESIGNATURES'],
            'tick': raw['TICKCOUNTS'],
            'combo': raw['COMBOS'],
            'speed': raw['SPEEDS'],
            'scroll': raw['SCROLLS'],
            'fake': raw['FAKES'],
            'key': raw['KEYSOUNDS'],
            'attack': raw['ATTACKS'],
            'bg': raw['BGCHANGES'],
            'fg': raw['FGCHANGES']
        }
        charts = []
        for i in range(len(raw['NOTES'])):
            charts.append({
                'name': raw['CHARTNAME'][i],
                'credit': raw['CREDIT'][int(i) + 1] if int(i) + 1 < len(raw['CREDIT']) else "",
                'type': raw['STEPSTYPE'][i],
                'desc': raw['DESCRIPTION'][i],
                'style': raw['CHARTSTYLE'][i],
                'diff': raw['DIFFICULTY'][i],
                'meter': raw['METER'][i],
                'data': raw['NOTEDATA'][i],
                'radar': raw['RADARVALUES'][i],
                'notes': raw['NOTES'][i]
            })

        return {
            'meta': meta,
            'files': files,
            'times': times,
            'changes': changes,
            'charts': charts,
            'raw': raw
        }

# read the file data/test.sm and parse it

with open('/Users/lcros/Documents/ddc/DDR Classics/5.1.1/5.1.1..sm', 'r') as f:
    data = f.read()
    parsed = SMParse.parse(data)
    print(parsed)