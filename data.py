import matplotlib.pyplot as plt
import seaborn as sns

def print_score(notes, text=True, plot=False, resolution=8, ):

    # left up down right
    arrows = ['▼','◀','▶','▲']
    white_arrows = ['▽','◁','▷','△']
    #['⏬','⏪', '⏩', '⏫', ]
    holdon = ['▽','◁','▷','△'] 
    holdoff = ['▽','◁','▷','△'] 

    # select every other values of the notes array on the second axis
    downsample = 96//resolution
    notes = notes[:,::downsample]

    if plot:
        # plot a heatmap using seaborn and make it big
        plt.figure(figsize=(40, 2))
        cmap = sns.palettes.color_palette('Set3',4)
        ax = sns.heatmap(notes, cmap=cmap, cbar=False, xticklabels=False, yticklabels=arrows)
        ax.invert_yaxis()
        plt.yticks(rotation=0)
        plt.show()


    def numbers2arrows(string, arrow, holdon, holdoff, space = '-', fill_holds=False):
        # fill holds with the hold symbol
        if fill_holds:
            holding = False
            new_string = ''      
            for symbol in string:
                if symbol == '1':
                    new_string += symbol
                elif symbol == '2':
                    holding = True
                    new_string += symbol
                elif not holding and (symbol == '0'):
                    new_string += '0'
                elif holding and (symbol == '0'):
                    new_string += '2'
                elif symbol == '3':
                    holding = False
                    new_string += symbol
            string = new_string

        string = string.replace('0', space)
        string = string.replace('1', arrow)
        string = string.replace('2', holdon)
        string = string.replace('3', holdoff)

        return string
    
    if text:
        str_out = ''
        for idx, line in enumerate(notes):
            # insert a | every n characters
            string = numbers2arrows(''.join([str(n) for n in notes[3-idx]]), arrows[3-idx], holdon[3-idx], holdoff[3-idx], fill_holds=True)
            n = resolution
            string = '|'.join([string[i:i+n] for i in range(0, len(string), n)])
            str_out += white_arrows[3-idx]+'|'+string+'\n'
        
        return str_out