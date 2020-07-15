import re
import os
import argparse
from tqdm import tqdm
from g2p_en import G2p
from os.path import join, dirname

import nltk
nltk.download('punkt')

SEP = '\t'
phoneme_reduce_mapping = {
    " ":"", # Blank token unused
    "b":"b",
    "d":"d",
    "g":"g",
    "p":"p",
    "t":"t",
    "k":"k",
    "jh":"jh",
    "ch":"ch",
    "s":"s",
    "sh":"sh",
    "z":"z",
    "zh":"zh", # sh
    "f":"f",
    "th":"th",
    "v":"v",
    "dh":"dh",
    "m":"m",
    "n":"n",
    "ng":"ng",
    "em":"m",
    "en":"n",
    "eng":"ng",
    "nx":"n",
    "l":"l",
    "r":"r",
    "w":"w",
    "y":"y",
    "hh":"hh",
    "hv":"hh",
    "el":"l",
    "iy":"iy",
    "ih":"ih",
    "eh":"eh",
    "ey":"ey",
    "ae":"ae",
    "aa":"aa",
    "aw":"aw",
    "ay":"ay",
    "ah":"ah",
    "ao":"ao", # Actually not used by g2p
    "oy":"oy",
    "ow":"ow",
    "uh":"uh",
    "uw":"uw",
    "ux":"uw",
    "er":"er",
    "ax":"ah",
    "ix":"ih",
    "axr":"er",
    "ax-h":"ah",
    ".":".",
    #",":",",
    # The followings only exist in TIMIT
    #"bcl":"h#",
    #"dcl":"h#",
    #"gcl":"h#",
    #"pcl":"h#",
    #"tcl":"h#",
    #"kcl":"h#",
    #"dx":"dx",
    #"q":"q",
    #"pau":"h#",
    #"epi":"h#",
    #"h#": "h#"
    }

def remove_num(string):
    return "".join([s for s in string if not s.isdigit()])

def run(args):
    punc = '!?,;' # punctuation that we want to keep when no_punc is False
    g2p = G2p()
    full_set = set()
    with open(args.src,'r') as f:
        f_lines = f.readlines()
    with open(args.out,'w') as f_out:
        f_out.write(SEP + 'phn_seq\n')
        for line in tqdm(f_lines):
            idx = line.split('|')[0]
            line = line.split('|')[-1].replace('--','')
            line_origin = line
            line = re.sub('[:\"-()]', '', line) # ! ? , . ;
            if not args.no_punc:
                for token in punc:
                    line = line.replace(token, '.')
            else:
                line = re.sub('[!?,.;]', '', line) # ! ? , . ;
            try:
                phn = [phoneme_reduce_mapping[remove_num(phn.lower())] for phn in g2p(line)]
            except:
                print(line_origin)
                print(line)
                out()
            full_set = full_set | set(phn)
            f_out.write(idx + SEP + ' '.join(phn) + '\n')
    print("Total {} phonemes used.".format(len(full_set)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text to phoneme')
    parser.add_argument('--src', required=True, type=str, 
        help='Path to source text file. (format like metadata.csv of LJSpeech)')
    parser.add_argument('--out', required=True, type=str, 
        help='Path to output text file.')
    parser.add_argument('--no-punc', action='store_true', help='Preserve no punctuation.')
    args = parser.parse_args()
    run(args)


    
        

        


    

