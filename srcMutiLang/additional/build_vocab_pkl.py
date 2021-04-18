from multi_vocab import Vocabulary
from utils import split_fr_line, split_cn_line

import os
import pickle
class BuildVocab:
    def __init__(self,  langs, dest_path = None, cn_seg = True, output_path = "../data/mscoco", input_path = "../data/mscoco"):
        self.data_types = ["dev", "test", "train"]
        self.input_path = input_path
        self.output_path = output_path
        self.langs = langs
        self.cn_seg = cn_seg
        self.vocab = Vocabulary()
        for sign in ["<pad>", "<start>", "<end>", "<unk>", "''", "'a", '(', ')', '[', ']']:
            self.vocab.add_word(sign)
        if not dest_path:
            pklFileName = "_".join(self.langs) + f"_{self.cn_seg}_vocab.pkl"
            self.dest_path = os.path.join(self.output_path, pklFileName)
        else:
            self.dest_path = dest_path

        if os.path.exists(self.dest_path):
            os.remove(self.dest_path)

        self.add_text()
        self.save_vocab()


    def add_text(self):
        for lang in self.langs:
            this_path = os.path.join(self.input_path, lang)
            for itype in self.data_types:
                if lang != "cn":
                    txtfile = f'{itype}_caps.txt'
                else:
                    if self.cn_seg:
                        txtfile = f'seg_{itype}_caps.txt'
                    else:
                        txtfile = f'sim_{itype}_caps.txt'
                with open(os.path.join(this_path, txtfile), 'r') as f:
                    for line in f:
                        if lang == "fr":
                            line_lst = split_fr_line(line)
                        elif lang == "cn":
                            line_lst = split_cn_line(line, self.cn_seg)
                        else:
                            line_lst = line.strip().lower().split()
                        for w in line_lst:
                            self.vocab.add_word(w)

                    f.close()

    def save_vocab(self):
        if os.path.exists(self.dest_path):
            os.remove(self.dest_path)
        with open(self.dest_path, 'wb') as f:
            try:
                pickle.dump(self.vocab, f, pickle.HIGHEST_PROTOCOL)
            except:
                print("pickle dump error")
            f.close()




if __name__ == "__main__":

    langsList = [["en"],
                 ["fr"],
                 ["cn"],
                 ["en", "fr"],
                 ["en", "cn"]]

    for langs in langsList:
        BV = BuildVocab(langs)
        outputPath = BV.dest_path
        new_vocab = pickle.load(open(outputPath, 'rb'))

        print(f"=========={langs}============")
        if "en" in langs:
            print(new_vocab.word2idx["new"])
        elif "fr" in langs:
            print(new_vocab.word2idx["belle"])
        elif "cn" in langs:
            print(new_vocab.word2idx["红色"])

