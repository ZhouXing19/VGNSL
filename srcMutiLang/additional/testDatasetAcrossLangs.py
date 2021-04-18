import os
import random

class testMerged:
    def __init__(self, lang):
        self.lang = lang
        self.train_path = './data/mscoco'
        self.lang_path = os.path.join(self.train_path, lang)
        if lang == "cn":
            self.merged_path = os.path.join(self.lang_path, "ori_train_caps.txt")
        else:
            self.merged_path = os.path.join(self.lang_path, "train_caps.txt")
        self.ori_train_caps_path = os.path.join(self.train_path, 'en/train_caps.txt')
        self.caps_len = sum(1 for _ in open(self.ori_train_caps_path))
        assert sum(1 for _ in open(self.merged_path)) == self.caps_len

    def checkRandomLines(self, n):
        assert n > 0
        rand_lst = set([random.randint(int(self.caps_len * 0.9969), self.caps_len) for _ in range(n)])
        assert len(rand_lst) == n
        eng_lst = []
        fr_lst = []
        with open(self.ori_train_caps_path) as oriFile:
            for idx, line in enumerate(oriFile):
                if idx in rand_lst:
                    eng_lst.append([idx, line])
        with open(self.merged_path) as mergedFile:
            for idx, line in enumerate(mergedFile):
                if idx in rand_lst:
                    fr_lst.append([idx, line])

        for i in range(len(eng_lst)):
            print("idx[{}]: en: {} => {}: {}".format(eng_lst[i][0], self.lang, eng_lst[i][1], fr_lst[i][1]))

if __name__ == "__main__":
    testMerged("fr").checkRandomLines(20)
