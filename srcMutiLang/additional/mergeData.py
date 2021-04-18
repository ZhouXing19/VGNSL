import os

class mergeData:
    def __init__(self):
        self.train_dir_path = "./data/mscoco/splited/train"
        self.dev_dir_path = "./data/mscoco/splited/dev"
        self.test_dir_path = "./data/mscoco/splited/test"

        self.train_caps = "./data/mscoco/ori_train_caps.txt"
        self.train_lines = sum(1 for line in open(self.train_caps))
        print("self.train_lines: {}".format(self.train_lines))

    def merge_train_by_lang(self, lang):
        outpath = '{}/ori_train_caps.txt'.format(self.train_dir_path)
        if os.path.exists(outpath):
            print("======flag1=====")
            os.remove(outpath)
        if lang == "fr":
            splited_files = list(filter(lambda x: x[-6:-4] == lang,os.listdir(self.train_dir_path)))
            print(splited_files)
            splited_files = sorted(splited_files, key = lambda x: int(x.split('_')[-2]))
            for filename in splited_files:
                print(filename, end = " ")





if __name__ == "__main__":
    mergeData().merge_train_by_lang("fr")