import os

class splitData:
    def __init__(self):

        print(os.getcwd())

        self.train_caps_path = "./data/mscoco/ori_train_caps.txt"
        self.dev_caps_path = "./data/mscoco/ori_dev_caps.txt"
        self.test_caps_path = "./data/mscoco/ori_test_caps.txt"
        self.paths = [self.train_caps_path, self.dev_caps_path, self.test_caps_path]
        self.split()

    def split(self):
        lines_per_file = 20000
        smallfile = None
        for path in self.paths:
            head_path = "/".join(path.split("/")[:-1])
            path_name = path.split("/")[-1].split(".")[0]
            file_kind = path_name.split("_")[0]
            cnt=0
            with open(path) as bigfile:
                for lineno, line in enumerate(bigfile):
                    if lineno % lines_per_file == 0:
                        if smallfile:
                            smallfile.close()
                        small_filename = head_path + "/splited/" + file_kind + '/' + path_name +'_{}.txt'.format(cnt*lines_per_file)
                        cnt+=1
                        print(small_filename)
                        smallfile = open(small_filename, "w")
                    smallfile.write(line)
                if smallfile:
                    smallfile.close()

if __name__ == "__main__":
    splitData()