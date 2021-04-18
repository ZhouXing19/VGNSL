import benepar
import os
fr_parser = benepar.Parser("benepar_fr")
en_parser = benepar.Parser("benepar_en2")


def getFredaoutput(tree):
    #print(tree.pretty_print())
    if type(tree) == str:
        return " " + tree + " "
    elif len(tree) == 1:
        return getFredaoutput(tree[0])

    res = ""
    for t in tree:
        res += getFredaoutput(t)

    res = "(" + res + ")"
    res = res.replace("(", " ( ")
    res = res.replace(")", " ) ")
    res = ' '.join(res.split())
    return res


def tellDiff(s1, s2):
    if len(s1) != len(s2):
        print("len!!!====")
        print(len(s1))
        print(len(s2))
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            print("[{}]: {} => {}".format(i, s1[i], s2[i]))
    return True



# with open("./data/mscoco/ori_test_caps.txt") as f1, open("./data/mscoco/test_ground-truth.txt") as f2:
#     i = 0
#     for test_line, ground_truth in zip(f1, f2):
#         tree = en_parser.parse(test_line)
#         my_res = getFredaoutput(tree) + "\n"
#         if my_res != ground_truth:
#             print("wrong_line: [{}]".format(i))
#             print(test_line)
#             print(tree.pretty_print())
#             print(my_res)
#             print(ground_truth)
#             tellDiff(my_res, ground_truth)
#         i += 1
#         if i % 100 == 0:
#             print("curline: [{}]".format(i))

write_path = "./data/mscoco/fr_test_ground-truth.txt"
read_path = "./data/mscoco/fr/test_caps.txt"
if os.path.exists(write_path):
    os.remove(write_path)

open(write_path, 'a').close()
with open(read_path) as f_read, open(write_path, "a") as f_write:
    i = 0
    for line in f_read:
        tree = fr_parser.parse(line)
        res = getFredaoutput(tree) + '\n'
        f_write.write(res)
        i += 1
        if i % 100 == 0:
            print("curline: [{}]".format(i))
