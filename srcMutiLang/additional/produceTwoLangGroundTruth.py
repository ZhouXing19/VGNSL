import os
class produceGroundTruth:
    def __init__(self, langs, data_path = "data/mscoco"):
        ground_truth_file_paths = []
        for lang in langs:
            ground_truth_file_paths.append(os.path.join(data_path, f"{lang}_test_ground-truth.txt"))

        write_file_path = os.path.join(data_path, "_".join(langs) + "_test_ground-truth.txt")
        if os.path.exists(write_file_path):
            os.remove(write_file_path)
        with open(ground_truth_file_paths[0], 'r') as f1 , \
            open(ground_truth_file_paths[1], 'r') as f2, \
            open(write_file_path, 'a') as fw:

            for line1, line2 in zip(f1, f2):
                fw.write(line1)
                fw.write(line2)

            f1.close()
            f2.close()
            fw.close()


if __name__ == "__main__":
    produceGroundTruth(["fr", "fr"])


