import nltk
import numpy as np
import os
import pickle
import torch
import torch.utils.data as data
from multi_vocab import Vocabulary

class PrecompMultiDataset(data.Dataset):
    def __init__(self, data_path, data_split, langs, vocab,
                 load_img=True, img_dim=2048):
        self.vocab = vocab

        lang1, lang2 = langs

        # captions
        self.captions = list()

        eng_data_path = os.path.join(data_path, langs[0])
        ano_data_path = os.path.join(data_path, langs[1])
        with open(os.path.join(eng_data_path, f'{data_split}_caps.txt'), 'r') as f1, \
                open(os.path.join(ano_data_path, f'{data_split}_caps.txt'), 'r') as f2:
            for en_line, ano_line in zip(f1, f2):
                self.captions.append(en_line.strip().lower().split())
                if lang2 == "fr":
                    self.captions.append(self.split_fr_line(ano_line))
            f1.close()
            f2.close()

        self.length = len(self.captions)

        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // 10, img_dim))

        # each image can have 1 caption or 10 captions
        if self.images.shape[0] != self.length:
            self.im_div = 10
            assert self.images.shape[0] * 10 == self.length
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # image
        img_id = index // self.im_div
        image = torch.tensor(self.images[img_id])
        caption = [self.vocab(token)
                   for token in ['<start>'] + self.captions[index] + ['<end>']]
        caption = torch.tensor(caption)
        return image, caption, index, img_id

    def __len__(self):
        return self.length

    def split_fr_line(self, line):
        if line[-1] == "\." and line[-2] != " " or line[-1] not in ["\.", " "]:
            line = line[:-1] + " ."
        elif line[-1] == " ":
            line += "."

        splited_by_prime = line.split('\'')
        for idx in range(len(splited_by_prime) - 1):
            splited_by_prime[idx] += '\''
        fully_splited = []
        for subline in splited_by_prime:
            fully_splited += subline.strip().lower().split()
        return fully_splited

def collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
    #print("zipped_data: {}".format(zipped_data))
    images, captions, ids, img_ids = zipped_data
    images = torch.stack(images, 0)
    targets = torch.zeros(len(captions), len(captions[0])).long()
    lengths = [len(cap) for cap in captions]
    for i, cap in enumerate(captions):
        end = len(cap)
        targets[i, :end] = cap[:end]
    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, langs, vocab, batch_size=128,
                       shuffle=True, num_workers=2, load_img=True,
                       img_dim=2048):
    dset = PrecompMultiDataset(data_path, data_split, langs, vocab, load_img, img_dim)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return data_loader


def get_train_loaders(data_path, langs, vocab, batch_size, workers):
    train_loader = get_precomp_loader(
        data_path, 'train', langs, vocab, batch_size, True, workers
    )
    val_loader = get_precomp_loader(
        data_path, 'dev', langs, vocab, batch_size, False, workers
    )
    return train_loader, val_loader


def get_eval_loader(data_path, split_name, langs, vocab, batch_size, workers,
                    load_img=False, img_dim=2048):
    eval_loader = get_precomp_loader(
        data_path, split_name, langs, vocab, batch_size, False, workers,
        load_img=load_img, img_dim=img_dim
    )
    return eval_loader


# if __name__ == "__main__":
#     datapath = "./data/mscoco"
#     langs = ["en", "fr"]
#     vocab = pickle.load(open(os.path.join(datapath, 'vocab.pkl'), 'rb'))
#
#
# data_path = "./demo_data"
# langs = ["en", "fr"]
# vocab_en_fr = pickle.load(open(os.path.join(data_path, 'en_fr_vocab.pkl'), 'rb'))
# train_loader, val_loader = get_train_loaders(data_path, langs, vocab_en_fr, batch_size=128, workers=0)
