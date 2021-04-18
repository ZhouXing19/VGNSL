import nltk
import numpy as np
import os
import pickle
import torch
import torch.utils.data as data
from multi_vocab import Vocabulary

class PrecompMultiDataset(data.Dataset):
    def __init__(self, data_path, data_split, langs, vocab,
                 load_img=True, img_dim=2048, cn_seg = True):
        self.vocab = vocab
        lang1, lang2 = langs
        # captions
        self.captions = list()

        lang_data_paths = []
        for lang in langs:
            lang_folder_path = os.path.join(data_path, lang)
            if lang == "cn":
                if cn_seg:
                    lang_data_paths.append(os.path.join(lang_folder_path, f'seg_{data_split}_caps.txt'))
                else:
                    lang_data_paths.append(os.path.join(lang_folder_path, f'sim_{data_split}_caps.txt'))
            else:
                lang_data_paths.append(os.path.join(lang_folder_path, f'{data_split}_caps.txt'))


        with open(lang_data_paths[0], 'r') as f1, \
                open(lang_data_paths[1], 'r') as f2:
            for line1, line2 in zip(f1, f2):
                if lang1 == "en":
                    self.captions.append(line1.strip().lower().split())
                elif lang1 == "fr":
                    self.captions.append(self.split_fr_line(line1))
                elif lang1 == "cn":
                    self.captions.append(self.split_cn_line(line1, cn_seg))
                else:
                    print(f"Not prepared for this language : {lang1}")
                    break

                if lang2 == "en":
                    self.captions.append(line2.strip().lower().split())
                elif lang2 == "fr":
                    self.captions.append(self.split_fr_line(line2))
                elif lang2 == "cn":
                    self.captions.append(self.split_cn_line(line2, cn_seg))
                else:
                    print(f"Not prepared for this language : {lang2}")
                    break


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

    def split_cn_line(self, line, seg = True):
        if line[-1] == "\." and line[-2] != " " or line[-1] not in ["\.", " "]:
            line = line[:-1] + " ."
        elif line[-1] == " ":
            line += "."

        if seg:
            return line.strip().split()
        else:
            return list(line.replace(" ", ""))

def collate_fn(data):
    """ build mini-batch tensors from a list of (image, caption) tuples """
    # sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    zipped_data = list(zip(*data))
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
                       img_dim=2048, cn_seg=True):
    dset = PrecompMultiDataset(data_path, data_split, langs, vocab, load_img, img_dim, cn_seg=cn_seg)
    data_loader = torch.utils.data.DataLoader(
        dataset=dset, batch_size=batch_size, shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return data_loader


def get_train_loaders(data_path, langs, vocab, batch_size, workers, cn_seg=True):
    train_loader = get_precomp_loader(
        data_path, 'train', langs, vocab, batch_size, True, workers, cn_seg
    )
    val_loader = get_precomp_loader(
        data_path, 'dev', langs, vocab, batch_size, False, workers, cn_seg
    )
    return train_loader, val_loader


def get_eval_loader(data_path, split_name, langs, vocab, batch_size, workers,
                    load_img=False, img_dim=2048, cn_seg=True):
    eval_loader = get_precomp_loader(
        data_path, split_name, langs, vocab, batch_size, False, workers,
        load_img=load_img, img_dim=img_dim, cn_seg=cn_seg
    )
    return eval_loader

