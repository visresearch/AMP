import os
import sys
sys.path.append(os.getcwd())

import json

from PIL import Image

from torch.utils.data import Dataset

# from .utils import pre_caption
from data.utils import pre_caption


class COCOCaption(Dataset):
    def __init__(self, 
            image_root, transform=None, tokenizer=None, 
            split='train',
            one_caption_per_image=True,
            ncaption=5,
            max_words=30, 
            prompt=''
        ):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        '''

        if split == 'train':
            filename = 'coco_karpathy_train.json'
        elif split == 'val':
            filename = 'coco_karpathy_val.json'
        else:
            raise NotImplementedError
        
        self.split = split

        self.annotation = json.load(open(os.path.join(image_root, filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.one_caption_per_image = one_caption_per_image
        self.ncaption = ncaption
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        if self.split == 'train':
            caption = self.prompt + pre_caption(
                ann['caption'], 
                self.max_words
            )
            return image, self.tokenizer([caption])[0]

        else:
            if self.one_caption_per_image:
                caption = self.prompt + pre_caption(
                    ann['caption'][0], 
                    self.max_words
                )
                return image, self.tokenizer([caption])[0]
            else:
                captions = []
                for cap in ann['caption']:
                    captions.append(
                        self.prompt + pre_caption(
                            cap, self.max_words)
                    )
                return image, self.tokenizer(captions[:self.ncaption])


class COCOCaptionMerge(Dataset):
    def __init__(self, 
            image_root, transform=None, tokenizer=None, 
            split='train',
            one_caption_per_image=True,
            only_image=False,
            ncaption=5,
            max_words=30, 
            prompt=''
        ):
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        '''

        if split == 'train':
            filename = 'coco_karpathy_train_merge.json'
        elif split == 'val':
            filename = 'coco_karpathy_val.json'
        elif split == 'test':
            filename = 'coco_karpathy_test.json'
        else:
            raise NotImplementedError
        
        self.split = split

        self.annotation = json.load(open(os.path.join(image_root, filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.one_caption_per_image = one_caption_per_image
        self.ncaption = ncaption
        self.only_image = only_image
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root, ann['image'])        
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        
        if self.only_image:
            return image
        
        if self.one_caption_per_image:
            caption = self.prompt + pre_caption(
                ann['caption'][0], 
                self.max_words
            )
            return image, self.tokenizer([caption])[0]
        else:
            captions = []
            for cap in ann['caption']:
                captions.append(
                    self.prompt + pre_caption(
                        cap, self.max_words)
                )
            return image, self.tokenizer(captions[:self.ncaption])
        

def count_captions():
    root = '/public/scccse/dataset/COCO2014'
    filename = 'coco_karpathy_train.json'

    annotations = json.load(open(os.path.join(root, filename),'r'))

    img_ids = {}

    for ann in annotations:
        id_ = ann['image_id']
        if id_ not in img_ids:
            img_ids[id_] = 1
        else:
            img_ids[id_] += 1

    max_ncaption = 0
    min_ncaption = 10000000
    for k, v in img_ids.items():
        if max_ncaption < v:
            max_ncaption = v
        
        if min_ncaption > v:
            min_ncaption = v
    
    print('max_ncaption:', max_ncaption)
    print('min_ncaption:', min_ncaption)
    

    return


def merge_captions():
    root = '/public/scccse/dataset/COCO2014'
    filename = 'coco_karpathy_train.json'

    annotations = json.load(open(os.path.join(root, filename),'r'))

    img_ids = {}

    for ann in annotations:
        id_ = ann['image_id']
        if id_ not in img_ids:
            img_ids[id_] = {
                "image": ann['image'],
                "caption": [ann['caption']],
            }
        else:
            img_ids[id_]["caption"].append(ann['caption'])

    with open('out/coco_karpathy_train_merge.json', 'w') as f:
        json.dump(list(img_ids.values()), f)


if __name__ == '__main__':
    # pass
    merge_captions()
