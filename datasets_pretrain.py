import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
from decord import VideoReader, cpu
import json
import pathlib
from torchvision import transforms
from config import pose_dirs

# load sub-pose
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]

            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[...,2]<=thr] = 0
            
        kps_with_scores[part] = torch.tensor(result)
        
    return kps_with_scores


# input: T, N, 3
# input is un-normed joints
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])

    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]


# build base dataset
class Base_Dataset(Dataset.Dataset):
    def collate_fn(self, batch):
        src_length_batch,name_batch,pose_tmp = [],[],[]
        pre_pose_ids_batch, header_ids_batch, = [], []
        gt_text_batch = []
        
        for name_sample, pose_sample, pre_pose_ids, header_ids, gt_text in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            pre_pose_ids_batch.append(pre_pose_ids)
            header_ids_batch.append(header_ids)
            gt_text_batch.append(gt_text)

        pose_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            pose_input[key] = img_batch
            if 'attention_mask' not in pose_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                pose_input['pose_attention_mask'] = img_padding_mask

                pose_input['name_batch'] = name_batch
                pose_input['src_length_batch'] = src_length_batch

        text_input = {}
        text_input['gt_text'] = gt_text_batch
        
        
        # Ensure Tensor obj in list
        pre_ids_tensors = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in pre_pose_ids_batch]
        header_ids_tensor = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in header_ids_batch]

        text_input['header_ids'] = torch.stack(header_ids_tensor, dim=0)
        text_input['pre_pose_ids'] = torch.stack(pre_ids_tensors, dim=0)
        text_input['pre_attn_mask'] = torch.ones_like(text_input['pre_pose_ids'])

        return pose_input, text_input
    
    def collate_fn_train(self, batch):
        src_length_batch,name_batch,pose_tmp = [],[],[]
        pre_pose_ids_batch, post_pose_ids_batch= [], []
        gt_text_batch, lexical_mask_batch = [], []
        
        for name_sample, pose_sample, pre_pose_ids, post_pose_ids, lexical_mask, gt_text in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            pre_pose_ids_batch.append(pre_pose_ids)
            post_pose_ids_batch.append(post_pose_ids)
            gt_text_batch.append(gt_text)
            lexical_mask_batch.append(lexical_mask)

        pose_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            pose_input[key] = img_batch
            if 'attention_mask' not in pose_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                pose_input['pose_attention_mask'] = img_padding_mask

                pose_input['name_batch'] = name_batch
                pose_input['src_length_batch'] = src_length_batch

        text_input = {}
        text_input['gt_text'] = gt_text_batch
        
        
        # Ensure Tensor obj in list
        pre_ids_tensors = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in pre_pose_ids_batch]
        post_ids_tensors = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in post_pose_ids_batch]
        lexical_mask_tensors = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in lexical_mask_batch]

        text_input['post_pose_ids'] = pad_sequence(post_ids_tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        text_input['lexical_mask'] = pad_sequence(lexical_mask_tensors, batch_first=True, padding_value=False)

        text_input['pre_pose_ids'] = torch.stack(pre_ids_tensors, dim=0)
        text_input['pre_attn_mask'] = torch.ones_like(text_input['pre_pose_ids'])
        
        # attention_mask for translation
        text_lens = [len(x) for x in post_ids_tensors]
        text_mask_list = [torch.ones(l, dtype=torch.long) for l in text_lens]
        text_input['post_attn_mask'] = pad_sequence(text_mask_list, batch_first=True, padding_value=1)

        return pose_input, text_input


class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase, tokenizer, mask_id=126336, language="Chinese"):
        super(S2T_Dataset, self).__init__()
        self.args = args
        self.max_length = args.max_length
        self.json_data = json.load(open(path, 'r', encoding='utf-8'))
        self.phase = phase
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.language = language

        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[args.dataset]

        elif "How2Sign" in self.args.dataset:
            if phase == 'dev':
                raise NotImplementedError("How2Sign dev set is not supported")
            self.pose_dir = pose_dirs[args.dataset].format(phase)
        else:
            raise NotImplementedError

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

    def __len__(self):
        return len(self.json_data)
    
    def __getitem__(self, index):
        sample = self.json_data[index]

        raw_text = sample['text']
        name_sample = sample['name']
        pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])
        
        # construct prompt and translation
        pre_pose_text = f"<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nTranslate sign language video to {self.language}:"
        pre_pose_ids = self.tokenizer(pre_pose_text, add_special_tokens=False)['input_ids']

        post_pose_text = f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{raw_text}<|endoftext|>"
        post_pose_ids = self.tokenizer(post_pose_text, padding=True, truncation=True, max_length=100, add_special_tokens=False)['input_ids']

        HEADER_TOKEN_LEN = 7
        header_ids = post_pose_ids[:HEADER_TOKEN_LEN]
        content_ids = post_pose_ids[HEADER_TOKEN_LEN:]

        if self.phase == "train":
            mask_ids = sample['mask_id']
            lexical_mask = [x not in mask_ids or x == self.mask_id for x in post_pose_ids[HEADER_TOKEN_LEN:]]

            return name_sample, pose_sample, pre_pose_ids, post_pose_ids, lexical_mask, raw_text
        else:

            return name_sample, pose_sample, pre_pose_ids, header_ids, raw_text

    
    def load_pose(self, path):
        pose = pickle.load(open(os.path.join(self.pose_dir, path.replace(".mp4", '.pkl')), 'rb'))
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0
                
        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)
            
        return kps_with_scores, None

    def __str__(self):
        return f'#total {len(self)}'

class S2T_Dataset_news(Base_Dataset):
    def __init__(self, path, args, phase, tokenizer, mask_id=126336, language="Chinese"):
        super(S2T_Dataset_news, self).__init__()
        self.args = args
        self.phase = phase
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.language = language
        self.max_length = args.max_length

        path = pathlib.Path(path)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
       
        if self.args.dataset == "CSL_News":
            self.pose_dir = pose_dirs[args.dataset]
        elif self.args.dataset == "YoutubeASL":
            self.pose_dir = pose_dirs[args.dataset]
        else:
            raise NotImplementedError
        sum_sample = len(self.annotation)
        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

        if phase == 'train':
            self.start_idx = int(sum_sample * 0.0)
            self.end_idx = int(sum_sample * 0.99)
        else:
            self.start_idx = int(sum_sample * 0.99)
            self.end_idx = int(sum_sample)
        
    def __len__(self):
        return self.end_idx - self.start_idx
        
    def __getitem__(self, index):
        num_retries = 10  

        # skip some invalid video sample
        success = False
        for _ in range(num_retries):
            sample = self.annotation[self.start_idx:self.end_idx][index]

            raw_text = sample['text']
            name_sample = sample['video']
           
            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'])
    
            except:
                import traceback

                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            
            success = True
            break
        
        if not success:
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")

        # construct prompt and translation
        pre_pose_text = f"<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nTranslate sign language video to {self.language}:"
        pre_pose_ids = self.tokenizer(pre_pose_text, add_special_tokens=False)['input_ids']

        post_pose_text = f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{raw_text}<|endoftext|>"
        post_pose_ids = self.tokenizer(post_pose_text, add_special_tokens=False)['input_ids']

        HEADER_TOKEN_LEN = 7
        header_ids = post_pose_ids[:HEADER_TOKEN_LEN]
        content_ids = post_pose_ids[HEADER_TOKEN_LEN:]

        if self.phase == "train":
            mask_ids = sample['mask_id']
            lexical_mask = [x not in mask_ids or x == self.mask_id for x in post_pose_ids[HEADER_TOKEN_LEN:]]

            return name_sample, pose_sample, pre_pose_ids, post_pose_ids, lexical_mask, raw_text
        else:

            return name_sample, pose_sample, pre_pose_ids, header_ids, raw_text
    
    def load_pose(self, pose_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), 'rb'))
        
        duration = len(pose['scores'])

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp)
        
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
                
        kps_with_scores = load_part_kp(skeletons, confs)

        return kps_with_scores, None

    def __str__(self):
        return f'#total {len(self)}'
