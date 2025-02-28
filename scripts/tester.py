import os
import argparse
import torch
import mmcv
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

from pipeline import semantic_segment_anything_inference, eval_pipeline, img_load, get_specific_mask, semantic_masks_generation
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.cityscapes_id2label import CONFIG as CONFIG_CITYSCAPES_ID2LABEL

import torch.distributed as dist
import torch.multiprocessing as mp

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12322'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def parse_args():
    parser = argparse.ArgumentParser(description='Semantically segment anything.')
    parser.add_argument('--input_folder_path', type=str, help='Specify the input folder to load images')
    parser.add_argument('--output_folder_path', type=str, help='Specify the output folder to save rgb image and segmented masks')
    parser.add_argument('--ckpt_path', default='ckp/sam_vit_h_4b8939.pth', help='specify the root path of SAM checkpoint')
    parser.add_argument('--class_name', default='sky', help='specify the target class for mask')
    parser.add_argument('--world_size', type=int, default=0, help='number of nodes')
    parser.add_argument('--dataset', type=str, default='ade20k', choices=['ade20k', 'cityscapes', 'foggy_driving'], help='specify the set of class names')
    parser.add_argument('--model', type=str, default='segformer', choices=['oneformer', 'segformer'], help='specify the semantic branch model')
    parser.add_argument('--save_viz', default=False, action='store_true', help='whether to save visualizations')

    args = parser.parse_args()
    return args

def init_semantic_branch(model, dataset, rank):
    # initiate the semantic branch 
    if model == 'oneformer':
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        if dataset == 'ade20k':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large").to(rank)
        elif dataset == 'cityscapes':
            semantic_branch_processor = OneFormerProcessor.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_swin_large").to(rank)
        elif dataset == 'foggy_driving':
            semantic_branch_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_cityscapes_dinat_large")
            semantic_branch_model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_cityscapes_dinat_large").to(rank)
        else:
            raise NotImplementedError()
    elif model == 'segformer':
        from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
        if dataset == 'ade20k':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640").to(rank)
        elif dataset == 'cityscapes' or args.dataset == 'foggy_driving':
            semantic_branch_processor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
            semantic_branch_model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(rank)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    print('[Model loaded] Semantic branch (your own segmentor) is loaded.')

    return semantic_branch_processor, semantic_branch_model

def init_mask_branch(ckpt_path, rank):
    # initiate the mask branch
    sam = sam_model_registry["vit_h"](checkpoint=ckpt_path).to(rank)
    mask_branch_model = SamAutomaticMaskGenerator( model=sam,
                                                   points_per_side=128 if args.dataset == 'foggy_driving' else 64,
                                                   pred_iou_thresh=0.86,
                                                   stability_score_thresh=0.92,
                                                   crop_n_layers=1,
                                                   crop_n_points_downscale_factor=2,
                                                   min_mask_region_area=100,  # Requires open-cv to run post-processing
                                                   output_mode='coco_rle', )

    print('[Model loaded] Mask branch (SAM) is loaded.')

    return sam, mask_branch_model


class SemanticMaskGenerator:
    def __init__(self, input_folder_path, output_folder_path, ckpt_path, class_name, dataset, model, world_size, save_viz, rank):
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.ckpt_path = ckpt_path
        self.class_name = class_name
        self.dataset = dataset
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.save_viz = save_viz

        print(f"Initializing mask branch!")
        self.sam, self.mask_branch_model = init_mask_branch(self.ckpt_path, self.rank)
        print(f"Initializing semantic branch!")
        self.semantic_branch_processor, self.semantic_branch_model = init_semantic_branch(self.model, self.dataset, rank)

    def run(self):
        sequences = [seq for seq in os.listdir(self.input_folder_path) if os.path.isdir(os.path.join(self.input_folder_path, seq))]

        sub_seqs = sequences[(len(sequences) // args.world_size + 1) * self.rank : (len(sequences) // args.world_size + 1) * (self.rank + 1)]

        for seq in tqdm(sub_seqs, desc="Processing sequences"):
            seq_path = os.path.join(self.input_folder_path, seq)
            rgb_image_folder = os.path.join(seq_path, 'input_rgb_0')
            rgb_image_names = [f for f in os.listdir(rgb_image_folder) if f.endswith('.png')]

            for rgb_image_name in tqdm(rgb_image_names, desc="  Processing image frame"):
                rgb_image = mmcv.imread(os.path.join(rgb_image_folder, rgb_image_name))

                if self.dataset == 'ade20k':
                    id2label = CONFIG_ADE20K_ID2LABEL
                elif self.dataset == 'cityscapes' or self.dataset == 'foggy_driving':
                    id2label = CONFIG_CITYSCAPES_ID2LABEL
                else:
                    raise NotImplementedError()
                
                with torch.no_grad():
                    # 获取id到标签的映射字典
                    label_dict = id2label['id2label'] 
                    # 生成语义掩码和图像中的语义类
                    semantic_masks, sematic_classes_in_img = semantic_masks_generation(self.rank, img=rgb_image, dataset=self.dataset, 
                                                                                       id2label=id2label, model=self.model,
                                                                                       semantic_branch_processor=self.semantic_branch_processor, 
                                                                                       semantic_branch_model=self.semantic_branch_model,
                                                                                       mask_branch_model=self.mask_branch_model)  
                    # 获取特定类的掩码
                    target_mask = get_specific_mask(self.class_name, semantic_masks, sematic_classes_in_img, label_dict)  
                    # 掩码输出路径
                    mask_output_path = os.path.join(self.output_folder_path, seq, 'mask')  
                    # 如果路径不存在，则创建
                    if not os.path.exists(mask_output_path):  
                        os.makedirs(mask_output_path)
                    # 保存掩码为.npy文件
                    np.save(os.path.join(mask_output_path, rgb_image_name.replace('.png', '.npy')), target_mask)  

                    # semantic_segment_anything_inference(rgb_image_name, 
                    #                                     self.output_folder_path, 
                    #                                     self.rank, 
                    #                                     img=rgb_image, 
                    #                                     save_img=self.save_viz,
                    #                                     semantic_branch_processor=self.semantic_branch_processor,
                    #                                     semantic_branch_model=self.semantic_branch_model,
                    #                                     mask_branch_model=self.mask_branch_model,
                    #                                     dataset=self.dataset,
                    #                                     id2label=id2label,
                    #                                     model=self.model)
                    



def main(rank, args):
    print(f"Main started!")
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    
    print(f"SemanticMaskGenerator!")
    semantic_mask_generator = SemanticMaskGenerator(args.input_folder_path, 
                                                    args.output_folder_path, 
                                                    args.ckpt_path,
                                                    args.class_name,
                                                    args.dataset,
                                                    args.model,
                                                    args.world_size,
                                                    args.save_viz,
                                                    rank)

    semantic_mask_generator.run()

    print(f"Main finished!")




if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output_folder_path):
        os.mkdir(args.output_folder_path)
    if args.world_size > 1:
        mp.spawn(main, args=(args,), nprocs=args.world_size, join=True)
    else:
        main(0, args)