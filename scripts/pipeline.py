import os
import torch
import torch.nn.functional as F
from PIL import Image
import mmcv
from tqdm import tqdm
from mmcv.utils import print_log
from mmdet.core.visualization.image import imshow_det_bboxes
from mmseg.core import intersect_and_union, pre_eval_to_metrics
from collections import OrderedDict
from prettytable import PrettyTable
import numpy as np
import pycocotools.mask as maskUtils
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
from clip import clip_classification
from clipseg import clipseg_segmentation
from oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation, oneformer_cityscapes_segmentation
from blip import open_vocabulary_classification_blip
from segformer import segformer_segmentation as segformer_func

oneformer_func = {
    'ade20k': oneformer_ade20k_segmentation,
    'coco': oneformer_coco_segmentation,
    'cityscapes': oneformer_cityscapes_segmentation,
    'foggy_driving': oneformer_cityscapes_segmentation
}

def load_filename_with_extensions(data_path, filename):
    """
    Returns file with corresponding extension to json file.
    Raise error if such file is not found.

    Args:
        filename (str): Filename (without extension).

    Returns:
        filename with the right extension.
    """
    full_file_path = os.path.join(data_path, filename)
    # List of image file extensions to attempt
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    # Iterate through image file extensions and attempt to upload the file
    for ext in image_extensions:
        # Check if the file with current extension exists
        if os.path.exists(full_file_path + ext):
            return full_file_path + ext  # Return True if file is successfully uploaded
    raise FileNotFoundError(f"No such file {full_file_path}, checked for the following extensions {image_extensions}")
def semantic_annotation_pipeline(filename, data_path, output_path, rank, save_img=False, scale_small=1.2, scale_large=1.6, scale_huge=1.6,
                                 clip_processor=None,
                                 clip_model=None,
                                 oneformer_ade20k_processor=None,
                                 oneformer_ade20k_model=None,
                                 oneformer_coco_processor=None,
                                 oneformer_coco_model=None,
                                 blip_processor=None,
                                 blip_model=None,
                                 clipseg_processor=None,
                                 clipseg_model=None,
                                 mask_generator=None):
    # 读取图像文件
    img = mmcv.imread(load_filename_with_extensions(data_path, filename))
    
    # 如果没有提供mask生成器，则从json文件中加载注释
    if mask_generator is None:
        anns = mmcv.load(os.path.join(data_path, filename+'.json'))
    else:
        # 否则使用mask生成器生成注释
        anns = {'annotations': mask_generator.generate(img)}
    
    # 初始化位掩码和类名列表
    bitmasks, class_names = [], []
    
    # 使用OneFormer模型进行COCO和ADE20K数据集的分割
    class_ids_from_oneformer_coco = oneformer_coco_segmentation(Image.fromarray(img), oneformer_coco_processor, oneformer_coco_model, rank)
    class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(Image.fromarray(img), oneformer_ade20k_processor, oneformer_ade20k_model, rank)
    
    # 遍历每个注释
    for ann in anns['annotations']:
        # 解码分割掩码并转换为布尔张量
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        
        # 获取有效像素的类ID
        coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
        ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]
        
        # 计算每个类ID的出现频率并获取最常见的类ID
        top_k_coco_propose_classes_ids = torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
        top_k_ade20k_propose_classes_ids = torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
        
        # 初始化本地类名集合
        local_class_names = set()
        
        # 将ADE20K和COCO数据集的类名添加到本地类名集合中
        local_class_names = set.union(local_class_names, set([CONFIG_ADE20K_ID2LABEL['id2label'][str(class_id.item())] for class_id in top_k_ade20k_propose_classes_ids]))
        local_class_names = set.union(local_class_names, set(([CONFIG_COCO_ID2LABEL['refined_id2label'][str(class_id.item())] for class_id in top_k_coco_propose_classes_ids])))
        
        # 根据注释的边界框裁剪图像的不同缩放版本
        patch_small = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                  scale=scale_small)
        patch_large = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                  scale=scale_large)
        patch_huge = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                  scale=scale_huge)
        
        # 裁剪有效掩码
        valid_mask_huge_crop = mmcv.imcrop(valid_mask.numpy(), np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                    scale=scale_huge)
        
        # 使用BLIP模型进行开放词汇分类
        op_class_list = open_vocabulary_classification_blip(patch_large, blip_processor, blip_model, rank)
        
        # 合并本地类名和开放词汇类名
        local_class_list = list(set.union(local_class_names, set(op_class_list)))
        
        # 使用CLIP模型进行分类
        mask_categories = clip_classification(patch_small, local_class_list, 3 if len(local_class_list) > 3 else len(local_class_list), clip_processor, clip_model, rank)
        
        # 使用CLIPSeg模型进行分割
        class_ids_patch_huge = clipseg_segmentation(patch_huge, mask_categories, clipseg_processor, clipseg_model, rank).argmax(0)
        
        # 将裁剪的有效掩码转换为张量
        valid_mask_huge_crop = torch.tensor(valid_mask_huge_crop)
        
        # 如果裁剪的有效掩码和分割结果的形状不匹配，则进行插值
        if valid_mask_huge_crop.shape != class_ids_patch_huge.shape:
            valid_mask_huge_crop = F.interpolate(
                valid_mask_huge_crop.unsqueeze(0).unsqueeze(0).float(),
                size=(class_ids_patch_huge.shape[-2], class_ids_patch_huge.shape[-1]),
                mode='nearest').squeeze(0).squeeze(0).bool()
        
        # 获取裁剪区域中最常见的类ID
        top_1_patch_huge = torch.bincount(class_ids_patch_huge[valid_mask_huge_crop].flatten()).topk(1).indices
        top_1_mask_category = mask_categories[top_1_patch_huge.item()]

        # 将最常见的类名和类提案添加到注释中
        ann['class_name'] = str(top_1_mask_category)
        ann['class_proposals'] = mask_categories
        class_names.append(str(top_1_mask_category))

        # 删除不再需要的变量
        del coco_propose_classes_ids
        del ade20k_propose_classes_ids
        del top_k_coco_propose_classes_ids
        del top_k_ade20k_propose_classes_ids
        del patch_small
        del patch_large
        del patch_huge
        del valid_mask_huge_crop
        del op_class_list
        del mask_categories
        del class_ids_patch_huge
        
    # 将注释结果保存为JSON文件
    mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))
    print('[Save] save SSA-engine annotation results: ', os.path.join(output_path, filename + '_semantic.json'))
    
    # 如果需要保存图像，则显示检测结果并保存
    if save_img:
        for ann in anns['annotations']:
            bitmasks.append(maskUtils.decode(ann['segmentation']))
        imshow_det_bboxes(img,
                        bboxes=None,
                        labels=np.arange(len(bitmasks)),
                        segms=np.stack(bitmasks),
                        class_names=class_names,
                        font_size=25,
                        show=False,
                        out_file=os.path.join(output_path, filename+'_semantic.png'))

    # 删除不再需要的变量
    del img
    del anns
    del class_ids_from_oneformer_coco
    del class_ids_from_oneformer_ade20k

def img_load(data_path, filename, dataset):
    # load image
    if dataset == 'ade20k':
        img = mmcv.imread(os.path.join(data_path, filename+'.jpg'))
    elif dataset == 'cityscapes' or dataset == 'foggy_driving':
        img = mmcv.imread(os.path.join(data_path, filename+'.png'))
    else:
        raise NotImplementedError()
    return img

def semantic_masks_generation(rank, img=None, dataset=None, id2label=None,
                              model='segformer',
                              semantic_branch_processor=None, 
                              semantic_branch_model=None,
                              mask_branch_model=None):
    # 生成初始注释，使用掩码分支模型
    anns = {'annotations': mask_branch_model.generate(img)}
    h, w, _ = img.shape  # 获取图像的高度和宽度
    class_names = []  # 存储类名的列表

    # 根据模型类型选择相应的处理函数
    if model == 'oneformer':
        class_ids = oneformer_func[dataset](Image.fromarray(img), semantic_branch_processor,
                                            semantic_branch_model, rank)
    elif model == 'segformer':
        class_ids = segformer_func(img, semantic_branch_processor, semantic_branch_model, rank)
    else:
        raise NotImplementedError()  # 如果模型类型不支持，抛出异常

    semantic_masks = class_ids.clone()  # 克隆class_ids以创建语义掩码

    # 按区域大小对注释进行排序
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)

    # 遍历每个注释
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()  # 解码并转换为布尔掩码
        propose_classes_ids = class_ids[valid_mask]  # 获取有效像素的类ID
        num_class_proposals = len(torch.unique(propose_classes_ids))  # 计算唯一类ID的数量

        # 如果只有一个类提案
        if num_class_proposals == 1:
            semantic_masks[valid_mask] = propose_classes_ids[0]  # 更新语义掩码
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]  # 获取类名
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]  # 设置类提案
            class_names.append(ann['class_name'])  # 添加类名到列表
            continue  # 跳过剩余步骤

        # 获取出现次数最多的类ID
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantic_masks[valid_mask] = top_1_propose_class_ids  # 更新语义掩码
        ann['class_name'] = top_1_propose_class_names[0]  # 设置类名
        ann['class_proposals'] = top_1_propose_class_names[0]  # 设置类提案
        class_names.append(ann['class_name'])  # 添加类名到列表

        # 删除不再需要的变量
        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
        del top_1_propose_class_names

    sematic_classes_in_img = torch.unique(semantic_masks)  # 获取图像中的唯一类

    return semantic_masks, sematic_classes_in_img

def get_specific_mask(mask_name, semantc_masks, sematic_classes_in_img, label_dict):
    target_mask = []

    for i in range(len(sematic_classes_in_img)):
        class_name = label_dict[str(sematic_classes_in_img[i].item())]  # 获取类名
        if class_name == mask_name:
            target_mask = semantc_masks == sematic_classes_in_img[i]  # 创建类掩码
            target_mask = target_mask.cpu().numpy().astype(np.uint8)  # 转换为numpy数组

    return target_mask



def semantic_segment_anything_inference(filename, output_path, rank, img=None, save_img=False,
                                        semantic_branch_processor=None,
                                        semantic_branch_model=None,
                                        mask_branch_model=None,
                                        dataset=None,
                                        id2label=None,
                                        model='segformer'):

    # 生成初始注释，使用掩码分支模型
    anns = {'annotations': mask_branch_model.generate(img)}
    h, w, _ = img.shape  # 获取图像的高度和宽度
    class_names = []  # 存储类名的列表

    # 根据模型类型选择相应的处理函数
    if model == 'oneformer':
        class_ids = oneformer_func[dataset](Image.fromarray(img), semantic_branch_processor,
                                            semantic_branch_model, rank)
    elif model == 'segformer':
        class_ids = segformer_func(img, semantic_branch_processor, semantic_branch_model, rank)
    else:
        raise NotImplementedError()  # 如果模型类型不支持，抛出异常

    semantc_mask = class_ids.clone()  # 克隆class_ids以创建语义掩码

    # 按区域大小对注释进行排序
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)

    # 遍历每个注释
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()  # 解码并转换为布尔掩码
        propose_classes_ids = class_ids[valid_mask]  # 获取有效像素的类ID
        num_class_proposals = len(torch.unique(propose_classes_ids))  # 计算唯一类ID的数量

        # 如果只有一个类提案
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]  # 更新语义掩码
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]  # 获取类名
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]  # 设置类提案
            class_names.append(ann['class_name'])  # 添加类名到列表
            continue  # 跳过剩余步骤

        # 获取出现次数最多的类ID
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantc_mask[valid_mask] = top_1_propose_class_ids  # 更新语义掩码
        ann['class_name'] = top_1_propose_class_names[0]  # 设置类名
        ann['class_proposals'] = top_1_propose_class_names[0]  # 设置类提案
        class_names.append(ann['class_name'])  # 添加类名到列表

        # 删除不再需要的变量
        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
        del top_1_propose_class_names

    sematic_class_in_img = torch.unique(semantc_mask)  # 获取图像中的唯一类
    semantic_bitmasks, semantic_class_names = [], []  # 初始化语义掩码和类名列表

    # 语义预测
    anns['semantic_mask'] = {}
    for i in range(len(sematic_class_in_img)):
        class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]  # 获取类名
        class_mask = semantc_mask == sematic_class_in_img[i]  # 创建类掩码
        class_mask = class_mask.cpu().numpy().astype(np.uint8)  # 转换为numpy数组
        semantic_class_names.append(class_name)  # 添加类名到列表
        semantic_bitmasks.append(class_mask)  # 添加掩码到列表
        anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
        anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')

    sky_mask_image = np.zeros((class_mask.shape[0], class_mask.shape[1], 3), dtype=np.uint8)
    for i in range(len(sematic_class_in_img)):
        class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]  # 获取类名
        if class_name == 'sky':
            class_mask = semantc_mask == sematic_class_in_img[i]  # 创建类掩码
            class_mask = class_mask.cpu().numpy().astype(np.uint8)  # 转换为numpy数组
            sky_mask_image[class_mask == 1] = [255, 255, 255]
            sky_mask_image[class_mask == 0] = [0, 0, 0]
            class_mask_image = Image.fromarray(sky_mask_image)
            class_mask_image.save(os.path.join(output_path, f"{filename}_class_{semantic_class_names[i]}.png"))
            

    # 如果需要保存图像
    if save_img:
        imshow_det_bboxes(img,
                          bboxes=None,
                          labels=np.arange(len(sematic_class_in_img)),
                          segms=np.stack(semantic_bitmasks),
                          class_names=semantic_class_names,
                          font_size=25,
                          show=False,
                          out_file=os.path.join(output_path, filename + '_semantic.png'))
        print('[Save] save SSA prediction: ', os.path.join(output_path, filename + '_semantic.png'))
        # 保存注释结果为JSON文件
        mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))

    # 手动清理不再需要的变量
    del img
    del anns
    del class_ids
    del semantc_mask
    del class_names
    del semantic_bitmasks
    del semantic_class_names


def eval_pipeline(gt_path, res_path, dataset):
    logger = None
    if dataset == 'cityscapes' or dataset == 'foggy_driving':
        class_names = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
    elif dataset == 'ade20k':
        class_names = ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag')
    file_client = mmcv.FileClient(**{'backend': 'disk'})
    pre_eval_results = []
    if dataset == 'cityscapes':
        prefixs = ['frankfurt','lindau','munster']
    elif dataset == 'foggy_driving':
        prefixs = ['public', 'pedestrian']
    elif dataset == 'ade20k':
        prefixs = ['']
    else:
        raise NotImplementedError
    for split in tqdm(prefixs, desc="Split loop"):
        gt_path_split = os.path.join(gt_path, split)
        res_path_split = os.path.join(res_path, split)
        filenames = [fn_ for fn_ in os.listdir(res_path_split) if '.json' in fn_]
        for i, fn_ in enumerate(tqdm(filenames, desc="File loop")):
            pred_fn = os.path.join(res_path_split, fn_)
            result = mmcv.load(pred_fn)
            num_classes = len(class_names)
            init_flag = True
            for id_str, mask in result['semantic_mask'].items():
                mask_ = maskUtils.decode(mask)
                h, w = mask_.shape
                if init_flag:
                    seg_mask = torch.zeros((1, 1, h, w))
                    init_flag = False
                mask_ = torch.from_numpy(mask_).unsqueeze(0).unsqueeze(0)
                seg_mask[mask_] = int(id_str)
            seg_logit = torch.zeros((1, num_classes, h, w))
            seg_logit.scatter_(1, seg_mask.long(), 1)
            seg_logit = seg_logit.float()
            seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()
            if dataset == 'cityscapes' or dataset == 'foggy_driving':
                gt_fn_ = os.path.join(gt_path_split, fn_.replace('_leftImg8bit_semantic.json','_gtFine_labelTrainIds.png'))
            elif dataset == 'ade20k':
                gt_fn_ = os.path.join(gt_path, fn_.replace('_semantic.json','.png'))
            img_bytes = file_client.get(gt_fn_)
            seg_map = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend='pillow').squeeze().astype(np.uint8)
            if dataset=='ade20k':
                seg_map = seg_map - 1
            pre_eval_results.append(intersect_and_union(
                            seg_pred,
                            seg_map,
                            num_classes,
                            255,
                            label_map=dict(),
                            reduce_zero_label=False))

    ret_metrics = pre_eval_to_metrics(pre_eval_results, ['mIoU'])
    ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print_log('per class results:', logger)
    print_log('\n' + class_table_data.get_string(), logger=logger)
    print_log('Summary:', logger)
    print_log('\n' + summary_table_data.get_string(), logger=logger)
