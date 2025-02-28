set -e

python3 scripts/tester.py --input_folder_path /mnt/nas/perception/yike/validation/temp/ \
                          --output_folder_path /mnt/nas/perception/yike/validation/segmentation/ \
                          --ckpt_path /home/wangyike/Workspace/playground/segmentation/Semantic-Segment-Anything/ckp/sam_vit_h_4b8939.pth \
                          --class_name sky \
                          --world_size 1 \
                          --dataset cityscapes \
                          --model segformer \
                          --save_viz