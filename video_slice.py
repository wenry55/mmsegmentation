from mmseg.apis import inference_segmentor, init_segmentor
from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, vis_pose_tracking_result)
from mmdet.apis import inference_detector, init_detector

import mmcv
import os
import cv2
from argparse import ArgumentParser
import pathlib
from datetime import datetime
import json

zoo = {
    'pspnet': {
        'config':
        'configs/pspnet/pspnet_r101-d8_512x512_40k_voc12aug.py',
        'checkpoint':
        'checkpoints/pspnet_r101-d8_512x512_40k_voc12aug_20200613_161222-bc933b18.pth'
    },
    'deeplabv3': {
        'config':
        'configs/deeplabv3/deeplabv3_r101-d8_512x512_40k_voc12aug.py',
        'checkpoint':
        'checkpoints/deeplabv3_r101-d8_512x512_40k_voc12aug_20200613_161432-0017d784.pth'
    },
    'dlc59': {
        'config':
        'configs/deeplabv3/deeplabv3_r101-d8_480x480_80k_pascal_context_59.py',
        'checkpoint':
        'checkpoints/deeplabv3_r101-d8_480x480_80k_pascal_context_59_20210416_113002-26303993.pth'
    },
    'psanet': {
        'config':
        'configs/psanet/psanet_r101-d8_512x512_20k_voc12aug.py',
        'checkpoint':
        'checkpoints/psanet_r101-d8_512x512_20k_voc12aug_20200617_110624-946fef11.pth'
    }
}


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 1 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def get_model_files(zoo_id):
    return zoo[zoo_id]['config'], zoo[zoo_id]['checkpoint']


def init_video(args):
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('video fps = ', fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(
        os.path.join(args.out_video_root,
                     f'vis_{os.path.basename(args.video_path)}'), fourcc, fps,
        size)
    return cap, fps, videoWriter


# optional
return_heatmap = False

# e.g. use ('backbone', ) to return backbone feature
output_layer_names = None


def main():
    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--out-video-root', type=str, default='.', help='Output directory')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')

    args = parser.parse_args()

    save_out_video = True

    config_file, checkpoint_file = get_model_files('psanet')

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    tot_frames = 0

    video = mmcv.VideoReader(args.video_path)
    cap = None
    fps = None
    if save_out_video:
        cap, fps, videoWriter = init_video(args)

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())
    dataset = pose_model.cfg.data['test']['type']

    for frame in video:
        tot_frames += 1
        if tot_frames % (fps / 2) == 0:

            img = cv2.imread(fpath, cv2.IMREAD_COLOR)

            mmdet_results = inference_detector(det_model, img)

            person_results = process_mmdet_results(mmdet_results,
                                                   args.det_cat_id)

            # test a single image, with a list of bboxes.
            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                img,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            result = inference_segmentor(model, frame)
            fr_outfile = os.path.join(
                args.out_video_root,
                pathlib.Path(args.video_path).stem + ".jpg")
            fr_resfile = os.path.join(
                args.out_video_root,
                pathlib.Path(args.video_path).stem + ".json")
            vis_img = model.show_result(
                frame, result, opacity=0.5, out_file=f'{fr_outfile}')
            with open(fr_resfile, 'w', encoding='utf8') as json_file:
                json.dump(result, json_file, ensure_ascii=False)

            videoWriter.write(vis_img)
            if tot_frames % 100 == 0:
                print(datetime.now(), tot_frames)

    cap.release()
    if save_out_video:
        videoWriter.release()


if __name__ == '__main__':
    main()
