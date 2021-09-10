from mmseg.apis import inference_segmentor, init_segmentor
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

    for frame in video:
        tot_frames += 1
        if tot_frames % (fps / 2) == 0:
            result = inference_segmentor(model, frame)
            fr_outfile = os.path.join(args.out_video_root, pathlib.Path(args.video_path).stem + ".jpg")
            fr_resfile = os.path.join(args.out_video_root, pathlib.Path(args.video_path).stem + ".json")
            vis_img = model.show_result(frame, result, opacity=0.5, out_file=f'{fr_outfile}')
            with open(fr_resfile, 'w', encoding ='utf8') as json_file:
                json.dump(result, json_file, ensure_ascii = False)

            videoWriter.write(vis_img)
            if tot_frames % 100 == 0:
                print(datetime.now(), tot_frames)

    cap.release()
    if save_out_video:
        videoWriter.release()


if __name__ == '__main__':
    main()
