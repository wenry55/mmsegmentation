from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
import cv2
from argparse import ArgumentParser



def main():
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')

    args = parser.parse_args()

    save_out_video=True

    config_file = 'configs/pspnet/pspnet_r101-d8_512x512_40k_voc12aug.py'
    checkpoint_file = 'checkpoints/pspnet_r101-d8_512x512_40k_voc12aug_20200613_161222-bc933b18.pth'

    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    #img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
    #result = inference_segmentor(model, img)
    #model.show_result(img, result, out_file='result.jpg', opacity=0.5)

    # test a video and show the results
    
    video = mmcv.VideoReader('video.mp4')
    cap = None
    if save_out_video:
        cap = cv2.VideoCapture(args.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(os.path.join(args.out_video_root, f'vis_{os.path.basename(args.video_path)}'), fourcc, fps, size)

    for frame in video:
        result = inference_segmentor(model, frame)
        vis_img = model.show_result(frame, result, opacity=0.5)
        if save_out_video:
            videoWriter.write(vis_img)

    cap.release()
    if save_out_video:
        videoWriter.release()

if __name__ == '__main__':
    main()
