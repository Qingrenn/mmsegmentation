# Copyright (c) OpenMMLab. All rights reserved.
import logging
import sys
import traceback
from argparse import ArgumentParser
from functools import partial
from typing import Callable, Optional

import cv2
import torch.multiprocessing as mp
from mmengine.logging import MMLogger
from mmengine.model.utils import revert_sync_batchnorm
from torch.multiprocessing import Process, set_start_method
from tqdm import tqdm

from mmseg.apis import inference_model, init_model
from mmseg.apis.inference import show_result_pyplot


def target_wrapper(target: Callable,
                   log_level: int,
                   ret_value: Optional[mp.Value] = None,
                   *args,
                   **kwargs):
    """The wrapper used to start a new subprocess.

    Args:
        target (Callable): The target function to be wrapped.
        log_level (int): Log level for logging.
        ret_value (mp.Value): The success flag of target.

    Return:
        Any: The return of target.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S')
    logger.level
    logger.setLevel(log_level)

    if ret_value is not None:
        ret_value.value = -1
    try:
        target(*args, **kwargs)
        if ret_value is not None:
            ret_value.value = 0
    except Exception as e:
        logger.error(e)
        traceback.print_exc(file=sys.stdout)


def create_process(name, target, args, kwargs, ret_value=None):
    logger = MMLogger.get_instance(name)
    logger.info(f'{name} start')
    wrap_func = partial(target_wrapper, target, logger.level, ret_value)
    process = Process(target=wrap_func, args=args, kwargs=kwargs)
    return process


def blend_frame(model, blend_queue, output_queue, show, show_wait_time):
    while True:
        frame, result = blend_queue.get(block=True, timeout=5)
        if frame is None:
            break
        # if `show` is True,
        # this line will block when there isn't a display device
        draw_img = show_result_pyplot(model, frame, result, show=show)
        output_queue.put(draw_img)
        if show:
            cv2.imshow('video_demo', draw_img)
            cv2.waitKey(show_wait_time)
    output_queue.put(None)  # end signal


def output_video(output_queue, output_fourcc, output_fps, output_height,
                 output_width, output_file):
    # init output video
    fourcc = cv2.VideoWriter_fourcc(*output_fourcc)
    writer = cv2.VideoWriter(output_file, fourcc, output_fps,
                             (output_width, output_height), True)

    while True:
        draw_img = output_queue.get(block=True, timeout=5)
        if draw_img is None:
            break
        if draw_img.shape[0] != output_height or draw_img.shape[
                1] != output_width:
            draw_img = cv2.resize(draw_img, (output_width, output_height))
        writer.write(draw_img)


def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file or webcam id')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output-file', default=None, type=str, help='Output video file path')
    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--output-fps', default=-1, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()
    assert args.show or args.output_file, \
        'At least one output should be enabled.'

    set_start_method('spawn', force=True)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # build input video
    if args.video.isdigit():
        args.video = int(args.video)
    cap = cv2.VideoCapture(args.video)
    assert (cap.isOpened())
    input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # communication between processes
    blend_queue = mp.Queue(maxsize=2)
    output_queue = mp.Queue(maxsize=2)

    # create blend process
    blend_process_flag = mp.Value('i', -1)
    blend_process = create_process(
        'blend_frame',
        target=blend_frame,
        args=(
            model,
            blend_queue,
            output_queue,
        ),
        kwargs={
            'show': args.show,
            'show_wait_time': args.show_wait_time
        },
        ret_value=blend_process_flag)
    blend_process.start()

    # create output process
    if args.output_file is not None:
        output_process_flag = mp.Value('i', -1)
        output_process = create_process(
            'output_video',
            target=output_video,
            args=(output_queue, ),
            kwargs={
                'output_file':
                args.output_file,
                'output_fourcc':
                args.output_fourcc,
                'output_fps':
                args.output_fps if args.output_fps > 0 else input_fps,
                'output_height':
                args.output_height
                if args.output_height > 0 else int(input_height),
                'output_width':
                args.output_width
                if args.output_width > 0 else int(input_width)
            },
            ret_value=output_process_flag)
        output_process.start()

    # start looping
    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    try:
        while True:
            flag, frame = cap.read()
            if not flag:
                break

            # test a single image
            result = inference_model(model, frame)
            blend_queue.put((frame, result))
            pbar.update(1)
    finally:
        blend_queue.put((None, None))  # end signal
        cap.release()

    # block until subprocesses finished
    blend_process.join()
    if blend_process_flag is not None:
        logger = MMLogger.get_instance('blend_frame')
        if blend_process_flag.value != 0:
            logger.error('blend failed.')
            exit(1)
        else:
            logger.info('blend succeeded.')

    if args.output_file is not None:
        output_process.join()
        if output_process_flag is not None:
            logger = MMLogger.get_instance('output_video')
            if output_process_flag.value != 0:
                logger.error('output failed.')
                exit(1)
            else:
                logger.info('output succeeded.')


if __name__ == '__main__':
    import time
    t0 = time.time()
    main()
    t1 = time.time()
    print(t1 - t0)
