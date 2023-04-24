import torch
import decord
import mmengine
import gradio as gr
import numpy as np

from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.visualization import ActionVisualizer

CONFIG = 'configs/tsn_r50.py'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = 'https://download.openmmlab.com/mmaction/v1.0/recognition/' \
             'tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/' \
             'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'

recognizer = init_recognizer(CONFIG, CHECKPOINT, device=DEVICE)

CLASSES = mmengine.list_from_file('demo/label_map_k400.txt')
visualizer = ActionVisualizer()
visualizer.dataset_meta = dict(classes=CLASSES)


def setup_text_cfg(w, h):

    def _get_adaptive_scale(img_shape, min_scale=0.3, max_scale=3.0):
        short_edge_length = min(img_shape)
        scale = short_edge_length / 224.
        return min(max(scale, min_scale), max_scale)

    img_scale = _get_adaptive_scale((w, h))
    text_cfg = dict(
        positions=np.array([(w - 5 * img_scale, img_scale * 5)]).astype(np.int32),
        font_sizes=int(img_scale * 7),
        vertical_alignments='top',
        horizontal_alignments='right')

    return text_cfg


def predict(video_input):
    vr = decord.VideoReader(video_input)
    anno = dict(
        video_reader=vr,
        total_frames=len(vr),
        label=-1,
        start_index=0,
        modality='RGB')
    video_frames = [f.asnumpy()[..., ::-1] for f in vr]
    pred_result = inference_recognizer(recognizer, anno)

    h, w = video_frames[0].shape[:2]
    text_cfg = setup_text_cfg(w, h)
    visualizer.add_datasample(
        name='x.mp4',
        video=video_frames,
        data_sample=pred_result,
        draw_gt=False,
        draw_pred=True,
        text_cfg=text_cfg,
        fps=30,
        out_type='video',
        out_path='demo/x.mp4')
    return 'demo/x.mp4'


fire = gr.Interface(
    fn=predict,
    title='MMAction2 Action Recognition',
    description="Recognize 400 action categories defined within the Kinetics400 dataset. ",
    inputs=gr.Video(),
    outputs=gr.Video(),
    examples=['demo/zelda.mp4', 'demo/shaowei.mp4', 'demo/baoguo.mp4', 'demo/cxk.mp4'])

if __name__ == '__main__':
    fire.launch()
