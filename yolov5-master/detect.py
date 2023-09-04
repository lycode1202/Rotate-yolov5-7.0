# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from models.experimental import attempt_load

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh, apply_classifier, set_logging, save_one_box, rotate_non_max_suppression, rotate_scale_coords)
from utils.torch_utils import select_device, smart_inference_mode
from utils.plots import colors, plot_one_box, polygon_plot_one_box, rotate_plot_one_box
import subprocess as sp

# 获取GPU显存
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND1 = "nvidia-smi --query-gpu=memory.free --format=csv"
    COMMAND2 = "nvidia-smi --query-gpu=memory.used --format=csv"
    COMMAND3 = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND1.split()))[1:]
    memory_free_values = [int(x.split()[0])/1024 for i, x in enumerate(memory_free_info)]
    memory_used_info = _output_to_list(sp.check_output(COMMAND2.split()))[1:]
    memory_used_values = [int(x.split()[0])/1024 for i, x in enumerate(memory_used_info)]
    memory_total_info = _output_to_list(sp.check_output(COMMAND3.split()))[1:]
    memory_total_values = [int(x.split()[0])/1024 for i, x in enumerate(memory_total_info)]
    print(f'{"Used":18s}\t{"Free":18s}\t{"Total":18s}')
    for free, used, total in zip(memory_free_values, memory_used_values, memory_total_values):
        print(f'{used:.3f} {"GB": <6s}{used/total:<9.2%}\t{free:.3f} {"GB": <6s}{free/total:<9.2%}\t{total:.2f}')



@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型权重文件地址
        source=ROOT / 'data/images',  # 种类：file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # 数据集配置文件
        imgsz=(640, 640),  # 待检测图片缩放大小 (height, width)
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU阈值
        max_det=1000,  # 每张图片最大推理数
        device='',  # 设备：cuda|cpu
        view_img=False,  # 展示结果
        save_txt=False,  # 以txt文件保存运行结果
        save_conf=False,  # 将置信度也保存到txt文件中
        save_crop=False,  # 保存裁剪的预测框
        nosave=False,  # 不保存图片和影像
        classes=None,  # 通过class过滤，即指定预测什么种类（可以预测猫，也可以预测狗，但是指定一个之后只会预测对应的）: --class 0, or --class 0 2 3
        agnostic_nms=False,  # 控制类别NMS：如果为False就是针对每一类预测框进行NMS、如果为True说明不需要区分类别
        augment=False,  # 数据增强
        visualize=False,  # 可视化检测结果，即需不需要把图片画上预测框输出
        update=False,  # 是否每一批更新权重（检测不需要、训练要设置）
        project=ROOT / 'runs/detect',  # 保存结果的文件夹
        name='exp',  # 保存第几轮的结果文件夹
        exist_ok=False,  # 查看创建的文件夹是否存在（False会抛异常）
        line_thickness=3,  # 边界框厚度 (像素)
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度值
        half=False,  # 使用 FP16 半精度推断
        dnn=False,  # 使用OpenCV DNN进行ONNX推理
        vid_stride=1,  # 视频帧速率步幅（每隔多少帧处理一次视频帧）
):
    # 获取待检测数据路径
    source = str(source)
    # save_img 取值为 True|False
    # nosave取假 并且 source值结尾不为txt后缀时 save_img取真
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # suffix是Path对象提供的方法用来判断文件后缀
    # is_file判断文件是否是图片或者是视频
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 检查所给地址是否是一个链接
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # isnumeric方法用来判断地址中是否是纯数字
    # 如果地址是纯数字字符 亦或是 以streams结尾 亦或是是url地址 webcam取值为真
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    # 判断是否是截图
    screenshot = source.lower().startswith('screen')
    # 根据地址下载相应文件
    if is_url and is_file:
        source = check_file(source)

    # 指定结果的保存路径
    # 创建文件夹（runs/detect/exp）,其中exp会自动增加
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    # 看需求是否保存labels文件，若需要保存到labels文件夹中
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    # 选取检测的设备（cuda\cpu）
    device = select_device(device)
    # 将配置文件输入模型中构建一个模型实例化对象
    # DetectMultiBackend负责管理和调度使用多个后端进行目标检测（PyTorch、TorchScript、ONNX、TensorRT等）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # attempt_load负责加载指定路径下的模型文件，并根据模型类型选择适当的加载方法。
    # 例如，如果模型是以 ".pt" 结尾的文件，attempt_load 函数将使用 PyTorch 的加载方法加载模型；如果模型是以 ".onnx" 结尾的文件，attempt_load 函数将使用 ONNX 的加载方法加载模型。
    stride, names, pt = model.stride, model.names, model.pt
    # check_img_size用来检查传入的--img-size，一般来说是32的倍数，如果不是会进行处理返回一个新的大小
    imgsz = check_img_size(imgsz, s=stride)
    get_gpu_memory()

    # 检测旋转物体不支持两阶段分类器（代表：FasterRCNN）
    classify = False
    assert not classify, "rotate does not support second-stage classifier"

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        # 加载视频流
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # 将数据集的长度赋值给batch_size可以确保模型在处理视频流时不会超出内存限制。
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    # 如果设备不是cpu的话，需要先预热一次
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    # seen变量用于跟踪模型已经看到过的图像数目。在训练过程中，每次看到一个新的图像，seen就会递增。这个变量在加载预训练权重模型时会用到，以确保继续训练时能够正确更新模型参数。
    # windows是一个空列表，用于存储检测结果中每个框的边界框坐标。在检测过程中，每个检测到的目标都会被表示为一个边界框，windows列表用于存储这些边界框的坐标信息，以便后续绘制检测结果或进行其他处理。
    # dt：dt是一个元组，包含三个元素。每个元素都是一个Profile()对象。Profile()是一个用于性能分析的类，在yolov5中被用于记录和打印模型的前向推理时间、FPS和内存占用情况。这三个元素分别对应输入图片的预处理时间、模型推理时间和后处理时间。
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # Apply rotate NMS
        with dt[2]:
            pred = rotate_non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            assert not save_crop, "rotate does not support save_crop"
            # imc = im0.copy() if save_crop else im0  # for save_crop
            # Annotator类用来实现向图片中画出预测框并且添加标签（这里使用Rotate的方法来画框子）
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = rotate_scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xywh, real, imagin, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywhn = (torch.tensor(xywh).view(1, 4) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywhn, real, imagin, conf) if save_conf else (cls, *xywhn, real, imagin)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        rotate_plot_one_box(torch.tensor([*xywh, real, imagin]).cpu().view(-1, 6).numpy(), im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                    # if save_crop:
                    #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # 每张图片的速度
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'rotate_best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images/cornleaf', help='file/dir/URL/glob/screen/0(webcam)')
    # parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser.add_argument('--visualize', action='store_true', help='visualize features')

    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')

    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    print('Notice: rotate_detect.py is designed for rotate cases')
    main(opt)
