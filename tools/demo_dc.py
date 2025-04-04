import argparse

import sys
import os

# 現在のファイルのディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# プロジェクトのルートディレクトリを取得（tools ディレクトリの一つ上）
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# project_root を sys.path に追加
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os.path as osp
import time
import cv2
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking, plot_tracking_detection
from trackers.ocsort_tracker.ocsort import OCSort
from trackers.hybrid_sort_tracker.hybrid_sort import Hybrid_Sort
from trackers.hybrid_sort_tracker.hybrid_sort_reid import Hybrid_Sort_ReID
from trackers.tracking_utils.timer import Timer
from fast_reid.fast_reid_interfece import FastReIDInterface

from trackers.dc_tracker.dc_sort import Dc_Sort

from depth_anything_v2.dinov2 import DINOv2  # DINOv2 モデルをインポート


from depth_anything_v2.dpt import DepthAnythingV2  # Depth-Anything V2 のクラスをインポート

#from depth_anything_v2.depth_anything_interface import DepthAnythingV2  # Depth-Anything V2 のクラスをインポート

import copy

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

from utils.args import make_parser, args_merge_params_form_exp

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        with_reid=False,
        with_depth=False,  # 深度推定のフラグ
        fast_reid_config=None,
        fast_reid_weights=None,
        depth_encoder='vitl',  # Depth-Anything V2 のエンコーダ
        depth_input_size=518,  # Depth-Anything V2 の入力サイズ
        depth_weights=None  # **ここに引数を追加**
        #depth_weights='checkpoints/depth_anything_v2_vitl.pth'  # Depth-Anything V2 の重みファイル
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.with_reid = with_reid
        self.with_depth = with_depth  # 追加
        self.depth_input_size = depth_input_size  # 追加
        self.depth_encoder = depth_encoder        # **追加**
        self.depth_weights = depth_weights  # **追加**
    
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        if self.with_reid:
            self.fast_reid_config = fast_reid_config
            self.fast_reid_weights = fast_reid_weights
            self.encoder = FastReIDInterface(self.fast_reid_config, self.fast_reid_weights, 'cuda')

        #if self.with_depth:
        #   self.depth_estimator = DepthAnythingInterface(encoder=depth_encoder, input_size=depth_input_size, device=self.device)

        #weights_path=depth_weightsを追加、本当に必要かは不明
        if self.with_depth:
            #self.depth_estimator = DepthAnythingV2(
            self.depth_estimator = DepthAnythingV2(
                encoder=self.depth_encoder,
                #input_size=depth_input_size,
                #weights_path=depth_weights,
                #device=self.device
            )
            ### モデルの重みをロード
            if depth_weights is None:
                depth_weights = 'checkpoints/depth_anything_v2_{}.pth'.format(self.depth_encoder)
            self.depth_estimator.load_state_dict(torch.load(depth_weights, map_location=self.device))
            self.depth_estimator.to(self.device)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # preproc 関数の返り値を3つに変更
        img, ratio, _ = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

            depth = None
            if self.with_depth:
                #depth = self.depth_estimator.infer_image(img_info["raw_img"], self.depth_estimator.input_size)
                depth = self.depth_estimator.infer_image(img_info["raw_img"],  input_size=self.depth_input_size)  # メソッド名を修正
                img_info["depth"] = depth

            if self.with_reid:
                if outputs[0] is not None and outputs[0].shape[0] > 0:
                    bbox_xyxy = copy.deepcopy(outputs[0][:, :4])
                    scale = min(self.test_size[0] / float(img_info["height"]), self.test_size[1] / float(img_info["width"]))
                    bbox_xyxy /= scale
                    id_feature = self.encoder.inference(img_info["raw_img"], bbox_xyxy.cpu().detach().numpy())
                else:
                    id_feature = np.array([])  # 空の配列を設定

        if self.with_reid and self.with_depth:
            return outputs, img_info, id_feature, depth
        elif self.with_reid:
            return outputs, img_info, id_feature
        elif self.with_depth:
            return outputs, img_info, depth
        else:
            return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    # Tracker の初期化
    if not args.hybrid_sort_with_reid and not args.with_depth:
        tracker = Hybrid_Sort(args, det_thresh=args.track_thresh,
                                iou_threshold=args.iou_thresh,
                                asso_func=args.asso,
                                delta_t=args.deltat,
                                inertia=args.inertia,
                                use_byte=args.use_byte)
    elif args.hybrid_sort_with_reid and not args.with_depth:
        tracker = Ex_Sort_ReID(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia)
    elif not args.hybrid_sort_with_reid and args.with_depth:
        tracker = Dc_Sort(args, det_thresh=args.track_thresh,
                                iou_threshold=args.iou_thresh,
                                asso_func=args.asso,
                                delta_t=args.deltat,
                                inertia=args.inertia,
                                use_byte=args.use_byte) 
    else:
        tracker = Ex_Sort_ReID(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia)  # ReID と Depth 対応

    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        if args.with_fastreid and args.with_depth:
            outputs, img_info, id_feature, depth = predictor.inference(img_path, timer)
        elif args.with_fastreid:
            outputs, img_info, id_feature = predictor.inference(img_path, timer)
        elif args.with_depth:
            outputs, img_info, depth = predictor.inference(img_path, timer)
        else:
            outputs, img_info = predictor.inference(img_path, timer)

        if outputs[0] is not None:
            if args.with_fastreid and args.with_depth:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, id_feature=id_feature, depth=depth)
            elif args.with_fastreid:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, id_feature=id_feature)
            elif args.with_depth:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, depth=depth)
            else:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

            online_tlwhs = []
            online_ids = []
            depths = []  # 各トラックの深度を格納するリスト
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                depth_out = t[5]  ## 深度情報を取得
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

                    # BBの中心座標を計算
                    center_x = int(tlwh[0] + tlwh[2] / 2)
                    center_y = int(tlwh[1] + tlwh[3] / 2)
                    if args.with_depth and depth is not None:
                        if 0 <= center_y < depth.shape[0] and 0 <= center_x < depth.shape[1]:
                            depth_value = depth[center_y, center_x]
                        else:
                            depth_value = 0  # 範囲外の場合のデフォルト値
                    else:
                        depth_value = 0  # 深度情報がない場合のデフォルト値

                    depths.append(depth_value)

                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1,{depth_value},{depth_out}\n"
                    )

            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        if args.save_result:
            if not args.demo_dancetrack:
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                save_folder = osp.join(vis_folder, timestamp)
            else:
                timestamp = args.path[-19:]
                save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

            # 深度情報の保存
            if args.save_depth and 'depth' in img_info:
                depth_save_path = osp.join(save_folder, "depth")
                os.makedirs(depth_save_path, exist_ok=True)
                depth_image = img_info['depth']
                depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_uint8 = depth_image_normalized.astype('uint8')
                cv2.imwrite(osp.join(depth_save_path, f"frame_{frame_id}.png"), depth_image_uint8)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")



def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo_type == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo_type == "video":
        save_path = args.out_path
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    # Tracker の初期化
    if not args.hybrid_sort_with_reid and not args.with_depth:
        tracker = Ex_Sort(args, det_thresh=args.track_thresh,
                                iou_threshold=args.iou_thresh,
                                asso_func=args.asso,
                                delta_t=args.deltat,
                                inertia=args.inertia,
                                use_byte=args.use_byte)
    elif args.hybrid_sort_with_reid and not args.with_depth:
        tracker = Ex_Sort_ReID(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia)
    elif not args.hybrid_sort_with_reid and args.with_depth:
        tracker = Ex_Sort_19(args, det_thresh=args.track_thresh,
                                iou_threshold=args.iou_thresh,
                                asso_func=args.asso,
                                delta_t=args.deltat,
                                inertia=args.inertia,
                                use_byte=args.use_byte)  # 深度閾値を設定
    else:
        tracker = Ex_Sort_ReID(args, det_thresh=args.track_thresh,
                                    iou_threshold=args.iou_thresh,
                                    asso_func=args.asso,
                                    delta_t=args.deltat,
                                    inertia=args.inertia
                                    )  # ReID と Depth 対応

    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            if args.with_fastreid and args.with_depth:
                outputs, img_info, id_feature, depth = predictor.inference(frame, timer)
            elif args.with_fastreid:
                outputs, img_info, id_feature = predictor.inference(frame, timer)
            elif args.with_depth:
                outputs, img_info, depth = predictor.inference(frame, timer)
            else:
                outputs, img_info = predictor.inference(frame, timer)

            if outputs[0] is not None:
                if args.with_fastreid and args.with_depth:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, id_feature=id_feature, depth=depth)
                elif args.with_fastreid:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, id_feature=id_feature)
                elif args.with_depth:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, depth=depth)
                else:
                    online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

                online_tlwhs = []
                online_ids = []
                depths = []
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = t[4]
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)

                        # BBの中心座標を計算
                        center_x = int(tlwh[0] + tlwh[2] / 2)
                        center_y = int(tlwh[1] + tlwh[3] / 2)
                        if args.with_depth and depth is not None:
                            if 0 <= center_y < depth.shape[0] and 0 <= center_x < depth.shape[1]:
                                depth_value = depth[center_y, center_x]
                            else:
                                depth_value = 0  # 範囲外の場合のデフォルト値
                        else:
                            depth_value = 0  # 深度情報がない場合のデフォルト値

                        depths.append(depth_value)

                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1,{depth_value}\n"
                        )

                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if args.save_result:
                vid_writer.write(online_im)

                # 深度情報の保存
                if args.save_depth and 'depth' in img_info:
                    depth_save_path = osp.join(save_folder, "depth")
                    os.makedirs(depth_save_path, exist_ok=True)
                    depth_image = img_info['depth']
                    depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                    depth_image_uint8 = depth_image_normalized.astype('uint8')
                    cv2.imwrite(osp.join(depth_save_path, f"frame_{frame_id}.png"), depth_image_uint8)

            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")



def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.expn)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, str(args.hybrid_sort_with_reid), "track_vis")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    #predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16,
    #                      with_reid=args.with_fastreid, fast_reid_config=args.fast_reid_config, fast_reid_weights=args.fast_reid_weights)    
    
    predictor = Predictor(
        model, 
        exp, 
        trt_file, 
        decoder, 
        args.device, 
        args.fp16,
        with_reid=args.with_fastreid, 
        with_depth=args.with_depth,  # 追加
        fast_reid_config=args.fast_reid_config, 
        fast_reid_weights=args.fast_reid_weights,
        depth_encoder=args.depth_encoder,  # 追加
        depth_input_size=args.depth_input_size,  # 追加
        depth_weights=args.depth_weights  # 追加
    )    
    
    current_time = time.localtime()
    if args.demo_type == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo_type == "video" or args.demo_type == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    args_merge_params_form_exp(args, exp)

    main(exp, args)
