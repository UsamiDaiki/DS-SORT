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
import random
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


from trackers.ds_tracker.ds_sort import Ds_Sort


from depth_anything_v2.dinov2 import DINOv2  # DINOv2 モデルをインポート


from depth_anything_v2.dpt import DepthAnythingV2  # Depth-Anything V2 のクラスをインポート

#from sam2.build_sam import build_sam2
#from sam2.sam2_image_predictor import SAM2ImagePredictor #sam2をインポート

from segment_anything import sam_model_registry, SamPredictor
import numpy as np


import copy

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

import itertools

# 一意で目立つカラーパレットを定義
PREDEFINED_COLORS = [
    (255, 0, 0),      # 赤
    (0, 255, 0),      # 緑
    (0, 0, 255),      # 青
    (255, 255, 0),    # 黄
    (255, 0, 255),    # マゼンタ
    (0, 255, 255),    # シアン
    (255, 165, 0),    # オレンジ
    (128, 0, 128),    # 紫
    (0, 128, 0),      # ダークグリーン
    (128, 128, 0),    # オリーブ
    (0, 128, 128),    # ティール
    (128, 0, 0),      # マルーン
    (0, 0, 128),      # ネイビーブルー
    (128, 128, 128),  # グレー
]

# 色の繰り返しを防ぐためにイテレータを作成
COLOR_ITER = itertools.cycle(PREDEFINED_COLORS)


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
        depth_weights=None,  # **ここに引数を追加**
        with_sam=False,  # SAMのフラグを追加
        sam_model_type="vit_h",  # 使用するSAMモデルの種類
        sam_checkpoint_path=None,  # SAMモデルの重みファイルパス

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
        self.with_sam = args.with_sam  # SAMフラグを追加
        
        
        #if self.with_depth:
        #    self.sam2_predictor = self.init_sam2_model(
        #        model_type=sam2_model_type,
        #        checkpoint_path=sam2_checkpoint_path,
        #        device=self.device
        #    )
    
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

        # SAMの設定
        if self.with_sam:
            self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
            self.sam_model.to(self.device)
            self.sam_predictor = SamPredictor(self.sam_model)


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

        # preproc 関数の返り値を3つに変更（例として）
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

            #depth_map = None
            #if self.with_depth:
            #    depth_map = self.depth_estimator.infer_image(img_info["raw_img"], input_size=self.depth_input_size)
            #    img_info["depth"] = depth_map

            depth_map = None
            if self.with_depth:
                # 1) 深度マップを推論 (float32, 0～1 or 0～max_range のスケール)
                depth_map = self.depth_estimator.infer_image(img_info["raw_img"], input_size=self.depth_input_size)

                # (2) 0～255 に正規化 (NORM_MINMAX)
                ##depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                #depth_normalized = depth_normalized.astype('uint8')  # 8bit にキャスト

                # 今回は 0～255 の深度をこの後使う
                #depth_map = depth_normalized
                img_info["depth"] = depth_map

            # セグメンテーション
            masks = []
            """
            if self.with_sam and outputs[0] is not None and outputs[0].shape[0] > 0:
                # 検出されたバウンディングボックスを取得
                bboxes = outputs[0][:, :4] / ratio
                image_rgb = cv2.cvtColor(img_info["raw_img"], cv2.COLOR_BGR2RGB)
                self.sam_predictor.set_image(image_rgb)

                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox.int().tolist()
                    input_box = np.array([x1, y1, x2, y2]).reshape(1, -1)
                    mask, _, _ = self.sam_predictor.predict(box=input_box, multimask_output=False)
                    masks.append(mask[0])
            """
            
            if self.with_sam and outputs[0] is not None and outputs[0].shape[0] > 0:
                # first round での閾値（例：args もしくは exp で設定済みの値）
                first_round_thresh = 0.75  # 例として 0.6 を利用
                # スコアが閾値以上の検出のみを対象とする
                    # outputs[0] の列数によってスコアの計算方法を変える
                if outputs[0].shape[1] == 5:
                    scores = outputs[0][:, 4] 
                else:
                    scores = outputs[0][:, 4] * outputs[0][:, 5]

                valid_inds = scores >= first_round_thresh
                if valid_inds.sum() > 0:
                    bboxes = outputs[0][valid_inds, :4] / ratio
                    image_rgb = cv2.cvtColor(img_info["raw_img"], cv2.COLOR_BGR2RGB)
                    self.sam_predictor.set_image(image_rgb)

                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox.int().tolist()
                        input_box = np.array([x1, y1, x2, y2]).reshape(1, -1)
                        mask, _, _ = self.sam_predictor.predict(box=input_box, multimask_output=False)
                        masks.append(mask[0])

            
            
            if self.with_depth and depth_map is not None:
                # SAMの処理結果は、valid_indsがTrueの検出に対するもの
                # ここで、元の検出結果の行数と同じ長さの depth_values を作成します。
                total_dets = outputs[0].shape[0]
                depth_values_full = []  # 全検出に対する深度値リスト
                valid_idx = 0  # masks_valid のインデックス用
                default_depth = -1  # 例：閾値を下回った検出には -1 を割り当てる
                
                for i in range(total_dets):
                    # valid_inds は tensor か NumPy配列かに応じて適宜変換
                    # ここでは valid_inds[i] がTrueならSAM処理済みと仮定
                    if valid_inds[i]:
                        # validな検出については、SAM処理済みの深度値を使う
                        depth_val = extract_depth_from_mask(depth_map, masks[valid_idx], method="median")
                        depth_values_full.append(depth_val)
                        valid_idx += 1
                    else:
                        # 閾値未満の検出にはデフォルト値を使う
                        depth_values_full.append(default_depth)
                
                # ここで depth_values_full の長さは outputs[0] の行数と同じになる
                if isinstance(outputs[0], torch.Tensor):
                    depth_tensor = torch.tensor(depth_values_full, dtype=outputs[0].dtype, device=outputs[0].device).unsqueeze(1)
                    outputs[0] = torch.cat([outputs[0], depth_tensor], dim=1)
                else:
                    depth_array = np.array(depth_values_full).reshape(-1, 1)
                    outputs[0] = np.concatenate([outputs[0], depth_array], axis=1)
            

            """
            # SAM のマスクと深度マップがある場合、各検出に対して深度値を算出
            if self.with_depth and depth_map is not None:
                depth_values = []
                for mask in masks:
                    # 各マスク領域の深度中央値を算出
                    depth_val = extract_depth_from_mask(depth_map, mask, method="median")
                    depth_values.append(depth_val)

                ##for i, bbox in enumerate(bboxes):
                ##    depth_val = extract_depth(depth_map, bbox)
                ##   depth_values.append(depth_val)

                # outputs[0] に深度情報（1列）を連結する  
                # outputs[0] の型が torch.Tensor か numpy.ndarray かに応じて処理を分ける
                if isinstance(outputs[0], torch.Tensor):
                    depth_tensor = torch.tensor(depth_values, dtype=outputs[0].dtype, device=outputs[0].device).unsqueeze(1)
                    outputs[0] = torch.cat([outputs[0], depth_tensor], dim=1)
                else:
                    depth_array = np.array(depth_values).reshape(-1, 1)
                    outputs[0] = np.concatenate([outputs[0], depth_array], axis=1)
                #print("Debug Depth1")
                #print(depth_values)
            """


            if self.with_reid:
                if outputs[0] is not None and outputs[0].shape[0] > 0:
                    bbox_xyxy = copy.deepcopy(outputs[0][:, :4])
                    scale = min(self.test_size[0] / float(img_info["height"]), self.test_size[1] / float(img_info["width"]))
                    bbox_xyxy /= scale
                    id_feature = self.encoder.inference(img_info["raw_img"], bbox_xyxy.cpu().detach().numpy())
                else:
                    id_feature = np.array([])  # 空の配列を設定
        
        
        return outputs, img_info, depth_map, masks


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    tracker = Ds_Sort(args, det_thresh=args.track_thresh,
                            iou_threshold=args.iou_thresh,
                            asso_func=args.asso,
                            delta_t=args.deltat,
                            inertia=args.inertia,
                            use_byte=args.use_byte) 

    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        # --- 推論の呼び出し ---
        outputs, img_info,  depth_map, masks = predictor.inference(img_path, timer)

        if outputs[0] is not None:
            online_targets=tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n"
                    )

            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # SAMによるマスクをオーバーレイ (args.with_sam==True かつ masks がある場合)
        if args.with_sam and outputs[0] is not None:
            # 先ほど定義した関数を使ってオーバーレイ
            online_im = overlay_masks_on_image(online_im, masks, alpha=0.5)

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = args.out_path if args.demo_type == "video" else osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # tracker の初期化（使用する tracker に合わせて分岐）
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
                             use_byte=args.use_byte)
    else:
        tracker = Ex_Sort_ReID(args, det_thresh=args.track_thresh,
                               iou_threshold=args.iou_thresh,
                               asso_func=args.asso,
                               delta_t=args.deltat,
                               inertia=args.inertia)

    timer = Timer()
    frame_id = 0
    results = []
    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        # 推論呼び出し（inference 内で outputs に深度値が含まれる）
        outputs, img_info, depth_map, masks = predictor.inference(frame, timer)

        if outputs[0] is not None:
            # tracker.update に outputs[0] を渡す
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                tid = t[4]
                # t[-1] に深度値が格納されている前提
                if tlwh[2] * tlwh[3] > args.min_box_area:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1,{t[-1]:.2f}\n"
                    )

            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # SAM によるマスクオーバーレイ
        if args.with_sam and outputs[0] is not None:
            online_im = overlay_masks_on_image(online_im, masks, alpha=0.5)

        if args.save_result:
            vid_writer.write(online_im)
            if args.save_depth and 'depth' in img_info:
                depth_save_path = osp.join(save_folder, "depth")
                os.makedirs(depth_save_path, exist_ok=True)
                depth_image = img_info['depth']
                depth_image_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_image_uint8 = depth_image_normalized.astype('uint8')
                cv2.imwrite(osp.join(depth_save_path, f"frame_{frame_id}.png"), depth_image_uint8)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def extract_depth_from_mask(depth_map, mask, method="median"):
    """
    マスク領域の深度値から代表値を取得する関数
    - depth_map: (H, W) の深度マップ (float32など)
    - mask: (H, W) の0/1またはboolマスク
    - method: "mean" または "median"
    """
    masked_depth = depth_map[mask > 0]
    if len(masked_depth) == 0:
        # マスク領域なし
        return 0.0

    if method == "mean":
        return float(masked_depth.mean())
    elif method == "median":
        return float(np.median(masked_depth))
    else:
        return float(masked_depth.mean())


def extract_depth(depth_map, bbox):
    """
    bbox の中心ピクセルから深度値を1点抽出して返す。
    - depth_map:  (H, W) の深度マップ (例: 0～255 の範囲, float32 か uint8 など)
    - bbox:       [x1, y1, x2, y2, (任意の他列...)] の配列
    
    Returns:
        depth_val: float または int (マップの型次第)
    """
    # bbox が [x1, y1, x2, y2] を持っている前提:
    x1, y1, x2, y2 = bbox[:4]
    
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    # 深度マップの範囲外チェック
    H, W = depth_map.shape[:2]  # 2次元の場合 shape=(H,W)
    if 0 <= center_x < W and 0 <= center_y < H:
        depth_val = depth_map[center_y, center_x]
    else:
        # 範囲外なら 0 や -1 など特別な値を返す
        depth_val = 0

    return float(depth_val)


def overlay_masks_on_image(image, masks, alpha=0.5):
    """
    SAMで推論した複数のマスクを画像にオーバーレイする関数
    - image: 元の画像 (H, W, 3) - BGR形式
    - masks: SAMで得られるマスクのリスト [(H, W), (H, W), ...] - 0/1またはbool
    - alpha: マスクの透明度(0〜1)
    """
    result_image = image.copy()
    
    # カラーパレットのイテレータをリセット（毎回同じ順序で色を適用）
    color_iter = itertools.cycle(PREDEFINED_COLORS)
    
    for mask in masks:
        color = next(color_iter)
        colored_mask = np.zeros_like(result_image, dtype=np.uint8)
        
        # 各チャンネルに色を適用
        for c in range(3):
            colored_mask[:, :, c] = mask * color[c]
        
        # オーバーレイ（αブレンド）
        result_image = cv2.addWeighted(result_image, 1, colored_mask, alpha, 0)
    
    return result_image


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
        depth_weights=args.depth_weights,  # 追加
        with_sam=args.with_sam,  # SAMのフラグを追加
        sam_model_type=args.sam_model_type,  # 使用するSAMモデルの種類
        sam_checkpoint_path=args.sam_checkpoint_path  # SAMモデルの重みファイルパス
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
