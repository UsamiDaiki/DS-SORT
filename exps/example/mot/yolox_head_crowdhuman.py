# exps/example/custom/yolox_head_crowdhuman.py

import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # ---------- model config ---------- 
        # YOLOX-M相当 (depth=0.67, width=0.75) 参考
        # https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolox.py
        # 公式設定は "yolox_m" は depth=0.67, width=0.75
        self.depth = 0.67
        self.width = 0.75

        # クラス数: 頭部のみ1クラス
        self.num_classes = 1

        # クラス名の設定
        self.class_names = ['head']  # クラス名を 'head' のみに設定

        # 画像サイズ設定
        #self.input_size = (640, 640)
        #self.random_size = (14, 26)  # マルチスケールでの最小/最大size
        #self.test_size = (640, 640)

        # ---------- training config ---------- 
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 16.0  # デフォルト相当
        self.max_epoch = 50  # 例：50エポック（適宜変更）
        self.eval_interval = 1
        self.print_interval = 50
        self.save_history_ckpt = False

        self.data_num_workers = 4
        self.batch_size = 16   # GPUのメモリに合わせて調整

        # ---------- transform config ---------- 
        # ここは頭部検出なので調整の余地あり（mosaicが強すぎると顔が崩れることも）
        self.mosaic_prob = 1.0
        self.mixup_prob = 0.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        # ---------- data dir and annotation ---------- 
        self.data_dir = "datasets/crowdhuman"
        self.train_ann = "train_head.json"
        self.val_ann = "val_head.json"
        self.test_ann = "val_head.json"  # valをそのままtestに使うなら

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import COCODataset, TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler

        dataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="CrowdHuman_train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=cache_img,
        )

        self.dataset = dataset

        sampler = InfiniteSampler(len(dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug
        )

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "batch_sampler": batch_sampler,
        }
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        return dataloader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform
        from torch.utils.data import DataLoader, SequentialSampler

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="CrowdHuman_val",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        sampler = SequentialSampler(valdataset)
        dataloader = DataLoader(
            valdataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.data_num_workers,
            pin_memory=True,
        )
        return dataloader

