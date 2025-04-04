# depth_anything_v2/depth_anything_interface.py を作成

import torch
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2  # 実際のインポートパスに合わせてください

class DepthAnythingInterface:
    def __init__(self, encoder='vitl', input_size=518, device='cuda'):
        self.device = device
        self.encoder = encoder
        self.input_size = input_size
        
        # モデルの設定と重みの読み込み
        self.model = DepthAnythingV2(**self.get_model_config(encoder))
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to(self.device).eval()
    
    def get_model_config(self, encoder):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        return model_configs[encoder]
    
    ###Noneから518に変更
    def infer_image(self, image, input_size=518):
        """
        入力画像から深度マップを生成します。
        Args:
            image (numpy.ndarray): 入力画像 (BGR形式)
            input_size (int): モデルの入力サイズ（デフォルトは初期化時の値）
        Returns:
            depth_map (numpy.ndarray): 深度マップ
        """
        if input_size is None:
            input_size = self.input_size
        
        # 画像の前処理
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size, input_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        

        # 深度推定
        with torch.no_grad():
            depth = self.model(img).squeeze().cpu().numpy()

        # 深度マップの正規化
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth = np.zeros_like(depth)
        depth = depth.astype(np.uint8)
        
        #with torch.no_grad():
        #    depth = self.model(img)
        #
        #depth = depth.squeeze().cpu().numpy()
        #depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
        
        return depth
