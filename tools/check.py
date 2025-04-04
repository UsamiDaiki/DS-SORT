import pandas as pd
import argparse
import math

def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description='Compute mean and variance of depth values for each track ID every N frames.')
    parser.add_argument('--results_file', type=str, required=True, help='Path to the results file.')
    parser.add_argument('--segment_size', type=int, default=20, help='Number of frames in each segment.')
    args = parser.parse_args()

    # 結果ファイルのパスとセグメントサイズ
    results_file = args.results_file
    segment_size = args.segment_size

    # 列名を定義
    columns = ['frame_id', 'tid', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
               'score', 'object_class', 'viewpoint', 'attributes', 'depth_value']

    # データを読み込む
    df = pd.read_csv(results_file, header=None, names=columns)

    # データ型を適切に変換
    df['depth_value'] = df['depth_value'].astype(float)
    df['frame_id'] = df['frame_id'].astype(int)
    df['tid'] = df['tid'].astype(int)

    # セグメント番号を計算（フレーム番号をセグメントサイズで割り、1から始まるセグメント番号を付与）
    df['segment'] = ((df['frame_id'] - 1) // segment_size) + 1

    # セグメント番号とトラックIDでデータをグループ化
    grouped = df.groupby(['segment', 'tid'])

    # 結果を保存するリスト
    results = []

    # 各グループ（セグメント内のトラックID）について処理
    for (segment, tid), group in grouped:
        # 深度値を取得
        depth_values = group['depth_value']

        # 深度値が存在する場合のみ計算
        if len(depth_values) > 0:
            # 平均と分散を計算
            mean_depth = depth_values.mean()
            variance_depth = depth_values.var()
        else:
            mean_depth = None
            variance_depth = None

        # 結果を保存
        results.append({
            'segment': segment,
            'tid': tid,
            'mean_depth': mean_depth,
            'variance_depth': variance_depth
        })

    # 結果をデータフレームに変換
    results_df = pd.DataFrame(results)

    # 結果を表示
    print(results_df)

    # 結果をCSVファイルに保存
    output_file = 'depth_mean_variance_per_segment.csv'
    results_df.to_csv(output_file, index=False)
    print(f'Results saved to {output_file}')

if __name__ == '__main__':
    main()
