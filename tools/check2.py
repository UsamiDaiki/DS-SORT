import pandas as pd
import argparse

def main():
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description='Export depth data for each track ID.')
    parser.add_argument('--results_file', type=str, required=True, help='Path to the results file.')
    parser.add_argument('--output_file', type=str, default='depth_data_wide_format.csv', help='Output CSV file name.')
    args = parser.parse_args()

    # 結果ファイルのパスを指定
    results_file = args.results_file  # コマンドライン引数から取得

    # 列名を定義（結果ファイルのフォーマットに合わせて）
    columns = ['frame_id', 'tid', 'bbox_left', 'bbox_top', 'bbox_width', 'bbox_height',
               'score', 'object_class', 'viewpoint', 'attributes', 'depth_value']

    # データを読み込む
    df = pd.read_csv(results_file, header=None, names=columns)

    # データ型を変換
    df['frame_id'] = df['frame_id'].astype(int)
    df['tid'] = df['tid'].astype(int)
    df['depth_value'] = pd.to_numeric(df['depth_value'], errors='coerce')

    # 必要な列を抽出
    depth_data = df[['frame_id', 'tid', 'depth_value']]

    # データをフレーム番号とトラックIDでソート
    depth_data = depth_data.sort_values(['tid', 'frame_id'])

    # データをピボットテーブルに変換
    depth_pivot = depth_data.pivot(index='frame_id', columns='tid', values='depth_value')

    # 列名を文字列に変換
    depth_pivot.columns = [f'tid_{tid}' for tid in depth_pivot.columns]

    # フレーム番号を列として追加
    depth_pivot.reset_index(inplace=True)

    # CSVファイルに保存
    output_file = args.output_file  # コマンドライン引数から取得（デフォルト値あり）
    depth_pivot.to_csv(output_file, index=False)
    print(f'Depth data saved to {output_file}')

if __name__ == '__main__':
    main()

