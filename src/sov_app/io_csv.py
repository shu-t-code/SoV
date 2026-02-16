from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def csv_to_nested_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """
    CSVの doc, path, value_json 形式から階層的なJSON辞書を再構築

    例:
      doc='geometry', path='/units', value_json='"mm"'
      → result['units'] = 'mm'
    """
    result = {}

    for _, row in df.iterrows():
        path = row["path"]
        value_json = row["value_json"]

        # JSONパース
        try:
            value = json.loads(value_json)
        except Exception:
            value = value_json  # パース失敗時は文字列のまま

        # パスを分解（先頭の / を除去）
        path = path.lstrip("/")
        keys = path.split("/")

        # 階層を辿って値を設定
        current = result
        for i, key in enumerate(keys[:-1]):
            # 配列インデックスか辞書キーか判定
            if key.isdigit():
                key = int(key)
                # 親がリストでない場合は初期化
                if not isinstance(current, list):
                    parent_key = keys[i - 1] if i > 0 else None
                    if parent_key and parent_key in current:
                        current[parent_key] = []
                        current = current[parent_key]
                # リストを必要なサイズに拡張
                while len(current) <= key:
                    current.append({})
                current = current[key]
            else:
                if key not in current:
                    # 次のキーが数字ならリスト、そうでなければ辞書
                    next_key = keys[i + 1] if i + 1 < len(keys) else None
                    if next_key and next_key.isdigit():
                        current[key] = []
                    else:
                        current[key] = {}
                current = current[key]

        # 最後のキーに値を設定
        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
            if not isinstance(current, list):
                current = []
            while len(current) <= final_key:
                current.append(None)
            current[final_key] = value
        else:
            current[final_key] = value

    return result


def load_data_from_csv(csv_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    model_onefile.csv から geometry と flow の両方のデータを読み込む
    （区切り: カンマ or タブ、ヘッダ有無も吸収）
    """
    # 1) 先頭行で区切り文字をざっくり判定
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        first_line = f.readline()

    # タブが多いならTSV扱い、そうでなければCSV扱い
    sep = "\t" if (first_line.count("\t") > first_line.count(",")) else ","

    # 2) まずは「ヘッダあり」で読む
    df = pd.read_csv(csv_path, sep=sep, dtype=str, encoding="utf-8-sig", engine="python")

    # 3) 列名の正規化（前後空白除去）
    df.columns = [str(c).strip() for c in df.columns]

    # 4) doc列が無いなら「ヘッダ無し3列」扱いで読み直す
    if "doc" not in df.columns:
        df = pd.read_csv(
            csv_path,
            sep=sep,
            header=None,
            names=["doc", "path", "value_json"],
            dtype=str,
            encoding="utf-8-sig",
            engine="python",
        )

    # 5) 念のため必要列チェック
    required = {"doc", "path", "value_json"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV format error: required columns {required}, got {list(df.columns)}")

    # geometry / flow を抽出
    geom_df = df[df["doc"] == "geometry"].copy()
    geom_dict = csv_to_nested_dict(geom_df) if not geom_df.empty else {}

    flow_df = df[df["doc"] == "flow"].copy()
    flow_dict = csv_to_nested_dict(flow_df) if not flow_df.empty else {}

    return geom_dict, flow_dict


def nested_dict_to_csv_rows(data: Dict[str, Any], doc_name: str, parent_path: str = "") -> List[Dict[str, str]]:
    """
    階層的な辞書をCSV行形式に変換

    Args:
        data: 変換する辞書
        doc_name: ドキュメント名 ('geometry' or 'flow')
        parent_path: 親パス（再帰用）

    Returns:
        CSV行のリスト [{doc, path, value_json}, ...]
    """
    rows = []

    def flatten(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}/{key}"
                flatten(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}/{i}"
                flatten(item, new_path)
        else:
            rows.append({
                "doc": doc_name,
                "path": path,
                "value_json": json.dumps(obj, ensure_ascii=False),
            })

    flatten(data, parent_path)
    return rows
