# Diagnosis first: `model_onefile_buttpair_with_fillet_attach.csv` の step30/40 切り分け

## 1. 確認したファイル
- `data/model_onefile_buttpair_with_fillet_attach.csv`
- `src/sov_app/engine/process_engine.py`
- `src/sov_app/engine/core_models.py`

## 2. step30 / step40 の loaded dict 要約

### step30 `30_mid_fitup_butt_A1_A2`
- `op`: `fitup_pair_chain`
- `base.instance`: `A1`
- `base` line: `points.B -> points.C`
- `guest.instance`: `A2`
- `guest` line: `points.A -> points.D`
- `model.butt_fitup`: `L_dist, d_nom, delta_y, eps_mA/B, eps_cA/B, g0, w0` を持つ
- `selector`: **なし**（`fitup_pair_chain` は `base/guest` 直接指定）

要点: step30 は `A1` 基準で `A2` の transform を更新し、さらに `state.butt_fitup_metrics` を記録する（この metric が後続の butt shrinkage の入力になる）。

### step40 `40_butt_welding_A1_A2`
- `op`: `welding_distortion`
- `target.selector`: `A2_only`
- `model.outplane_dz`: `Fixed(0)`
- `model.weak_bending_about_x`: `Fixed(0)`

要点: step40 は selector 解決で **A2 のみに適用**。処理内で `_apply_butt_transverse_shrinkage(inst_id="A2")` が呼ばれる。

## 3. step21 → 30 → 40 でどの instance / point が動く設計か

## step21 まで
- step20/21 (`fitup_attach_to_marking_line`) は `C1`,`C2` の transform と各点 offset を更新する。

## step30 (`fitup_pair_chain`) で起きること
- `base=A1, guest=A2` の rigid fitup なので、**直接動くのは A2 transform**。
- `C1/C2` は step30 の `base/guest` に含まれず、step30 内で参照も更新もされない。

## step40 (`welding_distortion`) で起きること
- selector が `A2_only` なので、`welding_distortion` 本体と butt shrinkage は **A2 にしか適用されない**。
- butt shrinkage の実装は A2 prototype の `A/B/C/D` 点に対する `point_offset` 追加であり、`C1/C2` 点へは直接作用しない。

## transform 追従の観点
- `AssemblyState.set_transform` は親子関係 (`frame.parent`) がある child だけを追従移動する。
- 本 CSV では `A1`,`A2` は `parent=world`、`C1/C2` も array 展開時 `parent=world`。
- つまり A2 が step30/40 で動いても、`C2` は親子追従で一緒には動かない。

## 4. `C1:points.c2` と `C2:points.c2` の距離分布がほぼ変わらない原因仮説

第一仮説（最有力）
- **観測点が butt shrinkage の作用経路外**。
- step30/40 は A2 側の fitup/shrinkage であり、`C1/C2` は world 直下の独立 instance のため、`C1:points.c2` と `C2:points.c2` の距離には伝播しない。

補足
- step40 で `outplane_dz=0` と `weak_bending_about_x=0` のため、A2 transform 側の見える変位も抑制される。
- それでも butt shrinkage は A2 点 offset として入るが、観測対象が C 側点だけなら統計変化は出にくい（またはゼロに近い）。

## 5. 最小差分で直すならどこを触るべきか

### A. 計測定義を変える（最小・安全）
- butt shrinkage の有無確認には、`A2` 側点（例: `A2:points.B` / `A2:points.C`）や A1-A2 継手近傍点を measurement に追加する。
- まずは「step40 で A2 点が縮む」ことを観測可能にする。

### B. もし「C 側距離にも効かせたい」要件なら
- 構造的に伝播させる必要がある。最小差分候補はどちらか:
  1) 親子フレーム設計を変更し、`C2` を `A2` 配下にする（または工程ごとに拘束を再適用）
  2) step40 後に `fitup_attach_to_marking_line` 相当の再拘束工程を追加して C 側へ変位を反映
- いずれも CSV/フロー責務として定義可能で、UI や step20/21 の既存 fillet 修正には触れずに対応できる。

## 6. 実行環境メモ
- このコンテナでは `numpy/pandas` が未導入で、`ProcessEngine` 実行による数値トレース script はそのままでは実行できなかった。
- そのため今回は「CSV 定義 + engine 実装 + parent 追従仕様」の静的切り分けに限定した。
