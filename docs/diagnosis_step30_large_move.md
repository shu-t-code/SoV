# Diagnosis: step30 butt fitup で剛体移動が大きく見える理由

## 1) 確認対象
- `data/model_onefile_buttpair_with_fillet_attach.csv`
- `src/sov_app/engine/process_engine.py`
- `src/sov_app/engine/core_models.py`

## 2) step30 loaded dict の要約
- `id`: `30_mid_fitup_butt_A1_A2`
- `op`: `fitup_pair_chain`
- `base.instance`: `A1`, `base.p0/p1`: `points.B -> points.C`
- `guest.instance`: `A2`, `guest.q0/q1`: `points.A -> points.D`
- `selector`: なし（`base/guest` 直指定）
- `model.butt_fitup`: `d_nom, g0, w0, L_dist, eps_mA/B, eps_cA/B, delta_y` を使用

CSV 上の設定値:
- `w0=150`, `d_nom=73`, `g0=4`, `L_dist=LN_fit_L(mean=4,std=1.5)`, `eps_m*=N(0,0.05)`, `eps_c*=N(0,0.1)`。

## 3) 実装上の意味づけ（w/g/g_real の使われ方）
`fitup_pair_chain` の butt 分岐では以下の計算になっている。

- `w = w0 + L + (eps_mB - eps_mA)`
- `dA = d_nom + eps_cA`, `dB = d_nom + eps_cB`
- `g_real = w - (dA + dB)`

ここで重要なのは、A2 を剛体移動させるターゲットが **`w` を直接使って** 構築されている点:

- `q0_target = p0 + w * v`
- `t = q0_target - q0`
- `A2.origin <- A2.origin + t`

つまり step30 の大きな剛体移動は `g_real` ではなく `w` スケールで決まる。

## 4) なぜ ~150 mm 級の移動になるか（幾何的内訳）
この CSV では、A1 の butt 基準線は `B(1998,0,0) -> C(1998,1000,0)`。

- `u = unit(C-B) = +Y`
- `n = +Z`
- `v = cross(n,u) = -X`

よって `q0_target = p0 + w*v` は実質 `x = 1998 - w` を指す。
`w ≈ 152.856` なら `q0_target.x ≈ 1845.144`。

step21 時点の `q0=A2:points.A` は概ね `x≈2002` 近傍なので、
`A2` は `Δx≈1845.144-2002≈-156.856 mm` 移動する。

この値は観測されている `C2` の `step21->30` 変位（約 `-156.655 mm`）と整合する。

## 5) g_real と step30変位の関係
`g_real≈6.8 mm` はこの実装では step30 の剛体再配置量そのものを決めていない。
- step30 の平行移動は `w` で決定
- `g_real` は主にメトリクス保持と step40 の横収縮 (`0.18*g_real`) 側で使われる

そのため、
- step21→30 で ~158 mm 級の距離変化
- step30→40 で ~1.2 mm 級（= 0.18*g_real）
というスケール差が生じるのは実装と一致する。

## 6) parent-child 修正の評価
`AssemblyState.set_transform` は親 transform 変更時に child を追従させる設計。

したがって今回の大変位は、
- parent-child 修正が新たな大移動を作ったのではなく、
- もともと step30 で A2 に入っていた `w` スケール移動が C2 に可視化された

と解釈するのが妥当。

## 7) 最小差分で次に直すなら（今回は未実装）
step30 の剛体ターゲット式（`q0_target = p0 + w*v`）の意味づけを見直すのが最小焦点。
候補:
1. `w` を絶対オフセットとして使うのをやめ、`g_real` あるいは `w-w_ref` の差分量で配置する
2. `w` をメトリクス専用にして、rigid alignment は別の基準（既存基準線一致 + 小さな gap 項）に分離する

いずれも `fitup_pair_chain` butt 分岐のターゲット構築部のみの局所修正で可能。
