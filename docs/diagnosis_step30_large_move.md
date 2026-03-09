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
- `w0=146`, `d_nom=73`, `g0=4`, `L_dist=LN_fit_L(mean=4,std=1.5)`, `eps_m*=N(0,0.05)`, `eps_c*=N(0,0.1)`。

## 3) 実装上の意味づけ（w/g/g_real の使われ方）
`fitup_pair_chain` の butt 分岐では以下の計算になっている。

- `w = w0 + L + (eps_mB - eps_mA)`
- `dA = d_nom + eps_cA`, `dB = d_nom + eps_cB`
- `g_real = w - (dA + dB)`

ここで重要なのは、A2 を剛体移動させるターゲットが marking-line basis で
`g_real` を通じて構築されている点:

- `q0_target = p0 + (dA + dB - w) * v = p0 - g_real * v`
- `t = q0_target - q0`
- `A2.origin <- A2.origin + t`

つまり `L_dist` は edge target に直接足し込まれるのではなく、
`w -> g_real -> q_target` の経路で効く。

## 4) なぜ ~150 mm 級の移動になるか（幾何的内訳）
この CSV では、A1 の butt 基準線は `B(1998,0,0) -> C(1998,1000,0)`。

- `u = unit(C-B) = +Y`
- `n = +Z`
- `v = cross(n,u) = -X`

よって `q0_target = p0 - g_real*v` は実質 `x = 1998 + g_real` を指す。
`w0=146, E[L]=4, d_nom=73` なら平均的に `E[g_real]=4` なので、平均 target は `x≈2002` 近傍になる。

step21 時点の `q0=A2:points.A` も概ね `x≈2002` 近傍なので、
平均的には step30 の剛体移動は小さくなる。

## 5) g_real と step30変位の関係
`g_real` はこの実装では step30 の剛体再配置量（`q_target`）を決める中心量であり、
同時に step40 の横収縮 (`0.18*g_real`) 側にも使われる。

このため `w0` と `L_dist` は平均値合わせではなく役割分担で設定するのが重要:
- `w = w0 + L + (Δm,B-Δm,A)`
- `w0=146`, `E[L]=4` なら `E[w]=150`
- `d_nom=73` なら `E[dA+dB]=146`、従って `E[g_real]=4`

## 6) parent-child 修正の評価
`AssemblyState.set_transform` は親 transform 変更時に child を追従させる設計。

したがって今回の大変位は、
- parent-child 修正が新たな大移動を作ったのではなく、
- step30 の target 構成（`g_real` 経由）に従う A2 側挙動が C2 に可視化された

と解釈するのが妥当。

## 7) 補足
現在の実装はすでに marking-line basis（`q_target = p + (dA+dB-w)*v`）に揃っているため、
今後の調整は「平均値合わせ」ではなく `w0` と `L_dist` の定義分担を保つことを優先する。
