from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .core_models import AssemblyState, FlowModel, GeometryModel, Validator, get_world_point
from .env import USE_OPEN3D
from .io_csv import load_data_from_csv, nested_dict_to_csv_rows
from .monte_carlo import MonteCarloSimulator
from .process_engine import ProcessEngine
from .util_logging import FileChangeHandler, HAS_WATCHDOG as UTIL_HAS_WATCHDOG, Observer
from .visualize import DistanceHistogramWidget, InteractivePointSelector, MatplotlibVisualizer, Open3DVisualizer

try:  # optional
    import open3d as o3d
except Exception:  # pragma: no cover - optional dependency
    o3d = None

HAS_WATCHDOG = UTIL_HAS_WATCHDOG and Observer is not None and FileChangeHandler is not None

logger = logging.getLogger("sov_app")


class MainWindow(QMainWindow):
    def __init__(self, csv_path: str | Path):
        super().__init__()
        self.setWindowTitle("幾何・工程統合可視化アプリ（完全CSV対応版）")
        self.setGeometry(100, 80, 1700, 900)

        self.geom: Optional[GeometryModel] = None
        self.flow: Optional[FlowModel] = None
        self.state: Optional[AssemblyState] = None

        # __main__.py から渡されたCSVパスを使用
        self.data_path = Path(csv_path).expanduser()

        self.steps_mask: List[bool] = []
        self.step_checkboxes: List[QCheckBox] = []

        self.visual_selected_points = None  # GUI選択した点の情報

        # Monte Carlo結果保存用
        self.mc_results: Optional[pd.DataFrame] = None

        self._setup_ui()
        self._setup_file_watcher()
        self.reload_all()

    # ---------- UI ----------
    def _setup_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([400, 800, 500])
        main_layout.addWidget(splitter)

    def _create_left_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        file_grp = QGroupBox("ファイル操作"); file_l = QVBoxLayout(file_grp)
        btn_reload = QPushButton("再読み込み"); btn_reload.clicked.connect(self.reload_all); file_l.addWidget(btn_reload)
        btn_save_data = QPushButton("model_onefile.csv 保存"); btn_save_data.clicked.connect(self.save_data); file_l.addWidget(btn_save_data)
        lay.addWidget(file_grp)

        self.editor_tabs = QTabWidget()
        self.geom_editor = QTextEdit(); self.geom_editor.setFont(QFont("Courier", 9))
        self.flow_editor = QTextEdit(); self.flow_editor.setFont(QFont("Courier", 9))
        self.editor_tabs.addTab(self.geom_editor, "geometry (JSON)")
        self.editor_tabs.addTab(self.flow_editor, "flow (JSON)")
        lay.addWidget(self.editor_tabs)

        tree_grp = QGroupBox("インスタンス一覧"); tree_l = QVBoxLayout(tree_grp)
        self.instance_tree = QTreeWidget(); self.instance_tree.setHeaderLabels(["ID", "Prototype", "Tags"])
        tree_l.addWidget(self.instance_tree); lay.addWidget(tree_grp)
        return w

    def _create_center_panel(self) -> QWidget:
        w = QWidget(); lay = QVBoxLayout(w)

        # ステップ制御（flow.steps から動的に生成）
        step_grp = QGroupBox("工程ステップ"); step_l = QVBoxLayout(step_grp)
        self.step_box_container = QWidget(); self.step_box_layout = QVBoxLayout(self.step_box_container)
        step_l.addWidget(self.step_box_container)
        btn_update = QPushButton("ビュー更新"); btn_update.clicked.connect(self.update_view); step_l.addWidget(btn_update)
        lay.addWidget(step_grp)

        # 【機能1&2】表示オプション拡張
        disp_grp = QGroupBox("表示オプション"); disp_l = QVBoxLayout(disp_grp)
        
        # グループ表示
        self.show_a_cb = QCheckBox("A群表示"); self.show_a_cb.setChecked(True); self.show_a_cb.stateChanged.connect(self.update_view); disp_l.addWidget(self.show_a_cb)
        self.show_c_cb = QCheckBox("C群表示"); self.show_c_cb.setChecked(True); self.show_c_cb.stateChanged.connect(self.update_view); disp_l.addWidget(self.show_c_cb)
        
        # 【機能1】偏差表示モード
        self.deviation_mode_cb = QCheckBox("偏差表示モード")
        self.deviation_mode_cb.setChecked(False)
        self.deviation_mode_cb.stateChanged.connect(self.update_view)
        disp_l.addWidget(self.deviation_mode_cb)
        
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("許容公差 [mm]:"))
        self.tol_edit = QDoubleSpinBox()
        self.tol_edit.setRange(0.1, 100.0)
        self.tol_edit.setValue(5.0)
        self.tol_edit.setSingleStep(0.5)
        self.tol_edit.editingFinished.connect(self.update_view)
        tol_layout.addWidget(self.tol_edit)
        disp_l.addLayout(tol_layout)
        
        # 【機能2】変形倍率スライダー
        deform_layout = QVBoxLayout()
        deform_label_layout = QHBoxLayout()
        deform_label_layout.addWidget(QLabel("変形倍率:"))
        self.deform_scale_label = QLabel("x1")
        deform_label_layout.addWidget(self.deform_scale_label)
        deform_label_layout.addStretch()
        deform_layout.addLayout(deform_label_layout)
        
        self.deform_scale_slider = QSlider(Qt.Horizontal)
        self.deform_scale_slider.setRange(1, 100)
        self.deform_scale_slider.setValue(1)
        self.deform_scale_slider.setTickPosition(QSlider.TicksBelow)
        self.deform_scale_slider.setTickInterval(10)
        self.deform_scale_slider.valueChanged.connect(self._on_deform_scale_changed)
        deform_layout.addWidget(self.deform_scale_slider)
        
        disp_l.addLayout(deform_layout)
        
        lay.addWidget(disp_grp)

        if USE_OPEN3D:
            self.view_label = QLabel("Open3D ウィンドウで表示します（ボタン押下で更新）")
            self.view_label.setAlignment(Qt.AlignCenter)
            lay.addWidget(self.view_label)
            self.visualizer = Open3DVisualizer()
        else:
            self.visualizer = MatplotlibVisualizer()
            lay.addWidget(self.visualizer)

        return w

    def _create_right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # スクロール対応
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # 検証ログ
        log_grp = QGroupBox("検証"); log_l = QVBoxLayout(log_grp)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True); self.log_text.setMaximumHeight(150); log_l.addWidget(self.log_text)
        btn_validate = QPushButton("検証実行"); btn_validate.clicked.connect(self.run_validation); log_l.addWidget(btn_validate)
        scroll_layout.addWidget(log_grp)

        # 【機能3】Monte Carlo（ワースト/ベストボタン追加）
        mc_grp = QGroupBox("Monte Carlo"); mc_l = QVBoxLayout(mc_grp)
        row1 = QHBoxLayout(); row1.addWidget(QLabel("試行回数:")); self.mc_n_spin = QSpinBox(); self.mc_n_spin.setRange(10, 20000); self.mc_n_spin.setValue(200); row1.addWidget(self.mc_n_spin); mc_l.addLayout(row1)
        row2 = QHBoxLayout(); row2.addWidget(QLabel("乱数シード:")); self.mc_seed_spin = QSpinBox(); self.mc_seed_spin.setRange(0, 999999); self.mc_seed_spin.setValue(42); row2.addWidget(self.mc_seed_spin); mc_l.addLayout(row2)
        btn_mc = QPushButton("Monte Carlo 実行"); btn_mc.clicked.connect(self.run_monte_carlo); mc_l.addWidget(btn_mc)
        btn_save = QPushButton("結果保存 (CSV/PNG)"); btn_save.clicked.connect(self.save_monte_carlo_results); mc_l.addWidget(btn_save)
        
        # 【機能3】ワースト/ベストケース再現ボタン
        case_layout = QHBoxLayout()
        self.btn_show_worst = QPushButton("Show Worst Case")
        self.btn_show_worst.clicked.connect(self.show_worst_case)
        case_layout.addWidget(self.btn_show_worst)
        
        self.btn_show_best = QPushButton("Show Best Case")
        self.btn_show_best.clicked.connect(self.show_best_case)
        case_layout.addWidget(self.btn_show_best)
        
        mc_l.addLayout(case_layout)
        
        self.mc_result_text = QTextEdit(); self.mc_result_text.setReadOnly(True); self.mc_result_text.setMaximumHeight(150); mc_l.addWidget(self.mc_result_text)
        scroll_layout.addWidget(mc_grp)

        # ========== 任意2点距離（モード選択付き） ==========
        pair_grp = QGroupBox("任意2点の距離（MC）"); pair_l = QVBoxLayout(pair_grp)
        
        # モード選択
        mode_layout = QHBoxLayout()
        self.point_select_mode_group = QButtonGroup()
        
        self.dropdown_mode_radio = QRadioButton("プルダウン選択")
        self.dropdown_mode_radio.setChecked(True)
        self.dropdown_mode_radio.toggled.connect(self._toggle_point_selection_mode)
        self.point_select_mode_group.addButton(self.dropdown_mode_radio)
        mode_layout.addWidget(self.dropdown_mode_radio)
        
        self.visual_mode_radio = QRadioButton("GUI上で選択")
        self.visual_mode_radio.toggled.connect(self._toggle_point_selection_mode)
        self.point_select_mode_group.addButton(self.visual_mode_radio)
        mode_layout.addWidget(self.visual_mode_radio)
        
        pair_l.addLayout(mode_layout)
        
        # プルダウン選択用ウィジェット
        self.dropdown_widget = QWidget()
        dropdown_layout = QVBoxLayout(self.dropdown_widget)
        
        dropdown_layout.addWidget(QLabel("名前1:"))
        self.instref_name1 = QLineEdit("instref1")
        dropdown_layout.addWidget(self.instref_name1)
        
        dropdown_layout.addWidget(QLabel("点1 - インスタンス:"))
        self.p1_inst_combo = QComboBox()
        self.p1_inst_combo.currentIndexChanged.connect(self._update_p1_ref_combo)
        dropdown_layout.addWidget(self.p1_inst_combo)
        
        dropdown_layout.addWidget(QLabel("点1 - 参照:"))
        self.p1_ref_combo = QComboBox()
        dropdown_layout.addWidget(self.p1_ref_combo)
        
        dropdown_layout.addWidget(QLabel("名前2:"))
        self.instref_name2 = QLineEdit("instref2")
        dropdown_layout.addWidget(self.instref_name2)
        
        dropdown_layout.addWidget(QLabel("点2 - インスタンス:"))
        self.p2_inst_combo = QComboBox()
        self.p2_inst_combo.currentIndexChanged.connect(self._update_p2_ref_combo)
        dropdown_layout.addWidget(self.p2_inst_combo)
        
        dropdown_layout.addWidget(QLabel("点2 - 参照:"))
        self.p2_ref_combo = QComboBox()
        dropdown_layout.addWidget(self.p2_ref_combo)
        
        pair_l.addWidget(self.dropdown_widget)
        
        # GUI選択用ウィジェット
        self.visual_select_widget = QWidget()
        visual_layout = QVBoxLayout(self.visual_select_widget)
        
        self.open_visual_selector_btn = QPushButton("3Dビューで点を選択")
        self.open_visual_selector_btn.clicked.connect(self._open_visual_point_selector)
        visual_layout.addWidget(self.open_visual_selector_btn)
        
        self.selected_points_label = QLabel("選択された点: なし")
        self.selected_points_label.setWordWrap(True)
        visual_layout.addWidget(self.selected_points_label)
        
        self.visual_select_widget.setVisible(False)
        pair_l.addWidget(self.visual_select_widget)

        btn_pair = QPushButton("距離分布を表示")
        btn_pair.clicked.connect(self.run_pair_distance_mc)
        pair_l.addWidget(btn_pair)

        # ヒストグラム埋め込みウィジェット
        self.distance_histogram = DistanceHistogramWidget()
        pair_l.addWidget(self.distance_histogram)

        # 統計情報テキスト
        self.pair_result_text = QTextEdit()
        self.pair_result_text.setReadOnly(True)
        self.pair_result_text.setMaximumHeight(100)
        pair_l.addWidget(self.pair_result_text)

        scroll_layout.addWidget(pair_grp)

        scroll.setWidget(scroll_content)
        lay.addWidget(scroll)

        return w

    # ---------- 【機能2】変形倍率スライダー ----------
    def _on_deform_scale_changed(self):
        """変形倍率スライダー変更時"""
        scale = self.deform_scale_slider.value()
        self.deform_scale_label.setText(f"x{scale}")
        self.update_view()

    # ---------- モード切り替え ----------
    def _toggle_point_selection_mode(self):
        """選択モードの切り替え"""
        is_visual = self.visual_mode_radio.isChecked()
        self.dropdown_widget.setVisible(not is_visual)
        self.visual_select_widget.setVisible(is_visual)
    
    def _open_visual_point_selector(self):
        """ビジュアル点選択ダイアログを開く"""
        if not (self.geom and self.state):
            self.log_message("データ未ロード", "error")
            return
        
        # 現在のステップを適用
        self.apply_steps()
        
        # 選択ダイアログを作成
        dialog = QDialog(self)
        dialog.setWindowTitle("点を選択")
        dialog.setGeometry(200, 100, 1000, 800)
        
        layout = QVBoxLayout(dialog)
        
        selector = InteractivePointSelector()
        selector.set_geometry_and_state(
            self.geom, 
            self.state, 
            {"A": self.show_a_cb.isChecked(), "C": self.show_c_cb.isChecked()}
        )
        
        def on_selection_completed(result):
            self.visual_selected_points = result
            self.selected_points_label.setText(
                f"点1: {result['point1']['instance']}:{result['point1']['ref']}\n"
                f"点2: {result['point2']['instance']}:{result['point2']['ref']}"
            )
            dialog.accept()
        
        selector.selection_completed.connect(on_selection_completed)
        layout.addWidget(selector)
        
        dialog.exec()

    # ---------- プルダウン更新 ----------
    def _update_p1_ref_combo(self):
        self.p1_ref_combo.clear()
        if not self.geom:
            return
        inst_id = self.p1_inst_combo.currentText()
        if inst_id:
            refs = self.geom.get_available_refs_for_instance(inst_id)
            self.p1_ref_combo.addItems(refs)

    def _update_p2_ref_combo(self):
        self.p2_ref_combo.clear()
        if not self.geom:
            return
        inst_id = self.p2_inst_combo.currentText()
        if inst_id:
            refs = self.geom.get_available_refs_for_instance(inst_id)
            self.p2_ref_combo.addItems(refs)

    def _rebuild_instance_combos(self):
        """geometry 読み込み後、インスタンスリストをプルダウンに反映"""
        self.p1_inst_combo.clear()
        self.p2_inst_combo.clear()
        if not self.geom:
            return
        inst_ids = self.geom.get_instance_ids()
        self.p1_inst_combo.addItems(inst_ids)
        self.p2_inst_combo.addItems(inst_ids)

    # ---------- File watcher ----------
    def _setup_file_watcher(self):
        if not HAS_WATCHDOG: return
        self.observer = Observer()
        self.observer.schedule(FileChangeHandler(self.on_file_changed), str(self.data_path.parent), recursive=False)
        self.observer.start()

    def on_file_changed(self):
        QTimer.singleShot(400, self.reload_all)

    # ---------- Load/Save ----------
    def reload_all(self):
        try:
            # ★存在しない場合は必ずエラーにする
            if not self.data_path.exists():
                raise FileNotFoundError(f"model_onefile.csv not found: {self.data_path}")
    
            # CSVから両方のデータを読み込み
            geom_data, flow_data = load_data_from_csv(self.data_path)
    
            if geom_data:
                self.geom = GeometryModel(geom_data)
                # エディタにはJSON形式で表示
                self.geom_editor.setPlainText(json.dumps(geom_data, indent=2, ensure_ascii=False))
    
            if flow_data:
                self.flow = FlowModel(flow_data)
                self.flow_editor.setPlainText(json.dumps(flow_data, indent=2, ensure_ascii=False))
    
            if self.flow:
                self._rebuild_step_checkboxes()
            if self.geom:
                self.state = AssemblyState(self.geom)
                self._rebuild_instance_combos()
    
            self.update_instance_tree()
            self.update_view()
    
            # ★成功したときだけ「完了」を出す
            self.log_message(f"ファイル読み込み完了: {self.data_path}", "info")
    
        except Exception as e:
            self.log_message(f"読み込みエラー: {e}\n{traceback.format_exc()}", "error")
            # もし「必ず落としたい」なら、次の1行を有効化
            # raise

    def _rebuild_step_checkboxes(self):
        for i in reversed(range(self.step_box_layout.count())):
            item = self.step_box_layout.itemAt(i)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.step_checkboxes.clear()
        self.steps_mask = []
        for step in (self.flow.steps if self.flow else []):
            cb = QCheckBox(step.get("id", step.get("op","step")))
            cb.setChecked(False)
            cb.stateChanged.connect(self.update_steps_mask)
            self.step_box_layout.addWidget(cb)
            self.step_checkboxes.append(cb)
            self.steps_mask.append(False)

    def save_data(self):
        """
        エディタの内容をCSVファイルとして保存
        geometryとflowの両方を1つのCSVに保存
        """
        try:
            # エディタから取得したJSONデータ
            geom_data = json.loads(self.geom_editor.toPlainText())
            flow_data = json.loads(self.flow_editor.toPlainText())
            
            # JSON → CSV行に変換
            geom_rows = nested_dict_to_csv_rows(geom_data, 'geometry')
            flow_rows = nested_dict_to_csv_rows(flow_data, 'flow')
            
            # 両方を結合
            all_rows = geom_rows + flow_rows
            
            # DataFrameに変換して保存
            df = pd.DataFrame(all_rows)
            df.to_csv(self.data_path, index=False)
            
            self.log_message("model_onefile.csv 保存完了", "info")
            self.reload_all()
        except Exception as e:
            self.log_message(f"保存エラー: {e}\n{traceback.format_exc()}", "error")

    # ---------- View / Steps ----------
    def update_instance_tree(self):
        self.instance_tree.clear()
        if not self.geom: return
        for inst_id, inst in self.geom.instances.items():
            proto = inst.get("prototype",""); tags = ", ".join(inst.get("tags",[]))
            self.instance_tree.addTopLevelItem(QTreeWidgetItem([inst_id, proto, tags]))

    def update_steps_mask(self):
        self.steps_mask = [cb.isChecked() for cb in self.step_checkboxes]

    def apply_steps(self):
        if not (self.geom and self.flow): return
        self.state = AssemblyState(self.geom)
        rng = np.random.default_rng(42)
        engine = ProcessEngine(self.geom, self.flow, rng)
        for i, st in enumerate(self.flow.steps):
            if i < len(self.steps_mask) and self.steps_mask[i]:
                engine.apply_step(st, self.state)

    def update_view(self):
        """【機能1&2】偏差表示モード＋変形倍率を反映してビュー更新"""
        if not (self.geom and self.state): return
        try:
            self.apply_steps()
            show_groups = {"A": self.show_a_cb.isChecked(), "C": self.show_c_cb.isChecked()}
            
            # 【機能1】偏差表示モード
            deviation_mode = self.deviation_mode_cb.isChecked()
            tol_mm = float(self.tol_edit.value())
            
            # 【機能2】変形倍率
            deform_scale = float(self.deform_scale_slider.value())
            
            self.visualizer.build_scene(
                self.geom, self.state, show_groups,
                deviation_mode=deviation_mode,
                tol_mm=tol_mm,
                deform_scale=deform_scale
            )
            
            if USE_OPEN3D:
                o3d.visualization.draw_geometries(self.visualizer.get_geometries(),
                                                  window_name="Assembly View", width=900, height=640)
        except Exception as e:
            self.log_message(f"ビュー更新エラー: {e}\n{traceback.format_exc()}", "error")

    # ---------- Validation ----------
    def run_validation(self):
        if not (self.geom and self.flow):
            self.log_message("データ未ロード", "error")
            return
        issues = Validator.validate(self.geom, self.flow)
        if not issues:
            self.log_message("検証OK: 問題なし", "info")
        else:
            msg = "検証結果:\n" + "\n".join([f"[{i['level'].upper()}] {i['message']}" for i in issues])
            self.log_message(msg, "warning")

    # ---------- Monte Carlo ----------
    def run_monte_carlo(self):
        if not (self.geom and self.flow):
            self.log_message("データ未ロード", "error"); return
        try:
            n = self.mc_n_spin.value(); seed = self.mc_seed_spin.value()
            self.log_message(f"Monte Carlo 実行中… (N={n})", "info"); QApplication.processEvents()
            sim = MonteCarloSimulator(self.geom, self.flow)
            self.mc_results = sim.run(n, self.steps_mask, seed)
            lines = [f"Monte Carlo 結果 (N={n})"]
            for col in [c for c in self.mc_results.columns if c != "trial"][:6]:
                s = self.mc_results[col]
                lines.append(f"- {col}: mean={s.mean():.3f}, std={s.std():.3f}, 95%CI=[{s.quantile(0.025):.3f}, {s.quantile(0.975):.3f}]")
            self.mc_result_text.setPlainText("\n".join(lines))
            self.log_message("Monte Carlo 完了", "info")
        except Exception as e:
            self.log_message(f"Monte Carlo エラー: {e}\n{traceback.format_exc()}", "error")

    def save_monte_carlo_results(self):
        if not hasattr(self, "mc_results") or self.mc_results is None:
            self.log_message("実行結果がありません", "warning"); return
        try:
            csv_path = "monte_carlo_results.csv"
            self.mc_results.to_csv(csv_path, index=False)
            import matplotlib.pyplot as plt
            cols = [c for c in self.mc_results.columns if c != "trial"]
            if cols:
                col = "edge_gap_x" if "edge_gap_x" in cols else cols[0]
                plt.figure(figsize=(8,6))
                plt.hist(self.mc_results[col], bins=30, alpha=0.8, edgecolor="black")
                plt.xlabel(col + " [mm]"); plt.ylabel("Freq"); plt.grid(True, alpha=0.3)
                plt.title("Monte Carlo Histogram")
                png_path = "monte_carlo_histogram.png"
                plt.savefig(png_path, dpi=150); plt.close()
                self.log_message(f"保存完了: {csv_path}, {png_path}", "info")
            else:
                self.log_message(f"保存完了: {csv_path}（可視化対象なし）", "info")
        except Exception as e:
            self.log_message(f"保存エラー: {e}", "error")

    # ---------- 【機能3】ワースト/ベストケース再現 ----------
    def build_state_for_trial(self, trial: int, seed_base: int) -> AssemblyState:
        """
        指定された試行番号のアセンブリ状態を再構築
        """
        rng = np.random.default_rng(seed_base + trial)
        state = AssemblyState(self.geom)
        engine = ProcessEngine(self.geom, self.flow, rng)
        for i, st in enumerate(self.flow.steps):
            if i < len(self.steps_mask) and self.steps_mask[i]:
                engine.apply_step(st, state)
        return state
    
    def show_worst_case(self):
        """ワーストケース（最大edge_gap_x）を表示"""
        if not hasattr(self, "mc_results") or self.mc_results is None:
            self.log_message("Monte Carlo結果がありません", "warning")
            return
        
        try:
            # 対象メトリクスを決定
            cols = [c for c in self.mc_results.columns if c != "trial"]
            if "edge_gap_x" in cols:
                metric_col = "edge_gap_x"
            elif len(cols) > 0:
                metric_col = cols[0]
            else:
                self.log_message("計測メトリクスがありません", "warning")
                return
            
            # 最大値の試行を特定
            worst_trial = int(self.mc_results[metric_col].idxmax())
            worst_value = float(self.mc_results.loc[worst_trial, metric_col])
            
            # 状態を再構築
            seed_base = self.mc_seed_spin.value()
            state_worst = self.build_state_for_trial(worst_trial, seed_base)
            
            # 3Dビューを更新
            show_groups = {"A": self.show_a_cb.isChecked(), "C": self.show_c_cb.isChecked()}
            deviation_mode = self.deviation_mode_cb.isChecked()
            tol_mm = float(self.tol_edit.value())
            deform_scale = float(self.deform_scale_slider.value())
            
            self.visualizer.build_scene(
                self.geom, state_worst, show_groups,
                deviation_mode=deviation_mode,
                tol_mm=tol_mm,
                deform_scale=deform_scale
            )
            
            if USE_OPEN3D:
                o3d.visualization.draw_geometries(
                    self.visualizer.get_geometries(),
                    window_name=f"Worst Case (trial={worst_trial}, {metric_col}={worst_value:.3f}mm)",
                    width=900, height=640
                )
            
            self.log_message(
                f"Worst Case表示: trial={worst_trial}, {metric_col}={worst_value:.3f} mm",
                "info"
            )
            
        except Exception as e:
            self.log_message(f"Worst Case表示エラー: {e}\n{traceback.format_exc()}", "error")
    
    def show_best_case(self):
        """ベストケース（最小edge_gap_x）を表示"""
        if not hasattr(self, "mc_results") or self.mc_results is None:
            self.log_message("Monte Carlo結果がありません", "warning")
            return
        
        try:
            # 対象メトリクスを決定
            cols = [c for c in self.mc_results.columns if c != "trial"]
            if "edge_gap_x" in cols:
                metric_col = "edge_gap_x"
            elif len(cols) > 0:
                metric_col = cols[0]
            else:
                self.log_message("計測メトリクスがありません", "warning")
                return
            
            # 最小値の試行を特定
            best_trial = int(self.mc_results[metric_col].idxmin())
            best_value = float(self.mc_results.loc[best_trial, metric_col])
            
            # 状態を再構築
            seed_base = self.mc_seed_spin.value()
            state_best = self.build_state_for_trial(best_trial, seed_base)
            
            # 3Dビューを更新
            show_groups = {"A": self.show_a_cb.isChecked(), "C": self.show_c_cb.isChecked()}
            deviation_mode = self.deviation_mode_cb.isChecked()
            tol_mm = float(self.tol_edit.value())
            deform_scale = float(self.deform_scale_slider.value())
            
            self.visualizer.build_scene(
                self.geom, state_best, show_groups,
                deviation_mode=deviation_mode,
                tol_mm=tol_mm,
                deform_scale=deform_scale
            )
            
            if USE_OPEN3D:
                o3d.visualization.draw_geometries(
                    self.visualizer.get_geometries(),
                    window_name=f"Best Case (trial={best_trial}, {metric_col}={best_value:.3f}mm)",
                    width=900, height=640
                )
            
            self.log_message(
                f"Best Case表示: trial={best_trial}, {metric_col}={best_value:.3f} mm",
                "info"
            )
            
        except Exception as e:
            self.log_message(f"Best Case表示エラー: {e}\n{traceback.format_exc()}", "error")

    # ---------- 任意2点距離（両モード対応） ----------
    def run_pair_distance_mc(self):
        if not (self.geom and self.flow):
            self.log_message("データ未ロード", "error"); return
        try:
            # モードに応じて点情報を取得
            if self.visual_mode_radio.isChecked():
                if not self.visual_selected_points:
                    self.log_message("まず3Dビューで点を選択してください", "warning")
                    return
                
                i1 = self.visual_selected_points['point1']['instance']
                r1 = self.visual_selected_points['point1']['ref']
                i2 = self.visual_selected_points['point2']['instance']
                r2 = self.visual_selected_points['point2']['ref']
                name1 = f"{i1}:{r1}"
                name2 = f"{i2}:{r2}"
            else:
                # プルダウンモード
                i1 = self.p1_inst_combo.currentText()
                r1 = self.p1_ref_combo.currentText()
                i2 = self.p2_inst_combo.currentText()
                r2 = self.p2_ref_combo.currentText()
                
                if not (i1 and r1 and i2 and r2):
                    self.log_message("インスタンスと参照を選択してください", "warning")
                    return
                
                name1 = self.instref_name1.text() or f"{i1}:{r1}"
                name2 = self.instref_name2.text() or f"{i2}:{r2}"

            n = self.mc_n_spin.value(); seed = self.mc_seed_spin.value()
            self.log_message(f"距離分布計算中… (N={n})", "info"); QApplication.processEvents()

            vals = []
            for t in range(n):
                rng = np.random.default_rng(seed + t)
                state = AssemblyState(self.geom)
                engine = ProcessEngine(self.geom, self.flow, rng)
                for i, st in enumerate(self.flow.steps):
                    if i < len(self.steps_mask) and self.steps_mask[i]:
                        engine.apply_step(st, state)
                P1 = get_world_point(self.geom, state, i1, r1)
                P2 = get_world_point(self.geom, state, i2, r2)
                vals.append(float(np.linalg.norm(P2 - P1)))

            vals = np.asarray(vals, dtype=float)
            
            # アプリ内にヒストグラム描画
            self.distance_histogram.plot_histogram(vals, name1, name2, n)

            # 統計情報表示（平均・標準偏差を明記）
            mean_val = vals.mean()
            std_val = vals.std()
            lo, hi = np.quantile(vals, [0.025, 0.975])
            
            self.pair_result_text.setPlainText(
                f"=== 任意2点間距離の統計 ===\n"
                f"試行回数: N={n}\n"
                f"点1: {name1}\n"
                f"点2: {name2}\n"
                f"\n"
                f"平均 (mean):      {mean_val:.3f} mm\n"
                f"標準偏差 (std):   {std_val:.3f} mm\n"
                f"95%信頼区間:      [{lo:.3f}, {hi:.3f}] mm"
            )
            self.log_message("距離分布計算完了", "info")

        except Exception as e:
            self.log_message(f"距離分布エラー: {e}\n{traceback.format_exc()}", "error")

    # ---------- Log ----------
    def log_message(self, message: str, level: str="info"):
        color = {"info":"black","warning":"orange","error":"red"}.get(level,"black")
        self.log_text.append(f'<span style="color:{color};">[{level.upper()}] {message}</span>')

    def closeEvent(self, event):
        if HAS_WATCHDOG and hasattr(self, "observer"):
            self.observer.stop(); self.observer.join()
        event.accept()

