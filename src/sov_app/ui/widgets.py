"""Reusable small GUI widgets extracted from the main window layout."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class FileActionsWidget(QGroupBox):
    """File related action buttons."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("ファイル操作", parent)
        layout = QVBoxLayout(self)

        self.reload_button = QPushButton("再読み込み")
        layout.addWidget(self.reload_button)

        self.save_data_button = QPushButton("model_onefile.csv 保存")
        layout.addWidget(self.save_data_button)


class DisplayOptionsWidget(QGroupBox):
    """Display option controls used in the center panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("表示オプション", parent)
        layout = QVBoxLayout(self)

        self.show_a_cb = QCheckBox("A群表示")
        self.show_a_cb.setChecked(True)
        layout.addWidget(self.show_a_cb)

        self.show_c_cb = QCheckBox("C群表示")
        self.show_c_cb.setChecked(True)
        layout.addWidget(self.show_c_cb)

        self.deviation_mode_cb = QCheckBox("偏差表示モード")
        self.deviation_mode_cb.setChecked(False)
        layout.addWidget(self.deviation_mode_cb)

        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("許容公差 [mm]:"))
        self.tol_edit = QDoubleSpinBox()
        self.tol_edit.setRange(0.1, 100.0)
        self.tol_edit.setValue(5.0)
        self.tol_edit.setSingleStep(0.5)
        tol_layout.addWidget(self.tol_edit)
        layout.addLayout(tol_layout)

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
        deform_layout.addWidget(self.deform_scale_slider)

        layout.addLayout(deform_layout)


class ValidationWidget(QGroupBox):
    """Validation log and action controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("検証", parent)
        layout = QVBoxLayout(self)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        layout.addWidget(self.log_text)

        self.validate_button = QPushButton("検証実行")
        layout.addWidget(self.validate_button)


class MonteCarloWidget(QGroupBox):
    """Monte Carlo settings and action controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Monte Carlo", parent)
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("試行回数:"))
        self.mc_n_spin = QSpinBox()
        self.mc_n_spin.setRange(10, 20000)
        self.mc_n_spin.setValue(200)
        row1.addWidget(self.mc_n_spin)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("乱数シード:"))
        self.mc_seed_spin = QSpinBox()
        self.mc_seed_spin.setRange(0, 999999)
        self.mc_seed_spin.setValue(42)
        row2.addWidget(self.mc_seed_spin)
        layout.addLayout(row2)

        self.run_mc_button = QPushButton("Monte Carlo 実行")
        layout.addWidget(self.run_mc_button)

        self.save_results_button = QPushButton("結果保存 (CSV/PNG)")
        layout.addWidget(self.save_results_button)

        case_layout = QHBoxLayout()
        self.btn_show_worst = QPushButton("Show Worst Case")
        case_layout.addWidget(self.btn_show_worst)

        self.btn_show_best = QPushButton("Show Best Case")
        case_layout.addWidget(self.btn_show_best)
        layout.addLayout(case_layout)

        self.mc_result_text = QTextEdit()
        self.mc_result_text.setReadOnly(True)
        self.mc_result_text.setMaximumHeight(150)
        layout.addWidget(self.mc_result_text)


class PairDistanceWidget(QGroupBox):
    """Controls for arbitrary pair distance Monte Carlo."""

    def __init__(self, distance_histogram: QWidget, parent: QWidget | None = None) -> None:
        super().__init__("任意2点の距離（MC）", parent)
        layout = QVBoxLayout(self)

        mode_layout = QHBoxLayout()
        self.point_select_mode_group = QButtonGroup(self)

        self.dropdown_mode_radio = QRadioButton("プルダウン選択")
        self.dropdown_mode_radio.setChecked(True)
        self.point_select_mode_group.addButton(self.dropdown_mode_radio)
        mode_layout.addWidget(self.dropdown_mode_radio)

        self.visual_mode_radio = QRadioButton("GUI上で選択")
        self.point_select_mode_group.addButton(self.visual_mode_radio)
        mode_layout.addWidget(self.visual_mode_radio)
        layout.addLayout(mode_layout)

        self.dropdown_widget = QWidget()
        dropdown_layout = QVBoxLayout(self.dropdown_widget)

        dropdown_layout.addWidget(QLabel("名前1:"))
        self.instref_name1 = QLineEdit("instref1")
        dropdown_layout.addWidget(self.instref_name1)

        dropdown_layout.addWidget(QLabel("点1 - インスタンス:"))
        self.p1_inst_combo = QComboBox()
        dropdown_layout.addWidget(self.p1_inst_combo)

        dropdown_layout.addWidget(QLabel("点1 - 参照:"))
        self.p1_ref_combo = QComboBox()
        dropdown_layout.addWidget(self.p1_ref_combo)

        dropdown_layout.addWidget(QLabel("名前2:"))
        self.instref_name2 = QLineEdit("instref2")
        dropdown_layout.addWidget(self.instref_name2)

        dropdown_layout.addWidget(QLabel("点2 - インスタンス:"))
        self.p2_inst_combo = QComboBox()
        dropdown_layout.addWidget(self.p2_inst_combo)

        dropdown_layout.addWidget(QLabel("点2 - 参照:"))
        self.p2_ref_combo = QComboBox()
        dropdown_layout.addWidget(self.p2_ref_combo)

        layout.addWidget(self.dropdown_widget)

        self.visual_select_widget = QWidget()
        visual_layout = QVBoxLayout(self.visual_select_widget)

        self.open_visual_selector_btn = QPushButton("3Dビューで点を選択")
        visual_layout.addWidget(self.open_visual_selector_btn)

        self.selected_points_label = QLabel("選択された点: なし")
        self.selected_points_label.setWordWrap(True)
        visual_layout.addWidget(self.selected_points_label)

        self.visual_select_widget.setVisible(False)
        layout.addWidget(self.visual_select_widget)

        self.run_pair_button = QPushButton("距離分布を表示")
        layout.addWidget(self.run_pair_button)

        layout.addWidget(distance_histogram)

        self.pair_result_text = QTextEdit()
        self.pair_result_text.setReadOnly(True)
        self.pair_result_text.setMaximumHeight(100)
        layout.addWidget(self.pair_result_text)
