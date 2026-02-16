"""Main window entrypoint for sov_app GUI.

Step 2 extracts small GUI widgets from the legacy one-file implementation so
this module can focus on screen composition and event wiring.
"""

from __future__ import annotations

import app_onefile
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)

from sov_app.widgets import (
    DisplayOptionsWidget,
    FileActionsWidget,
    MonteCarloWidget,
    PairDistanceWidget,
    ValidationWidget,
)


class MainWindow(app_onefile.MainWindow):
    """Compatibility wrapper that rewires small widgets via ``sov_app.widgets``."""

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        file_actions = FileActionsWidget()
        file_actions.reload_button.clicked.connect(self.reload_all)
        file_actions.save_data_button.clicked.connect(self.save_data)
        layout.addWidget(file_actions)

        self.editor_tabs = QTabWidget()
        self.geom_editor = QTextEdit()
        self.geom_editor.setFont(QFont("Courier", 9))
        self.flow_editor = QTextEdit()
        self.flow_editor.setFont(QFont("Courier", 9))
        self.editor_tabs.addTab(self.geom_editor, "geometry (JSON)")
        self.editor_tabs.addTab(self.flow_editor, "flow (JSON)")
        layout.addWidget(self.editor_tabs)

        tree_group = QGroupBox("インスタンス一覧")
        tree_layout = QVBoxLayout(tree_group)
        self.instance_tree = QTreeWidget()
        self.instance_tree.setHeaderLabels(["ID", "Prototype", "Tags"])
        tree_layout.addWidget(self.instance_tree)
        layout.addWidget(tree_group)

        return panel

    def _create_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        step_group = QGroupBox("工程ステップ")
        step_layout = QVBoxLayout(step_group)
        self.step_box_container = QWidget()
        self.step_box_layout = QVBoxLayout(self.step_box_container)
        step_layout.addWidget(self.step_box_container)

        update_button = QPushButton("ビュー更新")
        update_button.clicked.connect(self.update_view)
        step_layout.addWidget(update_button)
        layout.addWidget(step_group)

        display_options = DisplayOptionsWidget()
        display_options.show_a_cb.stateChanged.connect(self.update_view)
        display_options.show_c_cb.stateChanged.connect(self.update_view)
        display_options.deviation_mode_cb.stateChanged.connect(self.update_view)
        display_options.tol_edit.editingFinished.connect(self.update_view)
        display_options.deform_scale_slider.valueChanged.connect(self._on_deform_scale_changed)
        layout.addWidget(display_options)

        self.show_a_cb = display_options.show_a_cb
        self.show_c_cb = display_options.show_c_cb
        self.deviation_mode_cb = display_options.deviation_mode_cb
        self.tol_edit = display_options.tol_edit
        self.deform_scale_label = display_options.deform_scale_label
        self.deform_scale_slider = display_options.deform_scale_slider

        if app_onefile.USE_OPEN3D:
            self.view_label = QLabel("Open3D ウィンドウで表示します（ボタン押下で更新）")
            self.view_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.view_label)
            self.visualizer = app_onefile.Open3DVisualizer()
        else:
            self.visualizer = app_onefile.MatplotlibVisualizer()
            layout.addWidget(self.visualizer)

        return panel

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        validation_widget = ValidationWidget()
        validation_widget.validate_button.clicked.connect(self.run_validation)
        scroll_layout.addWidget(validation_widget)
        self.log_text = validation_widget.log_text

        monte_carlo_widget = MonteCarloWidget()
        monte_carlo_widget.run_mc_button.clicked.connect(self.run_monte_carlo)
        monte_carlo_widget.save_results_button.clicked.connect(self.save_monte_carlo_results)
        monte_carlo_widget.btn_show_worst.clicked.connect(self.show_worst_case)
        monte_carlo_widget.btn_show_best.clicked.connect(self.show_best_case)
        scroll_layout.addWidget(monte_carlo_widget)

        self.mc_n_spin = monte_carlo_widget.mc_n_spin
        self.mc_seed_spin = monte_carlo_widget.mc_seed_spin
        self.btn_show_worst = monte_carlo_widget.btn_show_worst
        self.btn_show_best = monte_carlo_widget.btn_show_best
        self.mc_result_text = monte_carlo_widget.mc_result_text

        self.distance_histogram = app_onefile.DistanceHistogramWidget()
        pair_distance_widget = PairDistanceWidget(self.distance_histogram)
        pair_distance_widget.dropdown_mode_radio.toggled.connect(self._toggle_point_selection_mode)
        pair_distance_widget.visual_mode_radio.toggled.connect(self._toggle_point_selection_mode)
        pair_distance_widget.p1_inst_combo.currentIndexChanged.connect(self._update_p1_ref_combo)
        pair_distance_widget.p2_inst_combo.currentIndexChanged.connect(self._update_p2_ref_combo)
        pair_distance_widget.open_visual_selector_btn.clicked.connect(self._open_visual_point_selector)
        pair_distance_widget.run_pair_button.clicked.connect(self.run_pair_distance_mc)
        scroll_layout.addWidget(pair_distance_widget)

        self.point_select_mode_group = pair_distance_widget.point_select_mode_group
        self.dropdown_mode_radio = pair_distance_widget.dropdown_mode_radio
        self.visual_mode_radio = pair_distance_widget.visual_mode_radio
        self.dropdown_widget = pair_distance_widget.dropdown_widget
        self.instref_name1 = pair_distance_widget.instref_name1
        self.p1_inst_combo = pair_distance_widget.p1_inst_combo
        self.p1_ref_combo = pair_distance_widget.p1_ref_combo
        self.instref_name2 = pair_distance_widget.instref_name2
        self.p2_inst_combo = pair_distance_widget.p2_inst_combo
        self.p2_ref_combo = pair_distance_widget.p2_ref_combo
        self.visual_select_widget = pair_distance_widget.visual_select_widget
        self.open_visual_selector_btn = pair_distance_widget.open_visual_selector_btn
        self.selected_points_label = pair_distance_widget.selected_points_label
        self.pair_result_text = pair_distance_widget.pair_result_text

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        return panel

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_center_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setSizes([400, 800, 500])

        main_layout.addWidget(splitter)
