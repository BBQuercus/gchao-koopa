import configparser
import glob
import re
import os

from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
)
from qtpy.QtCore import Qt
import napari
import numpy as np
import pandas as pd
import tifffile


class KoopaWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        # General config
        self.viewer = napari_viewer
        self.spots_cols = ["frame", "y", "x"]
        self.track_cols = ["particle", "frame", "y", "x"]
        self.params = [
            "blending",
            "color_mode",
            "colormap",
            "contour",
            "contrast_limits",
            "face_color",
            "gamma",
            "opacity",
            "out_of_slice_display",
            "size",
            "symbol",
            "visible",
        ]
        self.luigi = False

        # Viewer model params - https://napari.org/stable/api/napari.Viewer
        self.image_params = dict(blending="additive")
        self.label_params = dict(blending="translucent", opacity=0.7)
        self.point_params = dict(
            face_color="white",
            size=5,
            out_of_slice_display=True,
            opacity=1.0,
            border_width=1,
            border_color="white",
        )
        self.track_params = dict(tail_width=8, tail_length=30, head_length=2)

        # State tracking
        self.auto_hide_previous = False
        self.current_file_layers = []

        # Build plugin layout
        self.setLayout(QVBoxLayout())
        self.setup_logo_header()
        self.setup_config_parser()
        self.setup_file_dropdown()
        self.setup_file_navigation()
        self.setup_viewing_options()
        self.setup_progress_bar()

    def toggle_old_naming(self):
        self.luigi = not self.luigi
        napari.utils.notifications.show_info(
            f"Turned old naming scheme {'on' if self.luigi else 'off'}."
        )

    def clear_viewer(self):
        """Reset all layers to an empty window."""
        self.viewer.reset_view()
        self.viewer.layers.clear()

    def setup_logo_header(self):
        """Prepare widget for header."""
        widget = QWidget()
        widget.setLayout(QHBoxLayout())
        widget.layout().addWidget(QLabel("<h1>Koopa</h1>"))
        widget.layout().addWidget(
            QLabel("Keenly optimized obliging picture analysis.")
        )
        self.layout().addWidget(widget)

    def setup_config_parser(self):
        """Prepare widget for config reader."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Analysis Directory:</b>"))

        btn_widget = QPushButton("Select")
        btn_widget.clicked.connect(self.open_file_dialog)
        widget.layout().addWidget(btn_widget)

        luigi_widget = QPushButton("Toggle old naming")
        luigi_widget.clicked.connect(self.toggle_old_naming)
        widget.layout().addWidget(luigi_widget)
        self.layout().addWidget(widget)

    def setup_file_dropdown(self):
        """Prepare widget for single file selection."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Current File:</b>"))

        self.dropdown_widget = QComboBox()
        widget.layout().addWidget(self.dropdown_widget)

        btn_widget = QPushButton("Load")
        btn_widget.clicked.connect(self.load_file)
        btn_widget.setToolTip("Load the selected file and all associated data")
        widget.layout().addWidget(btn_widget)

        self.file_dropdown = widget
        self.file_dropdown.setDisabled(True)
        self.layout().addWidget(self.file_dropdown)

    def setup_file_navigation(self):
        """Prepare widget for file navigation."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Navigate Files:</b>"))
        prev_widget = QPushButton("Previous Image (←)")
        prev_widget.clicked.connect(lambda: self.change_file("prev"))
        prev_widget.setToolTip("Navigate to previous file")
        widget.layout().addWidget(prev_widget)
        next_widget = QPushButton("Next Image (→)")
        next_widget.clicked.connect(lambda: self.change_file("next"))
        next_widget.setToolTip("Navigate to next file")
        widget.layout().addWidget(next_widget)

        self.file_navigation = widget
        self.file_navigation.setDisabled(True)
        self.layout().addWidget(self.file_navigation)

    def setup_viewing_options(self):
        """Prepare widget accessibility options."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Viewing Options:</b>"))

        # Auto-hide checkbox
        self.auto_hide_checkbox = QCheckBox("Auto-hide previous layers")
        self.auto_hide_checkbox.setChecked(False)
        self.auto_hide_checkbox.stateChanged.connect(self.toggle_auto_hide)
        widget.layout().addWidget(self.auto_hide_checkbox)

        hideall_widget = QPushButton("Hide All Layers")
        hideall_widget.clicked.connect(self.hide_layers)
        widget.layout().addWidget(hideall_widget)

        autocontrast_widget = QPushButton("Auto-Contrast Raw Images")
        autocontrast_widget.clicked.connect(self.auto_contrast_raw_images)
        autocontrast_widget.setToolTip(
            "Apply automatic contrast adjustment to all raw image channels"
        )
        widget.layout().addWidget(autocontrast_widget)

        settings_save_widget = QPushButton("Save Settings")
        settings_save_widget.clicked.connect(self.save_settings)
        widget.layout().addWidget(settings_save_widget)
        settings_apply_widget = QPushButton("Apply Settings")
        settings_apply_widget.clicked.connect(self.apply_settings)
        widget.layout().addWidget(settings_apply_widget)
        self.layout().addWidget(widget)

    def setup_progress_bar(self):
        """Prepare widget with data loading progress bar."""
        widget = QWidget()
        widget.setLayout(QVBoxLayout())
        widget.layout().addWidget(QLabel("<b>Loading Status:</b>"))
        self.pbar = QProgressBar(self)
        self.pbar.setValue(0)
        widget.layout().addWidget(self.pbar)
        self.status_label = QLabel("Ready")
        widget.layout().addWidget(self.status_label)
        self.layout().addWidget(widget)

    def open_file_dialog(self):
        """Dialog to select analysis directory."""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        if dialog.exec_():
            self.clear_viewer()
            self.analysis_path = dialog.selectedFiles()[0]
            self.run_config_parser()
            self.get_file_list()

    def run_config_parser(self):
        """Retrieve config file from analysis directory."""
        config_file = os.path.join(
            os.path.abspath(self.analysis_path), "koopa.cfg"
        )
        if not os.path.exists(config_file):
            napari.utils.notifications.show_error(
                "Koopa config file does not exist!"
            )
            return None

        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.do_timeseries = self.config.getboolean("General", "do_timeseries")
        self.do_3d = self.config.getboolean("General", "do_3d")

        # Load channel names if available
        if self.config.has_option("General", "channel_names"):
            try:
                self.channel_names = eval(
                    self.config.get("General", "channel_names")
                )
            except Exception:
                self.channel_names = []
        else:
            self.channel_names = []

    def load_file(self):
        """Open all associated files and enable editing."""
        self.name = self.dropdown_widget.currentText()

        # Canvas reset
        self.pbar.setValue(0)
        self.status_label.setText(f"Loading: {self.name}")

        # Handle auto-hide if enabled
        if self.auto_hide_checkbox.isChecked():
            for layer in self.current_file_layers:
                if layer in self.viewer.layers:
                    layer.visible = False
        else:
            self.clear_viewer()

        self.current_file_layers = []
        self.pbar.setValue(10)

        # Images
        self.load_image()
        self.pbar.setValue(25)

        # Segmaps
        self.load_segmentation_cells()
        self.pbar.setValue(50)
        if self.config.getboolean(
            "SegmentationOther", "sego_enabled", fallback=False
        ):
            self.load_segmentation_other()
        self.pbar.setValue(75)

        # Points
        self.load_detection_raw()
        self.pbar.setValue(85)
        if self.config.getboolean(
            "SpotsColocalization", "coloc_enabled", fallback=False
        ):
            self.load_colocalization()
        self.pbar.setValue(100)
        self.status_label.setText(f"Loaded: {self.name}")

    def change_file(self, option: str):
        """Navigation prev/next to change files faster."""
        file_idx = self.files.index(self.dropdown_widget.currentText())
        if file_idx == 0 and option == "prev":
            napari.utils.notifications.show_warning(
                "Already at the first file!"
            )
            return
        if file_idx == len(self.files) - 1 and option == "next":
            napari.utils.notifications.show_warning(
                "Already at the last file!"
            )
            return

        new_idx = file_idx + 1 if option == "next" else file_idx - 1
        self.dropdown_widget.setCurrentText(self.files[new_idx])
        self.load_file()

    def get_file_list(self):
        """Find all files in analysis directory and update dropdown."""
        files = sorted(
            glob.glob(
                os.path.join(self.analysis_path, "preprocessed", "*.tif")
            )
        )

        if not files:
            napari.utils.notifications.show_error(
                "No preprocessed files found in the selected directory!"
            )
            return

        # Remove extensions
        files_clean = [os.path.basename(f).replace(".tif", "") for f in files]

        if not self.luigi:
            # Extract only the portion before the last hyphen or before `AxxZxxCxx` if that's present
            cleaned = []
            for name in files_clean:
                match = re.match(r"(.+?)(?:-[A]\d{2}Z\d{2}C\d{2})?$", name)
                if match:
                    cleaned.append(match.group(1))
            self.files = sorted(set(cleaned))
        else:
            self.files = sorted(set(files_clean))

        self.file_dropdown.setDisabled(False)
        self.file_navigation.setDisabled(False)
        self.dropdown_widget.clear()
        self.dropdown_widget.addItems(self.files)
        napari.utils.notifications.show_info(
            f"Found {len(self.files)} file{'s' if len(self.files) != 1 else ''}"
        )

    def get_channel_colormap(
        self, channel_idx: int, channel_name: str = None
    ) -> str:
        """Determine appropriate colormap for a channel based on name and config."""
        # Try to get channel name from config if available
        if (
            channel_name is None
            and hasattr(self, "channel_names")
            and channel_idx < len(self.channel_names)
        ):
            channel_name = self.channel_names[channel_idx]

        # Check config semantics to see if this channel is designated as nuclear or cyto
        used_colors = set()
        if hasattr(self, "config") and self.config:
            try:
                if self.config.has_option(
                    "SegmentationCells", "channel_nuclei"
                ):
                    nuclear_channel = self.config.getint(
                        "SegmentationCells", "channel_nuclei"
                    )
                    if channel_idx == nuclear_channel:
                        return "blue"
                    used_colors.add("blue")
                if self.config.has_option("SegmentationCells", "channel_cyto"):
                    cyto_channel = self.config.getint(
                        "SegmentationCells", "channel_cyto"
                    )
                    if channel_idx == cyto_channel:
                        return "yellow"
                    used_colors.add("yellow")
            except Exception:
                pass

        # Fallback to index-based color scheme, excluding already-used colors
        all_colors = ["blue", "yellow", "green", "red", "gray", "cyan"]
        available_colors = [c for c in all_colors if c not in used_colors]

        if not available_colors:
            available_colors = all_colors

        return available_colors[channel_idx % len(available_colors)]

    def load_image(self):
        """Open and display raw image data."""
        if self.luigi:
            fname = os.path.join(
                self.analysis_path, "preprocessed", f"{self.name}.tif"
            )
        else:
            files = glob.glob(
                os.path.join(
                    self.analysis_path, "preprocessed", f"{self.name}*.tif"
                )
            )
            if not files:
                napari.utils.notifications.show_error(
                    f"No file found for {self.name}"
                )
                return
            fname = files[0]
        self.image = tifffile.imread(fname)
        for idx, channel in enumerate(self.image):
            # Get channel name if available
            channel_name = None
            if hasattr(self, "channel_names") and idx < len(
                self.channel_names
            ):
                channel_name = self.channel_names[idx]
                layer_name = f"{channel_name} (C{idx})"
            else:
                layer_name = f"Channel {idx}"

            # Get appropriate colormap
            colormap = self.get_channel_colormap(idx, channel_name)

            # Add image with colormap
            layer = self.viewer.add_image(
                channel,
                name=layer_name,
                colormap=colormap,
                **self.image_params,
            )
            self.current_file_layers.append(layer)

    def load_segmentation_cells(self):
        """Open and display nuclear/cytoplasmic segmentation maps."""
        for name in ["nuclei", "cyto"]:
            if self.luigi:
                fname = os.path.join(
                    self.analysis_path,
                    f"segmentation_{name}",
                    f"{self.name}.tif",
                )
            else:
                files = glob.glob(
                    os.path.join(
                        self.analysis_path,
                        f"segmentation_{name}",
                        f"{self.name}*.tif",
                    )
                )
                if not files:
                    continue
                fname = files[0]
            if not os.path.exists(fname):
                continue
            segmap = tifffile.imread(fname).astype(int)
            layer = self.viewer.add_labels(
                segmap,
                name=f"Segmentation {name.capitalize()}",
                **self.label_params,
            )
            self.current_file_layers.append(layer)

    def load_segmentation_other(self):
        """Open and display additional segmentation maps."""
        channels_str = self.config.get(
            "SegmentationOther", "sego_channels", fallback="[]"
        )
        for channel in eval(channels_str):
            if self.luigi:
                fname = os.path.join(
                    self.analysis_path,
                    f"segmentation_{channel}",
                    f"{self.name}.tif",
                )
            else:
                files = glob.glob(
                    os.path.join(
                        self.analysis_path,
                        f"segmentation_{channel}",
                        f"{self.name}*.tif",
                    )
                )
                if not files:
                    napari.utils.notifications.show_warning(
                        f"No segmentation file for channel {channel}"
                    )
                    continue
                fname = files[0]
            segmap = tifffile.imread(fname).astype(int)
            layer = self.viewer.add_labels(
                segmap, name=f"Segmentation C{channel}", **self.label_params
            )
            self.current_file_layers.append(layer)

    def load_detection_raw(self):
        """Open and display raw spot detection points."""

        channels_str = self.config.get(
            "SpotsDetection", "detect_channels", fallback="[]"
        )
        for channel in eval(channels_str):
            folder = (
                f"detection_final_c{channel}"
                if self.do_3d or self.do_timeseries
                else f"detection_raw_c{channel}"
            )
            if self.luigi:
                fname = os.path.join(
                    self.analysis_path, folder, f"{self.name}.parq"
                )
            else:
                files = glob.glob(
                    os.path.join(
                        self.analysis_path, folder, f"{self.name}*.parq"
                    )
                )
                if not files:
                    napari.utils.notifications.show_warning(
                        f"No detection file in {folder}"
                    )
                    continue
                fname = files[0]
            df = pd.read_parquet(fname)

            if self.do_timeseries:
                layer = self.viewer.add_tracks(
                    df[self.track_cols],
                    name=f"Track C{channel}",
                    **self.track_params,
                )
            else:
                layer = self.viewer.add_points(
                    df[self.spots_cols],
                    name=f"Detection C{channel}",
                    **self.point_params,
                )
            self.current_file_layers.append(layer)

    def load_colocalization(self):
        """Open and display colocalization pairs (colocalized vs. non)."""

        channels_str = self.config.get(
            "SpotsColocalization", "coloc_channels", fallback="[]"
        )
        for i, j in eval(channels_str):
            if self.luigi:
                fname = os.path.join(
                    self.analysis_path,
                    f"colocalization_{i}-{j}",
                    f"{self.name}.parq",
                )
            else:
                files = glob.glob(
                    os.path.join(
                        self.analysis_path,
                        f"colocalization_{i}-{j}",
                        f"{self.name}*.parq",
                    )
                )
                if not files:
                    napari.utils.notifications.show_warning(
                        f"No colocalization file for {i}-{j}"
                    )
                    continue
                fname = files[0]
            df = pd.read_parquet(fname)
            df_coloc = df[df["channel"] == i]
            df_empty = df[df["channel"] == j]

            if self.do_timeseries:
                df_coloc = df_coloc.loc[
                    df_coloc[f"coloc_particle_{i}-{j}"].isna(), self.track_cols
                ]
                df_empty = df_empty.loc[
                    ~df_empty[f"coloc_particle_{i}-{j}"].isna(),
                    self.track_cols,
                ]
                self.viewer.add_tracks(
                    df_coloc,
                    name=f"Track {i}-{j} Coloc",
                    colormap="red",
                    **self.track_params,
                )
                self.viewer.add_tracks(
                    df_empty,
                    name=f"Track {i}-{j} Empty",
                    colormap="blue",
                    **self.track_params,
                )
            else:
                df_coloc = df_coloc.loc[
                    df_coloc[f"coloc_particle_{i}-{j}"] != 0, self.spots_cols
                ]
                df_empty = df_empty.loc[
                    df_empty[f"coloc_particle_{i}-{j}"] == 0, self.spots_cols
                ]
                bland_point_params = self.point_params.copy()
                bland_point_params.pop("face_color", None)
                bland_point_params.pop("border_color", None)
                self.viewer.add_points(
                    df_coloc,
                    name=f"Detection {i}-{j} Coloc",
                    face_color="red",
                    border_color="red",
                    **bland_point_params,
                )
                self.viewer.add_points(
                    df_empty,
                    name=f"Detection {i}-{j} Empty",
                    face_color="blue",
                    border_color="blue",
                    **bland_point_params,
                )

    def toggle_auto_hide(self, state):
        """Toggle auto-hide mode for layers."""
        self.auto_hide_previous = state == Qt.Checked
        napari.utils.notifications.show_info(
            f"Auto-hide mode {'enabled' if self.auto_hide_previous else 'disabled'}"
        )

    def hide_layers(self):
        """Set visibility of all layers to False."""
        for layer in self.viewer.layers:
            layer.visible = False

    def auto_contrast_raw_images(self):
        """Apply automatic contrast adjustment to raw image channels."""
        adjusted_count = 0
        for layer in self.viewer.layers:
            # Only adjust Image layers, and preferably ones that look like raw channels
            if layer.__class__.__name__ == "Image":
                # Check if it's a raw channel (contains "Channel" or "(C" in name)
                if "Channel" in layer.name or "(C" in layer.name:
                    # Calculate percentile-based contrast limits
                    data = layer.data
                    p5 = np.percentile(data, 5)
                    p95 = np.percentile(data, 95)
                    layer.contrast_limits = [p5, p95]
                    adjusted_count += 1

        if adjusted_count > 0:
            napari.utils.notifications.show_info(
                f"Auto-contrast applied to {adjusted_count} raw image channel{'s' if adjusted_count != 1 else ''}"
            )
        else:
            napari.utils.notifications.show_warning(
                "No raw image channels found to adjust"
            )

    @staticmethod
    def rgb_to_hex(rgb: np.ndarray):
        """Convert (R, G, B) format to #RRGGBB."""
        return ("#{:X}{:X}{:X}").format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )

    def save_settings(self):
        """Save user specific viewing settings (e.g. contrast)."""
        self.settings = {}
        for idx, layer in enumerate(self.viewer.layers):
            layer_settings = {}
            for param in self.params:
                if hasattr(layer, param):
                    # NOTE currently doesn't get set properly via interface
                    # Bug in napari not in plugin
                    if param == "face_color":
                        value = self.rgb_to_hex(layer.face_color[0, :3])
                    elif param == "size":
                        value = layer.size[0, 0]
                    else:
                        value = getattr(layer, param)
                    layer_settings[param] = value
            self.settings[idx] = layer_settings
        napari.utils.notifications.show_info("Current settings saved.")

    def apply_settings(self):
        """Apply previously saved user specific viewing settings."""
        if not hasattr(self, "settings"):
            napari.utils.notifications.show_error("No settings saved!")
            return None

        for idx, layer in enumerate(self.viewer.layers):
            if idx in self.settings:
                for param, value in self.settings[idx].items():
                    setattr(layer, param, value)
        napari.utils.notifications.show_info("Saved settings applied.")
