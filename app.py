import sys
import json
import base64
import os
import tempfile
import math
import numpy as np
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem,
    QPushButton, QComboBox, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QDialog, QFormLayout, QDoubleSpinBox, QDialogButtonBox,
    QToolBar, QLabel, QGroupBox, QMenu, QInputDialog, QAbstractItemView
)
from PyQt6.QtGui import QPixmap, QIcon, QPen, QBrush, QAction, QPainter, QCursor, QKeySequence, QCursor, QImage, QPainter
from PyQt6.QtCore import Qt, QPointF, QByteArray, QBuffer, QSize, QRectF
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt


class MyPixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap: QPixmap, name: str = "Image"):
        super().__init__(pixmap)
        self.name = name
        self.setFlags(
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable
        )
        self.setTransformOriginPoint(self.boundingRect().center())
        self.setScale(1.0)


class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, main_window):
        super().__init__(scene)
        self.main = main_window
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)

        self.dragging_item = None
        self.drag_start_pos = None
        self.initial_rotation = 0.0
        self.initial_scale = 1.0
        self.initial_dist = 1.0
        self._pan_start = None

    def wheelEvent(self, event):
        if self.main.mode == "view":
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if self.main.mode == "view":
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._pan_start = event.pos()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return

        scene_pos = self.mapToScene(event.pos())
        item = self.itemAt(event.pos())

        if isinstance(item, MyPixmapItem):
            self.dragging_item = item
            self.drag_start_pos = scene_pos

            if self.main.current_tool == "rotate":
                self.initial_rotation = item.rotation()
            elif self.main.current_tool == "resize":
                center = item.sceneBoundingRect().center()
                self.initial_scale = item.scale()
                vec = scene_pos - center
                self.initial_dist = math.sqrt(vec.x()**2 + vec.y()**2) or 1.0

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.main.mode == "view":
            return
        if self._pan_start is not None:
            delta = self._pan_start - event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + delta.y())
            self._pan_start = event.pos()
            return

        if self.dragging_item and self.main.current_tool != "move":
            scene_pos = self.mapToScene(event.pos())
            center = self.dragging_item.sceneBoundingRect().center()

            if self.main.current_tool == "rotate":
                start_vec = self.drag_start_pos - center
                curr_vec = scene_pos - center
                angle_delta = math.degrees(
                    math.atan2(curr_vec.y(), curr_vec.x()) - math.atan2(start_vec.y(), start_vec.x())
                )
                self.dragging_item.setRotation(self.initial_rotation + angle_delta)

            elif self.main.current_tool == "resize":
                vec = scene_pos - center
                curr_dist = math.sqrt(vec.x()**2 + vec.y()**2) or 1.0
                scale_factor = curr_dist / self.initial_dist
                self.dragging_item.setScale(max(0.1, self.initial_scale * scale_factor))
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.main.mode == "view":
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            return

        if self.dragging_item:
            self.main.check_object_bounds(self.dragging_item)
            self.dragging_item = None

        super().mouseReleaseEvent(event)


class CustomSizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Собственные размеры (см)")
        layout = QFormLayout(self)
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1, 1000)
        self.width_spin.setValue(21.0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 1000)
        self.height_spin.setValue(29.7)
        layout.addRow("Ширина (см):", self.width_spin)
        layout.addRow("Высота (см):", self.height_spin)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pen Slicer")
        self.resize(1400, 900)

        self.PIXELS_PER_CM = 22
        self.current_paper = "A4"
        self.paper_width_cm = 21.0
        self.paper_height_cm = 29.7
        self.current_tool = "move"
        self.updating_selection = False
        self.clipboard = []
        self.mode = "preparation"
        self.commands = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        left_group = QGroupBox("Список объектов")
        left_layout = QVBoxLayout(left_group)
        self.object_list_widget = QListWidget()
        self.object_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.object_list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.object_list_widget.customContextMenuRequested.connect(self.show_context_menu)
        self.object_list_widget.itemSelectionChanged.connect(self.on_list_selection_changed)
        self.object_list_widget.installEventFilter(self)
        left_layout.addWidget(self.object_list_widget)
        main_layout.addWidget(left_group, 2)

        center_layout = QVBoxLayout()
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(24, 24))

        self.tool_move = QAction("Перемещение", self)
        self.tool_move.setCheckable(True)
        self.tool_move.triggered.connect(lambda: self.set_tool("move"))

        self.tool_rotate = QAction("Поворот", self)
        self.tool_rotate.setCheckable(True)
        self.tool_rotate.triggered.connect(lambda: self.set_tool("rotate"))

        self.tool_resize = QAction("Размер", self)
        self.tool_resize.setCheckable(True)
        self.tool_resize.triggered.connect(lambda: self.set_tool("resize"))

        self.toolbar.addAction(self.tool_move)
        self.toolbar.addAction(self.tool_rotate)
        self.toolbar.addAction(self.tool_resize)
        self.tool_move.setChecked(True)

        center_layout.addWidget(self.toolbar)

        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self.scene, self)
        center_layout.addWidget(self.view)

        self.fig = Figure()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.25)
        self.slider_ax = self.fig.add_axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(self.slider_ax, 'Frame', 1, 1, valinit=1, valstep=1)
        self.slider.on_changed(self.update_frame)
        center_layout.addWidget(self.canvas)
        self.canvas.hide()
        
        # Переменные для зума и перемещения
        self.canvas_pan_data = None
        self.is_panning = False
        self.canvas.mpl_connect('scroll_event', self.on_canvas_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_press)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        main_layout.addLayout(center_layout, 6)

        right_group = QGroupBox("Управление")
        right_layout = QVBoxLayout(right_group)

        size_label = QLabel("Формат листа:")
        self.paper_combo = QComboBox()
        self.paper_combo.addItems(["A3", "A4", "A5", "Custom"])
        self.paper_combo.setCurrentText("A4")
        self.paper_combo.currentTextChanged.connect(self.change_paper_size)

        self.btn_load = QPushButton("Загрузить проект")
        self.btn_save = QPushButton("Сохранить проект")
        self.btn_image = QPushButton("Загрузить изображение")
        self.btn_slice = QPushButton("Нарезать")
        self.btn_save_file = QPushButton("Сохранить файл")
        self.btn_back = QPushButton("Вернуться в подготовку")

        self.btn_load.clicked.connect(self.load_project)
        self.btn_save.clicked.connect(self.save_project)
        self.btn_image.clicked.connect(self.load_image)
        self.btn_slice.clicked.connect(self.slice_stub)
        self.btn_save_file.clicked.connect(self.save_cut_file_again)
        self.btn_back.clicked.connect(self.back_to_preparation)

        for btn in (self.btn_load, self.btn_save, self.btn_image, self.btn_slice, self.btn_save_file, self.btn_back):
            right_layout.addWidget(btn)

        self.btn_save_file.hide()
        self.btn_back.hide()

        right_layout.addWidget(size_label)
        right_layout.addWidget(self.paper_combo)
        right_layout.addStretch()
        main_layout.addWidget(right_group, 2)

        self.paper_border = None
        self.update_paper_canvas()
        self.scene.selectionChanged.connect(self.on_scene_selection_changed)

        self.setup_shortcuts()

    def prepare_lines(self):
        self.lines = []
        current_line_x = []
        current_line_y = []
        for x, y, pen in self.commands:
            if pen == 1:
                if current_line_x:
                    self.lines.append((current_line_x, current_line_y))
                current_line_x = [x]
                current_line_y = [y]
            else:
                current_line_x.append(x)
                current_line_y.append(y)
        if current_line_x:
            self.lines.append((current_line_x, current_line_y))

    def draw_frame(self, frame):
        self.ax.clear()
        
        offset_y = 1 * self.PIXELS_PER_CM
        length = (self.paper_width_cm / 4) * self.PIXELS_PER_CM
        x_start = -length / 2
        x_end = length / 2
        self.ax.plot([x_start, x_end], [offset_y, offset_y], color='black')
        
        sheet_width_px = self.paper_width_cm * self.PIXELS_PER_CM
        sheet_height_px = self.paper_height_cm * self.PIXELS_PER_CM
        
        x_min = -sheet_width_px / 2
        x_max = sheet_width_px / 2
        y_min = 0
        y_max = sheet_height_px
        
        step = 0
        for line_x, line_y in self.lines:
            if step >= frame:
                break
            num_points = min(len(line_x), frame - step)
            if num_points > 1:
                self.ax.plot(line_x[:num_points], line_y[:num_points])
            step += len(line_x)
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_title(f"Frame: {frame}")
        self.ax.set_aspect('equal')
        self.canvas.draw()

    def update_frame(self, val):
        self.draw_frame(int(val))

    def on_canvas_scroll(self, event):
        """Обработка масштабирования колёсиком мыши"""
        if event.inaxes != self.ax or self.mode != "view":
            return
        
        # Прерываем панорамирование при прокрутке
        self.is_panning = False
        
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        xdata = event.xdata
        ydata = event.ydata
        
        # Коэффициент масштабирования
        factor = 0.8 if event.button == 'up' else 1.25
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        
        self.canvas.draw_idle()

    def on_canvas_press(self, event):
        """Начало перемещения (средняя кнопка)"""
        if event.inaxes != self.ax or self.mode != "view":
            return
        if event.button == 2:  # Средняя кнопка
            self.is_panning = True
            self.canvas_pan_data = (event.xdata, event.ydata)

    def on_canvas_release(self, event):
        """Завершение перемещения"""
        self.is_panning = False
        self.canvas_pan_data = None

    def on_canvas_motion(self, event):
        """Перемещение при зажатой средней кнопке"""
        if not self.is_panning or event.inaxes != self.ax or self.mode != "view" or not self.canvas_pan_data:
            return
        
        dx = event.xdata - self.canvas_pan_data[0]
        dy = event.ydata - self.canvas_pan_data[1]
        
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        
        self.ax.set_xlim([cur_xlim[0] - dx, cur_xlim[1] - dx])
        self.ax.set_ylim([cur_ylim[0] - dy, cur_ylim[1] - dy])
        
        self.canvas_pan_data = (event.xdata, event.ydata)
        self.canvas.draw_idle()

    def setup_view_mode(self):
        self.view.hide()
        self.canvas.show()
        self.btn_save_file.show()
        self.btn_back.show()
        self.toolbar.hide()
        self.prepare_lines()
        self.slider.valmax = len(self.commands)
        if len(self.commands) > 1:
            self.slider.ax.set_xlim(1, len(self.commands))
        else:
            self.slider.ax.set_xlim(0.5, 1.5)
        self.slider.set_val(1)
        self.canvas_pan_data = None  # Сброс данных панорамирования
        self.is_panning = False  # Сброс флага панорамирования
        self.draw_frame(1)

    def back_to_preparation(self):
        self.mode = "preparation"
        self.view.show()
        self.canvas.hide()
        self.btn_save_file.hide()
        self.btn_back.hide()
        self.toolbar.show()
        self.commands = None

    def save_cut_file_again(self):
        if self.commands:
            self.save_cut_file(self.commands)

    def setup_shortcuts(self):
        for shortcut, func in [
            (QKeySequence.StandardKey.Undo, self.undo),
            (QKeySequence.StandardKey.SelectAll, self.select_all),
            (QKeySequence.StandardKey.Copy, self.copy_selected),
            (QKeySequence.StandardKey.Paste, self.paste),
        ]:
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(func)
            self.addAction(action)

    def undo(self):
        print("Undo not implemented yet")

    def select_all(self):
        self.scene.clearSelection()
        for item in self.scene.items():
            if isinstance(item, MyPixmapItem):
                item.setSelected(True)

    def copy_selected(self):
        self.clipboard = [item for item in self.scene.selectedItems() if isinstance(item, MyPixmapItem)]

    def paste(self):
        if not self.clipboard:
            return
        offset = 20
        self.scene.clearSelection()
        for original in self.clipboard:
            new_item = MyPixmapItem(original.pixmap(), original.name + " (копия)")
            new_item.setPos(original.pos() + QPointF(offset, offset))
            new_item.setRotation(original.rotation())
            new_item.setScale(original.scale())
            self.scene.addItem(new_item)
            self.add_to_object_list(new_item)
            new_item.setSelected(True)
            offset += 15
        self.update_all_bounds()

    def eventFilter(self, obj, event):
        if obj is self.object_list_widget and event.type() == event.Type.KeyPress:
            if event.key() == Qt.Key.Key_F2:
                self.rename_selected_object()
                return True
        return super().eventFilter(obj, event)

    def rename_selected_object(self):
        selected = self.object_list_widget.selectedItems()
        if not selected: return
        list_item = selected[0]
        item = list_item.data(Qt.ItemDataRole.UserRole)
        if not item: return
        new_name, ok = QInputDialog.getText(self, "Переименовать", "Новое имя:", text=item.name)
        if ok and new_name.strip():
            item.name = new_name.strip()
            list_item.setText(item.name)

    def show_context_menu(self, position):
        if not self.object_list_widget.selectedItems(): return
        menu = QMenu(self)
        menu.addAction("Переименовать (F2)", self.rename_selected_object)
        menu.addAction("Удалить", self.delete_selected)
        menu.exec(self.object_list_widget.mapToGlobal(position))

    def delete_selected(self):
        for list_item in list(self.object_list_widget.selectedItems()):
            item = list_item.data(Qt.ItemDataRole.UserRole)
            if item and item.scene():
                item.scene().removeItem(item)
            self.object_list_widget.takeItem(self.object_list_widget.row(list_item))

    def set_tool(self, tool: str):
        self.current_tool = tool
        self.tool_move.setChecked(tool == "move")
        self.tool_rotate.setChecked(tool == "rotate")
        self.tool_resize.setChecked(tool == "resize")

    def change_paper_size(self):
        size_name = self.paper_combo.currentText()
        if size_name == "Custom":
            dlg = CustomSizeDialog(self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                w = dlg.width_spin.value()
                h = dlg.height_spin.value()
            else:
                return
        else:
            sizes = {"A3": (29.7, 42.0), "A4": (21.0, 29.7), "A5": (14.8, 21.0)}
            w, h = sizes[size_name]

        self.current_paper = size_name
        self.paper_width_cm = w
        self.paper_height_cm = h
        self.update_paper_canvas()

    def update_paper_canvas(self):
        w_px = self.paper_width_cm * self.PIXELS_PER_CM
        h_px = self.paper_height_cm * self.PIXELS_PER_CM
        self.scene.setSceneRect(0, 0, w_px, h_px)

        if self.paper_border and self.paper_border.scene():
            self.scene.removeItem(self.paper_border)

        self.paper_border = QGraphicsRectItem(0, 0, w_px, h_px)
        self.paper_border.setPen(QPen(Qt.GlobalColor.black, 6))
        self.paper_border.setBrush(QBrush(Qt.GlobalColor.white))
        self.paper_border.setZValue(-10)
        self.scene.addItem(self.paper_border)
        self.update_all_bounds()

    def check_object_bounds(self, item: MyPixmapItem):
        if self.paper_border:
            item.setOpacity(1.0 if self.paper_border.rect().contains(item.sceneBoundingRect()) else 0.5)

    def update_all_bounds(self):
        for item in self.scene.items():
            if isinstance(item, MyPixmapItem):
                self.check_object_bounds(item)

    def add_to_object_list(self, item: MyPixmapItem):
        list_item = QListWidgetItem(item.name)
        thumb = item.pixmap().scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        list_item.setIcon(QIcon(thumb))
        list_item.setData(Qt.ItemDataRole.UserRole, item)
        self.object_list_widget.addItem(list_item)

    def on_list_selection_changed(self):
        if self.updating_selection: return
        self.updating_selection = True
        self.scene.clearSelection()
        for list_item in self.object_list_widget.selectedItems():
            gitem = list_item.data(Qt.ItemDataRole.UserRole)
            if gitem:
                gitem.setSelected(True)
        self.updating_selection = False

    def on_scene_selection_changed(self):
        if self.updating_selection: return
        self.updating_selection = True
        self.object_list_widget.clearSelection()
        for gitem in self.scene.selectedItems():
            for i in range(self.object_list_widget.count()):
                li = self.object_list_widget.item(i)
                if li.data(Qt.ItemDataRole.UserRole) == gitem:
                    li.setSelected(True)
                    break
        self.updating_selection = False

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить изображение", "",
                                              "Изображения (*.png *.jpg *.jpeg *.bmp *.gif)")
        if not path: return
        pixmap = QPixmap(path)
        if pixmap.isNull(): return
        name = os.path.basename(path)
        item = MyPixmapItem(pixmap, name)
        item.setPos((self.scene.width() - pixmap.width()) / 2,
                    (self.scene.height() - pixmap.height()) / 2)
        self.scene.addItem(item)
        self.add_to_object_list(item)
        self.check_object_bounds(item)

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "", "Проекты (*.json)")
        if not path: return
        items_data = []
        for item in self.scene.items():
            if isinstance(item, MyPixmapItem):
                ba = QByteArray()
                buffer = QBuffer(ba)
                buffer.open(QBuffer.OpenModeFlag.WriteOnly)
                item.pixmap().save(buffer, "PNG")
                buffer.close()
                b64 = base64.b64encode(ba.data()).decode('utf-8')
                items_data.append({
                    "name": item.name,
                    "b64": b64,
                    "pos_x": item.pos().x(),
                    "pos_y": item.pos().y(),
                    "rotation": item.rotation(),
                    "scale": item.scale()
                })
        project = {
            "paper": self.current_paper,
            "width_cm": self.paper_width_cm,
            "height_cm": self.paper_height_cm,
            "items": items_data
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(project, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "Готово", f"Проект сохранён:\n{path}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить проект", "", "Проекты (*.json)")
        if not path: return
        with open(path, "r", encoding="utf-8") as f:
            project = json.load(f)

        self.scene.clear()
        self.object_list_widget.clear()
        self.paper_border = None

        self.current_paper = project.get("paper", "A4")
        self.paper_width_cm = project.get("width_cm", 21.0)
        self.paper_height_cm = project.get("height_cm", 29.7)
        self.paper_combo.setCurrentText(self.current_paper)

        self.update_paper_canvas()

        for data in project.get("items", []):
            ba = QByteArray(base64.b64decode(data["b64"]))
            pixmap = QPixmap()
            pixmap.loadFromData(ba)
            if pixmap.isNull(): continue
            item = MyPixmapItem(pixmap, data["name"])
            item.setPos(data["pos_x"], data["pos_y"])
            item.setRotation(data.get("rotation", 0))
            item.setScale(data.get("scale", 1.0))
            self.scene.addItem(item)
            self.add_to_object_list(item)
            self.check_object_bounds(item)

    def slice_stub(self):
        if not self.paper_border:
            QMessageBox.warning(self, "Ошибка", "Лист не инициализирован")
            return

        paper_rect = self.paper_border.rect()
        w_px = int(paper_rect.width())
        h_px = int(paper_rect.height())
        
        items_to_render = []
        for item in self.scene.items():
            if isinstance(item, MyPixmapItem):
                item_rect = item.sceneBoundingRect()
                if item_rect.intersects(paper_rect) or paper_rect.contains(item_rect):
                    items_to_render.append(item)
        
        if not items_to_render:
            QMessageBox.warning(self, "Ошибка", "Нет изображений внутри листа для нарезки")
            return

        item_positions = {}
        offset_x = paper_rect.x()
        offset_y = paper_rect.y()
        for item in items_to_render:
            item_positions[item] = item.pos()
            item.setPos(item.pos() - QPointF(offset_x, offset_y))

        original_rect = self.scene.sceneRect()
        self.scene.setSceneRect(0, 0, w_px, h_px)

        original_pen = self.paper_border.pen()
        self.paper_border.setPen(QPen(Qt.PenStyle.NoPen))
        self.scene.removeItem(self.paper_border)

        image = QImage(w_px, h_px, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.scene.render(painter)
        painter.end()

        for item, pos in item_positions.items():
            item.setPos(pos)

        self.scene.addItem(self.paper_border)
        self.paper_border.setPen(original_pen)
        self.scene.setSceneRect(original_rect)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
            image.save(tmp_path, "PNG")

        try:
            dot_list = self.generate_dot_list(tmp_path)

            transformed = self.transform_to_plotter_coords(dot_list, w_px, h_px)

            self.commands = transformed
            self.mode = "view"
            self.setup_view_mode()

            QMessageBox.information(
                self, 
                "Нарезка завершена",
                f"Сгенерировано {len(transformed)} команд.\n"
                f"Нажмите 'Сохранить файл' для сохранения."
            )

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def generate_dot_list(self, image_path: str, epsilon_coeff=0.012):
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Не удалось прочитать изображение")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        full_array = []

        for cnt in contours:
            if cv2.arcLength(cnt, True) < 8:
                continue

            epsilon = epsilon_coeff * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            points = approx.reshape(-1, 2).tolist()

            if not points:
                continue

            for i, pt in enumerate(points):
                state = 1 if i == 0 else 0
                full_array.append([int(pt[0]), int(pt[1]), state])

            if points:
                first = points[0]
                full_array.append([int(first[0]), int(first[1]), 0])

        return full_array

    def transform_to_plotter_coords(self, dot_list, width_px: int, height_px: int):

        center_x = width_px // 2
        result = []

        for x, y, state in dot_list:
            new_x = x - center_x
            new_y = height_px - y
            result.append([int(new_x), int(new_y), state])

        return result

    def save_cut_file(self, dot_list):
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "Сохранить путь нарезки", 
            "", 
            "JSON (*.json);;Text (*.txt)"
        )
        if not path:
            return

        data = {
            "paper": self.current_paper,
            "width_cm": self.paper_width_cm,
            "height_cm": self.paper_height_cm,
            "pixels_per_cm": self.PIXELS_PER_CM,
            "commands": dot_list
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        txt_path = path.rsplit('.', 1)[0] + '.txt'
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"PATH FOR PLOTTER - {self.current_paper}\n")
            f.write(f"Total points: {len(dot_list)}\n\n")
            for x, y, state in dot_list:
                f.write(f"{x:6d}, {y:6d}, {state}\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())