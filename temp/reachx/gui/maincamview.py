from typing import List

from PySide6.QtCore import Slot, Qt, Signal
from PySide6.QtGui import QMouseEvent, QShortcut, QKeySequence
from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QFrame, QGridLayout, QCheckBox, \
    QComboBox, QSizePolicy, QStackedWidget, QWidget

from reachx.data.datamanager import DataManager
from reachx.common import RX, RxCam, RxEvent
from reachx.uicommon import RxIcons
from reachx.gui.baseview import BaseView
from reachx.gui.camframe import CamFrame
from reachx.gui.timeline import TimeLineWidget

_RX = RX()
""" Application-wide constants. """


class _ClickableQLabel(QLabel):
    """ A QLabel that emits a signal when the mouse is pressed within its bounds. """
    clicked = Signal()
    """ Signal indicates that the mouse was pressed when the cursor was over this label. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def mousePressEvent(self, event: QMouseEvent):
        self.clicked.emit()
        super().mousePressEvent(event)


class MainCamView(BaseView):
    """
    The main cameras view, displaying current frames from the two primary rig camera videos: ``RxCam.SIDE/FRONT`` for
    a legcay-cam session, and ``RxCam.LEFT/RIGHT`` for a fixed-cam session.

    This view serves as the "central widget" in the main application window. It is always present and cannot be
    hidden by the user. In addition to housing the widgets displaying the current frames for the two primary cameras,
    it includes a number of widgets for playing the video or navigating to any particular frame:

     - Immediately below the two camera frame widgets is a custom PyQtGraph ``PlotWidget`` that displays any recorded
       events in a selected timespan on either side of the "current frame". In particular, it allows user to fine-tune
       the current frame number by simply clicking anywhere on the widget. For details, see ``TimeLineWidget`` class.
     - Push buttons controlling playback: increase/decrease playback speed, and stop. A readout label reflects the
       current playback speed in FPS.
     - Push buttons for stepping forward or backward in the video timeline IAW a step size selected via combo box.
     - Another readout label shows the current frame number. Click it to toggle the readout between elapsed time in
       the format "min:sec.ms" or the integer frame number.
     - A slider lets the user quickly pan through the entire video.
     - A pair of buttons let the user advance/rewind to the next/previous "reach epoch" (defined as the next/previous
       pellet delivery event after/before the current frame).
     - Additional widgets let the user choose the time span of the timeline widget, whether that widget's
       legend is shown, and whether the widget is hidden altogether.

    A number of application-level shortcuts are defined here because the related UI widgets are housed here:
     - Right-arrow = increase playback speed, Left-arrow = decrease playback speed, and space-bar = stop.
     - Up-arrow = go to previous reach epoch, down-arrow = next reach epoch.
     - Right-bracket = step forward, left-bracket = step backward.
     - Backslash = Increment step size selection (with wrap-around back to 1).
    """

    STEP_SIZES: List[int] = [1, 5, 10, 50, 100, 200]
    """ Supported step sizes (in #frames) for stepping forward or backward through video. """

    def __init__(self, data_manager: DataManager) -> None:
        super().__init__('Side/Front Cams', None, data_manager)
        self._side_cam = CamFrame(RxCam.SIDE, data_manager)
        """ The side camera frame container (legacy-cam sessions only). """
        self._front_cam = CamFrame(RxCam.FRONT, data_manager)
        """ The front camera frame container (legacy-cam sessons only). """
        self._left_cam = CamFrame(RxCam.LEFT, data_manager)
        """ The left camera frame container (fixed-cam sessions only). """
        self._right_cam = CamFrame(RxCam.RIGHT, data_manager)
        """ The right camera frame container (fixed-cam sessions only). """
        self._left_stack = QStackedWidget()
        """ Contains the side and left camera frame containers, only one of which is visible at a time. """
        self._right_stack = QStackedWidget()
        """ Contains the front and right camera frame containers, only one of which is visible at a time. """""
        self._time_line = TimeLineWidget(data_manager)
        """ Custom widget displaying the event timeline on either side of the current frame. """
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        """ Slider selects the current frame number for display. """
        self._curr_frame_readout = _ClickableQLabel(f"0000")
        """ A label reflecting the frame number or elapsed time corresponding to the frame on display. """
        self._show_frame_number = True
        """ True if readout currently displays frame number, False if it displays frame time as 'MMM:ss.mmm' """
        self._fps_readout = QLabel()
        """ A label reflecting the current playback speed in frames per second. """
        self._inc_playspeed_btn = QPushButton(RxIcons.FORWARD, "")
        """ Press this button to increase forward playback speed (or reduce reverse playback speed). """
        self._dec_playspeed_btn = QPushButton(RxIcons.REVERSE, "")
        """ Press this button increase reverse playback speed (or reduce forward playback speed). """
        self._stop_btn = QPushButton(RxIcons.STOP, "")
        """ Press this button stop playback in progress. """

        self._step_fwd_btn = QPushButton(RxIcons.STEP_FWD, "")
        """ Press this button to step forward in video by a set number of frames -- the "step size". """
        self._step_back_btn = QPushButton(RxIcons.STEP_BACK, "")
        """ Press this button to step backward in vieewo by a set number of frames -- the "step size. """
        self._step_size_combo = QComboBox()
        """ Combo box selects the current step size in # of video frames. """

        self._next_reach_btn = QPushButton(RxIcons.NEXT, "")
        """ Advances to the start of the next reach epoch. """
        self._prev_reach_btn = QPushButton(RxIcons.PREV, "")
        """ Rewinds to the start of the previous reach epoch. """
        self._reach_start_evt_cb = QComboBox()
        """ Combo box selects which ``RxEvent`` defines the start of a reach epoch. """
        self._timeline_show_cb = QCheckBox("Show")
        """ If checked, the timeline widget is visible; otherwise it is hidden. """
        self._timeline_legend_cb = QCheckBox("Legend")
        """ If checked, timeline event legend is shown; otherwise it is hidden. """
        self._timeline_span_combo = QComboBox()
        """ Combo box selects the span for the timeline widget (in # of frames). """
        self._show_marker_labels_cb = QCheckBox("Show marker labels")
        """ If checked, any body part markers on the current frame are labeled in both cam frames. """
        self._predictions_cb = QCheckBox("Predictions")
        """ If checked, predicted body part locations (if available) are rendered in both cam frames."""
        self._hand_pellet_cb = QCheckBox("Hand/pellet only")
        """ If checked, only predicted hand and pellet locations are rendered, else all available body parts. """

        self._frame_slider.setTickPosition(QSlider.TickPosition.NoTicks)
        self._frame_slider.setRange(0, 1)
        self._frame_slider.setSliderPosition(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_value_changed)

        self._curr_frame_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        self._curr_frame_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self._curr_frame_readout.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._curr_frame_readout.clicked.connect(self._on_toggle_frame_readout_style)

        self._fps_readout.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        self._fps_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self._fps_readout.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._inc_playspeed_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._inc_playspeed_btn.setStyleSheet("QPushButton { border: none; }")
        self._inc_playspeed_btn.setToolTip("Increment playback speed (shortcut: \u2192 )")
        self._inc_playspeed_btn.clicked.connect(self._increment_playback_speed)
        self._dec_playspeed_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._dec_playspeed_btn.setStyleSheet("QPushButton { border: none; }")
        self._dec_playspeed_btn.setToolTip("Decrement playback speed (shortcut: \u2190 )")
        self._dec_playspeed_btn.clicked.connect(self._decrement_playback_speed)
        self._stop_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._stop_btn.setStyleSheet("QPushButton { border: none; }")
        self._stop_btn.setToolTip("Stop (shortcut: spacebar )")
        self._stop_btn.clicked.connect(self._stop_playback)

        self._step_fwd_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._step_fwd_btn.setStyleSheet("QPushButton { border: none; }")
        self._step_fwd_btn.setToolTip("Step forward (shortcut: ']' )")
        self._step_fwd_btn.clicked.connect(self._step_forward)
        self._step_back_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._step_back_btn.setStyleSheet("QPushButton { border: none; }")
        self._step_back_btn.setToolTip("Step backward (shortcut: '[' )")
        self._step_back_btn.clicked.connect(self._step_backward)

        # set up the combo box that selects the frame step size
        self._step_size_combo.addItems([str(k) for k in self.STEP_SIZES])
        self._step_size_combo.setCurrentText(str(self.STEP_SIZES[0]))
        self._step_size_combo.setToolTip("Step size in #frames (shortcut: '\\' )")

        self._next_reach_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._next_reach_btn.setStyleSheet("QPushButton { border: none; }")
        self._next_reach_btn.setToolTip("Advance to pellet delivery for next reach epoch (shortcut: \u2191 )")
        self._next_reach_btn.clicked.connect(self._go_to_next_reach)
        self._prev_reach_btn.setIconSize(RxIcons.ICON_FIXED_SIZE)
        self._prev_reach_btn.setStyleSheet("QPushButton { border: none; }")
        self._prev_reach_btn.setToolTip("Rewind to pellet delivery for previous reach epoch (shortcut: \u2193 )")
        self._prev_reach_btn.clicked.connect(self._go_to_previous_reach)

        self._reach_start_evt_cb.addItems([str(e) for e in [RxEvent.T6000, RxEvent.DETECTED,
                                                            RxEvent.DELIVERY, RxEvent.T5000]])
        self._reach_start_evt_cb.setCurrentText(str(RxEvent.DELIVERY))
        self._reach_start_evt_cb.setToolTip("Choose what event marks the start of a 'reach epoch'")

        # application-level shortcut keys: R/L arrows for forward/reverse playback, spacebar to stop; up/down arrows
        # for next/previous reach epoch; R/L brackets for stepping forward/backward
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._increment_playback_speed)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._decrement_playback_speed)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._stop_playback)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self, self._step_forward)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self, self._step_backward)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backslash), self, self._increment_step_size)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Up), self, self._go_to_next_reach)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Down), self, self._go_to_previous_reach)
        shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)

        self._timeline_show_cb.setChecked(True)
        self._timeline_show_cb.checkStateChanged.connect(self._on_toggle_timeline_shown)
        self._timeline_legend_cb.checkStateChanged.connect(self._on_toggle_timeline_legend)

        # set up the combo box that selects the timeline span (note we have to convert to/from str)
        self._timeline_span_combo.addItems([str(k) for k in TimeLineWidget.TIME_SPANS])
        self._timeline_span_combo.setCurrentText(str(self._time_line.timespan))
        self._timeline_span_combo.currentTextChanged.connect(self._on_timeline_span_changed)
        self._timeline_span_combo.setToolTip("Timeline span in frames")

        # CamFrames are initially configured to hide marker labels but show all predicted body part locations...
        self._show_marker_labels_cb.setChecked(False)
        self._show_marker_labels_cb.checkStateChanged.connect(self._on_toggle_marker_labels)
        self._predictions_cb.setChecked(True)
        self._predictions_cb.checkStateChanged.connect(self._on_toggle_predicted_locations)
        self._predictions_cb.setToolTip("If checked, available predicted body part locations are shown")
        self._hand_pellet_cb.setChecked(False)
        self._hand_pellet_cb.checkStateChanged.connect(self._on_toggle_predicted_locations)
        self._hand_pellet_cb.setToolTip("If checked, only display predicted locations for hand(s) and pellet.")

        # the side and front cams are at index 0 of the stacked widgets, left and right at index 1
        self._left_stack.addWidget(self._create_centered_wrappper_for(QLabel("Side Camera"), self._side_cam))
        self._left_stack.addWidget(self._create_centered_wrappper_for(QLabel("Left Camera"), self._left_cam))
        self._left_stack.setCurrentIndex(0)
        self._right_stack.addWidget(self._create_centered_wrappper_for(QLabel("Front Camera"), self._front_cam))
        self._right_stack.addWidget(self._create_centered_wrappper_for(QLabel("Right Camera"), self._right_cam))
        self._right_stack.setCurrentIndex(0)

        main_layout = QVBoxLayout()
        cam_grid = QGridLayout()
        cam_grid.addWidget(self._left_stack, 0, 1, alignment=Qt.AlignmentFlag.AlignTop)
        cam_grid.addWidget(self._right_stack, 0, 3, alignment=Qt.AlignmentFlag.AlignTop)
        cam_grid.setColumnMinimumWidth(2, 5)
        cam_grid.setColumnStretch(0, 1)
        cam_grid.setColumnStretch(4, 1)
        cam_grid.addWidget(self._time_line, 1, 0, 1, 5)
        main_layout.addLayout(cam_grid)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(2)
        main_layout.addWidget(separator)

        control_line = QHBoxLayout()
        buttons = QHBoxLayout()
        buttons.setSpacing(1)
        buttons.addWidget(self._dec_playspeed_btn)
        buttons.addWidget(self._stop_btn)
        buttons.addWidget(self._inc_playspeed_btn)
        buttons.addWidget(self._fps_readout)
        control_line.addLayout(buttons)
        control_line.addWidget(self._frame_slider, stretch=1)
        control_line.addWidget(self._curr_frame_readout)

        buttons = QHBoxLayout()
        buttons.setSpacing(1)
        buttons.addWidget(self._step_back_btn)
        buttons.addWidget(self._step_size_combo)
        buttons.addWidget(self._step_fwd_btn)
        control_line.addLayout(buttons)
        buttons.addSpacing(5)

        main_layout.addLayout(control_line)

        control_line = QHBoxLayout()
        control_line.addWidget(QLabel("Timeline:"))
        control_line.addWidget(self._timeline_show_cb)
        control_line.addWidget(self._timeline_legend_cb)
        control_line.addWidget(self._timeline_span_combo)
        control_line.addSpacing(30)
        control_line.addWidget(self._show_marker_labels_cb)
        control_line.addWidget(self._predictions_cb)
        control_line.addWidget(self._hand_pellet_cb)
        control_line.addStretch(1)
        buttons = QHBoxLayout()
        buttons.setSpacing(1)
        buttons.addWidget(QLabel("Reach Epochs:"))
        buttons.addWidget(self._reach_start_evt_cb)
        buttons.addWidget(self._prev_reach_btn)
        buttons.addWidget(self._next_reach_btn)
        control_line.addLayout(buttons)
        main_layout.addLayout(control_line)

        main_layout.addStretch(1)

        self.view_container.setLayout(main_layout)

        self._reset()

        self.data_manager.active_workspace_switched.connect(self._reset)
        self.data_manager.active_workspace_path_changed.connect(self._reset)
        self.data_manager.session_loaded.connect(self._reset)
        self.data_manager.session_manager.frame_ready.connect(self._refresh)
        self.data_manager.session_manager.playback_started.connect(self._refresh)
        self.data_manager.session_manager.playback_stopped.connect(self._refresh)

    @staticmethod
    def _create_centered_wrappper_for(label: QLabel, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QGridLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(label, 0, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(widget, 1, 0, Qt.AlignmentFlag.AlignCenter)
        container.setLayout(layout)
        return container

    def _reset(self) -> None:
        # must update the cam stacks if we've switched from a fixed-cam to a legacy-cam workspace or vice versa
        if self._left_stack.currentIndex() == 0 and self.data_manager.active_workspace.is_fixed_cam:
            self._left_stack.setCurrentIndex(1)
            self._right_stack.setCurrentIndex(1)
        elif self._left_stack.currentIndex() == 1 and not self.data_manager.active_workspace.is_fixed_cam:
            self._left_stack.setCurrentIndex(0)
            self._right_stack.setCurrentIndex(0)

        self._frame_slider.valueChanged.disconnect(self._on_frame_slider_value_changed)
        n = self.data_manager.session_manager.total_frames
        curr = self.data_manager.session_manager.current_frame_num if n > 0 else 0
        self._frame_slider.setRange(0, n-1 if n > 0 else 0)
        self._frame_slider.setSliderPosition(curr)
        self._frame_slider.setEnabled(n > 0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_value_changed)

        self._refresh()

    def _refresh(self) -> None:
        n = int(self.data_manager.session_manager.total_frames)
        curr = int(self.data_manager.session_manager.current_frame_num) if n > 0 else 0

        self._inc_playspeed_btn.setEnabled(n > 0 and curr < n - 1)
        self._dec_playspeed_btn.setEnabled(n > 0 and curr > 0)
        self._stop_btn.setEnabled(self.data_manager.session_manager.playback_in_progress)

        self._step_fwd_btn.setEnabled(n > 0 and curr < n - 1)
        self._step_back_btn.setEnabled(n > 0 and curr > 0)

        self._frame_slider.setEnabled(not self.data_manager.session_manager.playback_in_progress)
        if curr != self._frame_slider.sliderPosition():
            self._frame_slider.setSliderPosition(curr)

        events = self.data_manager.session_manager.events
        if events is not None:
            self._next_reach_btn.setEnabled(events.find_next_occurrence(self.reach_epoch_start_event, curr) > -1)
            self._prev_reach_btn.setEnabled(events.find_prev_occurrence(self.reach_epoch_start_event, curr) > -1)
        else:
            self._next_reach_btn.setEnabled(False)
            self._prev_reach_btn.setEnabled(False)

        self._refresh_readout()

        speed = self.data_manager.session_manager.current_playback_speed
        self._fps_readout.setText("[stopped]" if speed == 0 else f"{speed:04d} fps")

    def _refresh_readout(self) -> None:
        n = self.data_manager.session_manager.total_frames
        curr = self.data_manager.session_manager.current_frame_num if n > 0 else 0

        if self._show_frame_number:
            n_digits = len(str(n)) if n > 0 else 4
            self._curr_frame_readout.setText(f"{curr:0{n_digits}d}")
        else:
            t = max(0.0, self.data_manager.session_manager.current_elapsed_time)
            self._curr_frame_readout.setText(_RX.format_elapsed_time(t, in_hours=False))

    def _on_toggle_frame_readout_style(self) -> None:
        """
        Toggle the style of the readout label displaying either the frame number in a sunken panel or the
        elapsed time in a raised panel.
        """
        if self._show_frame_number:
            self._show_frame_number = False
            self._curr_frame_readout.setFrameStyle(QFrame.Shadow.Raised | QFrame.Shape.Panel)
        else:
            self._show_frame_number = True
            self._curr_frame_readout.setFrameStyle(QFrame.Shadow.Sunken | QFrame.Shape.Panel)
        self._refresh_readout()

    @Slot()
    def _on_frame_slider_value_changed(self) -> None:
        """ Handler responding to value changes in the slider controlling the current frame number. """
        frame = self._frame_slider.sliderPosition()
        self.data_manager.session_manager.go_to_frame(frame)

    @Slot()
    def _increment_playback_speed(self) -> None:
        self.data_manager.session_manager.start_or_adjust_playback()

    @Slot()
    def _decrement_playback_speed(self) -> None:
        self.data_manager.session_manager.start_or_adjust_playback(reduce_speed=True)

    @Slot()
    def _stop_playback(self) -> None:
        self.data_manager.session_manager.stop_playback()

    @Slot()
    def _step_forward(self) -> None:
        step = int(self._step_size_combo.currentText())
        self.data_manager.session_manager.stop_playback()
        frame = self.data_manager.session_manager.current_frame_num + step
        self.data_manager.session_manager.go_to_frame(frame)

    @Slot()
    def _step_backward(self) -> None:
        step = int(self._step_size_combo.currentText())
        self.data_manager.session_manager.stop_playback()
        frame = self.data_manager.session_manager.current_frame_num - step
        self.data_manager.session_manager.go_to_frame(frame)

    @Slot()
    def _increment_step_size(self) -> None:
        idx, n = self._step_size_combo.currentIndex(), self._step_size_combo.count()
        self._step_size_combo.setCurrentIndex((idx + 1) % n)

    @property
    def reach_epoch_start_event(self) -> RxEvent:
        """ The event type that marks the start of a reach epoch, as selected by user via a combo box in this view. """
        evt_str = self._reach_start_evt_cb.currentText()
        try:
            evt = RxEvent(evt_str)
            return evt
        except ValueError:
            return RxEvent.DELIVERY

    @Slot()
    def _go_to_next_reach(self) -> None:
        events = self.data_manager.session_manager.events
        curr = self.data_manager.session_manager.current_frame_num
        if events is not None:
            frame = self.data_manager.session_manager.events.find_next_occurrence(self.reach_epoch_start_event, curr)
            if frame > -1:
                self.data_manager.session_manager.go_to_frame(frame)

    @Slot()
    def _go_to_previous_reach(self) -> None:
        events = self.data_manager.session_manager.events
        curr = self.data_manager.session_manager.current_frame_num
        if events is not None:
            frame = self.data_manager.session_manager.events.find_prev_occurrence(self.reach_epoch_start_event, curr)
            if frame > -1:
                self.data_manager.session_manager.go_to_frame(frame)

    @Slot(Qt.CheckState)
    def _on_toggle_timeline_shown(self, check_state: Qt.CheckState) -> None:
        """ Handler shows/hides the timeline widget. """
        self._time_line.setVisible(check_state == Qt.CheckState.Checked)

    @Slot(Qt.CheckState)
    def _on_toggle_timeline_legend(self, check_state: Qt.CheckState) -> None:
        """ Handler toggles the timeline legend on/off. """
        self._time_line.show_legend = (check_state == Qt.CheckState.Checked)

    @Slot(str)
    def _on_timeline_span_changed(self, span: str) -> None:
        """ Handler update the timepan for the timeline widget. """
        if isinstance(span, str) and span.isdigit():
            self._time_line.change_timespan(int(span))

    @Slot(Qt.CheckState)
    def _on_toggle_marker_labels(self, check_state: Qt.CheckState) -> None:
        """ Handler toggles the visibility of the marker labels on the ``CamFrame`` widgets. """
        show = check_state == Qt.CheckState.Checked
        for cam_frame in [self._side_cam, self._front_cam, self._left_cam, self._right_cam]:
            cam_frame.show_hide_marker_labels(show)

    @Slot(Qt.CheckState)
    def _on_toggle_predicted_locations(self, _: Qt.CheckState) -> None:
        """ Handler updates visibility of predicted body part locations on the ``CamFrame`` widgets. """
        show = self._predictions_cb.isChecked()
        hand_pellet_only = self._hand_pellet_cb.isChecked()
        for cam_frame in [self._side_cam, self._front_cam, self._left_cam, self._right_cam]:
            cam_frame.show_predicted_markers(show, hand_pellet_only)
