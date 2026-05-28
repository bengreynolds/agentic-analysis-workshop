from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QTabWidget

from reachx.common import RX, RxCam
from reachx.data.datamanager import DataManager
from reachx.gui.baseview import BaseView
from reachx.gui.camframe import CamFrame

_RX = RX()
""" Application-wide constants. """


class OtherCamView(BaseView):
    """
    This view displays the current video frame on the ``RxCam.STIM`` and ``RxCam.FAST`` cameras, which may or may not be
    recorded during a ReachX legacy-cam experiment session. They are not part of the newer fixed-cam setups.

    The view is for display purposes only (no user interactions implemented), and the camera frame widgets will
    simply display "No video available" if the corresponding camera was not recorded.

    The `CamFrame` widgets handle all the important functionality. This is merely a container for those widgets.
    """

    def __init__(self, data_manager: DataManager) -> None:
        super().__init__('Additional Cameras', None, data_manager)
        self._tab_widget = QTabWidget()
        """ The additional camera frames are housed as individual tabs in this tab panel. """
        self._no_cams_message = QLabel("No additional cameras were recorded.")
        """ Message displayed when neither cam is available. """

        self._tab_widget.addTab(CamFrame(RxCam.STIM, data_manager, False), "Stim Camera")
        self._tab_widget.addTab(CamFrame(RxCam.FAST, data_manager, False), "Fast Camera")

        self._no_cams_message.setStyleSheet("font-weight: bold;")

        main_layout = QVBoxLayout()
        main_layout.addWidget(self._no_cams_message, alignment=Qt.AlignmentFlag.AlignHCenter)
        main_layout.addWidget(self._tab_widget)
        main_layout.addStretch(1)

        self.view_container.setLayout(main_layout)

        self._reset()

        self.data_manager.active_workspace_switched.connect(self._reset)
        self.data_manager.session_loaded.connect(self._reset)

    def _reset(self) -> None:
        """
        Whenever the active workspace changes or a different experiment session is loaded, this method will show/hide
        the message which indicate that no additional camera sources are available.
        """
        cam_ids = self.data_manager.session_manager.available_camera_ids
        available = (RxCam.STIM in cam_ids) or (RxCam.FAST in cam_ids)
        self._no_cams_message.setVisible(not available)
