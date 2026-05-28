from typing import Optional

from PySide6.QtCore import QSize, QObject
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QWidget, QDockWidget

from reachx.data.datamanager import DataManager


class BaseView(QObject):
    """
    A base class defining functionality common to all application view widgets as well as Qt-style signals that can be
    used for inter-view communications.
    """

    def __init__(self, title: str, background: Optional[QColor], data_manager: DataManager):
        """
        Create an empty view with the specified title.

        :param title: The view's title.
        :param background: An alternative background color (intended for debug use when testing view layout).
        :param data_manager: The source for all recorded video data and analysis results presented in any view.
        """
        super().__init__(parent=None)
        self.view_container = QWidget()
        """ The widget that contains this view. """
        self._title = title
        """ The view's title. """
        self.data_manager = data_manager
        """ View queries the application data manager for whatever data or analysis results it needs. """

        if isinstance(background, QColor):
            self.view_container.setAutoFillBackground(True)
            palette = self.view_container.palette()
            palette.setColor(QPalette.ColorRole.Window, background)
            self.view_container.setPalette(palette)
        self.view_container.setContentsMargins(0, 0, 0, 0)
        self.view_container.setMinimumSize(QSize(100, 100))
        self.view_container.setWindowTitle(title)

    @property
    def title(self) -> str:
        """ The view's title. """
        return self._title

    @property
    def is_parent_dock_hidden(self) -> bool:
        """
        True if the docking widget containing this view is currently hidden. If the view is not contained in a
        docking widget, it is assumed to be hidden.
        """
        widget = self.view_container.parentWidget()
        if isinstance(widget, QDockWidget):
            return widget.isHidden()
        return True

    @property
    def container(self) -> QWidget:
        """ The widget container for this view. """
        return self.view_container
