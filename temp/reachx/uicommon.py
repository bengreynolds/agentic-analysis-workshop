"""
uicommon.py: ReachX user interface-related utility classes.

These were relocated from ``common.py`` so that module can be imported standalone, with no dependencies on any other
ReachX module nor the Pyside UI framework.
"""

from __future__ import annotations

from importlib import resources as impresources
from typing import Optional

from PySide6.QtCore import QSize, QObject, Signal
from PySide6.QtGui import QIcon, QCursor, QPixmap

from reachx import assets as reach_assets


class RxIcons:
    """
    Container for ReachX application icons and custom cursors

    Usage:
     - All icon and cursor images are found in the `assets` folder.
     - Designed as a singleton that is not really used. All icons and cursors are available as class members.
     - Be sure to call `load_icons_and_cursors()` during startup, after the application object has been created.
    """
    STOP: QIcon
    """ Stop playback icon. """
    FORWARD: QIcon
    """ Forward playback icon. """
    REVERSE: QIcon
    """ Reverse playback icon. """
    NEXT: QIcon
    """ Go to next event icon. """
    PREV: QIcon
    """ Go to previous event icon. """
    TRASH: QIcon
    """ A generic delete object operation icon. """
    NEW: QIcon
    """ A generic create new operation icon. """
    STEP_FWD: QIcon
    """ Step forward in video by a set number of frames. """
    STEP_BACK: QIcon
    """ Step backward in video by a set number of frames. """
    ICON_FIXED_SIZE = QSize(24, 24)
    """ Use this size as a rule for icon size. The source images may be larger but are auto-scaled by QIcon. """

    DELETE_CURSOR: QCursor
    """ Custom cursor indicating a "delete" action. """
    MARK_CURSOR: QCursor
    """ Custom cursor indicating a "marking" action. """

    _instance: Optional[RxIcons] = None
    """ The singleton instance. """

    def __new__(cls, *args, **kwargs):
        """ Overridden to enforce singleton instance. """
        if cls._instance is None:
            print("Creating instance!")
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        pass

    # noinspection PyTypeChecker
    @staticmethod
    def load_icons_and_cursors() -> None:
        try:

            with impresources.path(reach_assets, "stop_playback.png") as p:
                RxIcons.STOP = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "forward_playback.png") as p:
                RxIcons.FORWARD = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "reverse_playback.png") as p:
                RxIcons.REVERSE = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "next.png") as p:
                RxIcons.NEXT = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "prev.png") as p:
                RxIcons.PREV = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "trash.png") as p:
                RxIcons.TRASH = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "new_add.png") as p:
                RxIcons.NEW = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "step_fwd.png") as p:
                RxIcons.STEP_FWD = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "step_back.png") as p:
                RxIcons.STEP_BACK = QIcon(str(p.absolute()))
            with impresources.path(reach_assets, "delete_cursor.png") as p:
                pixmap = QPixmap(p)
                RxIcons.DELETE_CURSOR = QCursor(pixmap, pixmap.width() // 2, pixmap.height() // 2)
            with impresources.path(reach_assets, "pin_cursor.png") as p:
                pixmap = QPixmap(p)
                RxIcons.MARK_CURSOR = QCursor(pixmap, pixmap.width() // 2, pixmap.height() // 2)
        except Exception as e:
            print(f"Error occurred while loading application icons: {e}")


class BackgroundTask(QObject):
    """
    A cancelable task running on a background thread and delivering progress updates and log messages using
    PySide signals. Intended as the base class for any time-consuming task object that runs in a separate background
    thread in ReachX.

    USAGE: Extend this class and override the `run()` method to perform the actual task. Use the signals defined here
    to deliver short messages, indicate progress, and notify when the task is done. While performing the task, check
    `was_canceled()` frequently in case the task was canceled on the GUI thread.

    The task object must be moved to the background thread before invoking `run()`.
    """

    progress_updated: Signal = Signal(int)
    """ Task progress updated. Argument is integer completion percentage in [0..100]. """
    message_logged: Signal = Signal(str)
    """ An informational progress or error message is logged. Argument is a non-empty string. """
    data_ready: Signal = Signal(object)
    """ Data object ready for consumption. It is up to the receiver to determine the type of data object. """
    finished: Signal = Signal()
    """ Task has finished."""

    def __init__(self) -> None:
        super().__init__()
        self._canceled = False
        """ Flag set if task was canceled. """

    def run(self) -> None:
        """
        Perform this task. This is the method that should be invoked on the background thread when that thread starts.
        This method is essentially an empty placeholder and must be overridden.
        """
        self.message_logged.emit("Starting up...")
        # THE INTERESTING STUFF GOES HERE
        if not self._canceled:
            self.message_logged.emit("Done!")
        self.finished.emit()

    def cancel(self) -> None:
        """ Cancel this task. Once canceled, further invocations of this method have no effect. """
        self._canceled = True

    def was_canceled(self) -> bool:
        return self._canceled
