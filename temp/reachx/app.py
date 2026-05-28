"""
app.py - The main entry point for the application.

The Reach application is launched without any arguments: 'python -m reachx.app'.
"""
import sys

from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
import pyqtgraph as pg

from reachx.uicommon import RxIcons
from reachx.config.app_log import get_application_logger
from reachx.data.datamanager import DataManager
from reachx.gui.viewmanager import ViewManager


class ReachMainWindow(QMainWindow):
    """
    The main application window. The Reach UI is built and controlled by the ViewManager singleton. This is merely the
    container for the UI.
    """

    def __init__(self, mgr: DataManager):
        super().__init__()
        self._first_shown = True
        self._data_manager = mgr
        self._data_manager.set_main_window(self)
        self._view_manager = ViewManager(self, mgr)
        """ The model-view controller. Constructs the UI and the data model and hooks them together. """

    def closeEvent(self, event: QCloseEvent) -> None:
        """ [QWidget override] Closing the main window quits the application -- unless the user vetoes the quit. """
        self._view_manager.quit()
        event.ignore()

    def showEvent(self, event: QShowEvent):
        """ [QWidget override] Need to notify DataManager when main application window is first shown. """
        super().showEvent(event)  # Call the base class method
        if self._first_shown:
            self._first_shown = False
            self._data_manager.on_main_window_first_shown()


if __name__ == "__main__":
    # start logging as early as possible
    get_application_logger()

    main_app = QApplication(sys.argv)  # any additional command-line arguments are ignored

    # some visual issues with native Windows appearance in dark mode, so we use platform-agnostic 'Fusion' style
    if sys.platform.startswith("win"):
        main_app.setStyle("Fusion")

    # change certain PyQtGraph global config options. Images are typically packed into Numpy array in row-major
    # order, but the default in PyQtGraph is column-major order. Turn off antialising if performance is an issue.
    pg.setConfigOptions(antialias=True, imageAxisOrder="row-major")

    RxIcons.load_icons_and_cursors()

    # before raising application window, create the data manager and load configuration. This must succeed.
    data_mgr = DataManager()
    cfg_error = data_mgr.on_startup()
    if cfg_error is None:
        main_window = ReachMainWindow(mgr=data_mgr)
        get_application_logger().debug("About to show main window.")
        main_window.show()
        get_application_logger().debug("Starting event loop.")
        exit_code = main_app.exec()
        get_application_logger().info(f"Exited with exit code {exit_code}")
    else:
        msg_box = QMessageBox(QMessageBox.Icon.Critical, "Error",
                              f"Unable to find/load configuration files:\n   {cfg_error}\n\nExiting.",
                              buttons=QMessageBox.StandardButton.Ok)
        msg_box.exec()
        exit_code = 1

    # Any after-exit tasks can go here (should not take too long!)
    sys.exit(exit_code)
