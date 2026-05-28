import logging
from typing import List

from PySide6.QtCore import Slot, QAbstractListModel, Qt, QTimer
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QHBoxLayout, QListView

from reachx.config.app_log import get_application_logger, get_next_log_message
from reachx.data.datamanager import DataManager
from reachx.gui.baseview import BaseView


class LogView(BaseView):
    """
    The ReachX application log message view.

    This view continuously polls the application-wide logger for new messages and displays a list of the 200 most recent
    messages (most recent at the bottom). It also includes a combo box by which the user can set the log-level of the
    application-wide logger, and at the same time control which log-level messages are displayed:
     - 'DEBUG': All messages shown.
     - 'INFO': All 'DEBUG' messages are hidden.
     - 'WARNING: All 'DEBUG' and 'INFO' messages are hidden.

    Polling happens once per second, with a maximum of 5 new messages retrieved each time. Obviouslly, this is NOT
    intended for displaying thousands of log messages being generated at high rates! Message filtering is based on
    whether the message string contains one or more of the log level keywords -- so, for example an INFO message
    containing 'DEBUG' would be mistakenly filtered out!

    This view will typically be hidden by users, but it will be useful to open the view whenever something "goes wrong"
    in ReachX. It also could be helpful while testing/debugging the application. In particular, use LogView to set the
    logger level to DEBUG while testing!
    """
    _MAX_LEN: int = 200
    """ 
    Maximum number of messages displayed. Once this capacity is reached, the oldest message is discarded before
    adding a new one.
    """
    _LOG_LEVELS: List[str] = [
        logging.getLevelName(logging.DEBUG),
        logging.getLevelName(logging.INFO),
        logging.getLevelName(logging.WARNING)
    ]
    """ For setting the application logger's level and filtering out which log messages to display here. """

    def __init__(self, data_manager: DataManager) -> None:
        super().__init__('Log Messages', None, data_manager)
        self._message_log = QListView()
        """ The log messages are displayed in this view, most recent at the bottom of the list. """
        self._model = LogView._LogList()
        """ The model for our message log listing. """
        self._level_combo = QComboBox()
        """ Combo box selects the log level for filtering which messages are displayed. """

        self._message_log.setModel(self._model)
        self._message_log.setViewMode(QListView.ViewMode.ListMode)
        self._message_log.setSelectionMode(QListView.SelectionMode.NoSelection)

        # set up the combo box that selects the log level for filtering displayed messages
        self._level_combo.addItems(self._LOG_LEVELS)
        self._level_combo.setCurrentText(logging.getLevelName(get_application_logger().level))
        self._level_combo.currentTextChanged.connect(self._filter_level_changed)

        main_layout = QVBoxLayout()
        control_line = QHBoxLayout()
        control_line.addWidget(self._level_combo)
        control_line.addStretch(1)
        main_layout.addLayout(control_line)
        main_layout.addWidget(self._message_log)

        self.view_container.setLayout(main_layout)

    @Slot(str)
    def _filter_level_changed(self, current_text: str) -> None:
        get_application_logger().setLevel(logging.getLevelName(current_text))
        self._model.filter_messages_displayed(current_text)

    class _LogList(QAbstractListModel):
        """
        The model for the QListView displaying the message log. It takes care of polling the application logger
        for any new log messages and updating itself accordingly. Polling happens once a second and a max of 5 log
        messages are retrieved each time.
        """
        def __init__(self, /):
            super().__init__()
            self._messages: List[str] = list()
            """ Log message buffer, most recent message last. """
            self._filter_level: str = LogView._LOG_LEVELS[0]
            """ Log level: Filter out messages below this level: DEBUG < INFO < WARNING. """
            self._filtered: List[str] = list()
            """ Filtered version of the log message buffer. Empty and ignored if filter level is 'DEBUG'. """

            # we check for new log messages once per second
            timer = QTimer(self)
            timer.timeout.connect(self.check_for_new_log_messages)
            timer.start(1000)

        def rowCount(self, /, parent=...) -> int:
            return len(self._messages if self._filter_level == 'DEBUG' else self._filtered)

        def data(self, index, /, role=...):
            if role == Qt.ItemDataRole.DisplayRole:
                msgs = self._messages if self._filter_level == 'DEBUG' else self._filtered
                return msgs[index.row()]
            elif role == Qt.ItemDataRole.CheckStateRole:
                return None
            elif role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            return None

        def headerData(self, section: int, orientation, /, role=...):
            if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
                return "Recent log messages"
            else:
                return super().headerData(section, orientation, role)

        def filter_messages_displayed(self, level: str) -> None:
            if (self._filter_level == level) or (level not in LogView._LOG_LEVELS):
                return

            self.beginResetModel()
            self._filter_level = level
            self._refilter()
            self.endResetModel()

        def _refilter(self) -> None:
            self._filtered.clear()
            if self._filter_level == 'DEBUG':   # no filtering
                return
            if self._filter_level == 'INFO':   # filter out 'DEBUG' messages only
                self._filtered.extend([m for m in self._messages if 'DEBUG' not in m])
            else:
                self._filtered.extend([m for m in self._messages if ('DEBUG' not in m) and ('INFO' not in m)])

        @Slot()
        def check_for_new_log_messages(self) -> None:
            msgs_to_add: List[str] = list()

            i = 5   # no more than this many messages added at one time
            while i > 0:
                msg = get_next_log_message()
                if msg is None:
                    break
                msgs_to_add.append(msg)
                i -= 1
            if len(msgs_to_add) == 0:
                return

            self.beginResetModel()
            while len(self._messages) + len(msgs_to_add) > LogView._MAX_LEN:
                self._messages.pop(0)
            for m in msgs_to_add:
                self._messages.append(m)
            if self._filter_level != 'DEBUG':
                self._refilter()
            self.endResetModel()
