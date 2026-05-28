from importlib import resources as impresources
from typing import Dict

from PySide6.QtCore import Slot
from PySide6.QtWidgets import QTextBrowser, QComboBox, QVBoxLayout, QHBoxLayout, QLabel

import reachx.assets as reach_assets
from reachx.config.app_log import get_application_logger
from reachx.data.datamanager import DataManager
from reachx.gui.baseview import BaseView


class HelpView(BaseView):
    """
    A read-only "help" view that displays a concise user guide for ReachX. The user guide is inside a view class so the
    user can hide/dock/float the guide like most other application views.

    The user guide content is found in a number of markdown files in the `assets` folder. Each file corresponds to a
    different section of the guide. The view itself includes a dropdown box for selecting a section, and a text browser
    element for displaying the contents of the corresponding markdown file.

    DEVNOTE: It's a lot easier to write the user guide pages in markdown rather than HTML. However, QTextBrowser does
    not give us a lot of control over the internal translation of the markdown text to HTML. Had to play around with
    the default stylesheet to get a reasonable appearance.
    """

    SECTIONS: Dict[str, str] = {
        'Overview': 'overview.md', 'Fixed/Legacy-Cam Support': 'fixed_v_legacy.md',
        'Configuration': 'config.md', 'User Interface': 'ui.md',
        'Reviewing Video': 'video_playback.md', 'Modeling': 'modeling.md', 'Curating Reaches': 'curation.md',
        'Trajectory Analysis': 'traj_analysis.md', 'Session Files': 'sessionfiles.md',
        'Keyboard Shortcuts': 'shortcuts.md'
    }
    """ The individual section markdown filenames, keyed by the user-facing section name. """
    DEF_SECTION: str = 'Overview'
    """ At startup this section of the user guide is displayed. """

    def __init__(self, data_manager: DataManager) -> None:
        super().__init__('Help', None, data_manager)
        self._help_browser = QTextBrowser()
        """ The user guide content is displayed entirely in this widget. """
        self._section_combo = QComboBox()
        """ Combo box selects which section of the user's guide is displayed. """

        # set up the combo box that selects which section of the user guide to view
        self._section_combo.addItems([k for k in self.SECTIONS.keys()])
        self._section_combo.setCurrentText(self.DEF_SECTION)
        self._section_combo.currentTextChanged.connect(self._load_section)

        # configure text brower, then load the default section initially
        self._help_browser.setReadOnly(True)
        self._help_browser.setOpenExternalLinks(True)
        self._help_browser.document().setDefaultStyleSheet("""
                    body { font-family: sans-serif; line-height: 1.4; }
                    h2 { color: cornflowerblue; }
                    h3 { color: navy; font-style: italic; }
                    h4 { text-decoration: underline; }
                    a { color: #ae2012; text-decoration: underline; }
                    p { margin-bottom: 2em; }
                    pre, code {
                        font-family: monospace;
                        background-color: #e9ecef;
                        color: darkgreen;
                        border-radius: 3px;
                        line-height: 1;
                    }
                    ul, ol { margin-left: 20px; }
                    blockquote {
                        border-left: 4px solid #cccccc;
                        padding-left: 10px;
                        margin-left: 15px;
                        color: #666666;
                        font-style: italic;
                    }
                """)

        self._load_section(self.DEF_SECTION)

        main_layout = QVBoxLayout()
        control_line = QHBoxLayout()
        control_line.addWidget(QLabel('Chapter:'))
        control_line.addWidget(self._section_combo)
        control_line.addStretch(1)
        main_layout.addLayout(control_line)
        main_layout.addWidget(self._help_browser)

        self.view_container.setLayout(main_layout)

    @Slot(str)
    def _load_section(self, section_name: str) -> None:
        """
        Whenever the user selects a different section of the user's guide, load that section's markdown into the text
        browser.
        :param section_name: The name of the section selected
        """
        try:
            filename = self.SECTIONS.get(section_name, None)
            if filename is None:
                return
            # noinspection PyTypeChecker
            inp_file = (impresources.files(reach_assets) / filename)
            with inp_file.open("r") as f:
                markdown = f.read()
                self._help_browser.setMarkdown(markdown)
                self._help_browser.setHtml(self._help_browser.toHtml())  # HACK
        except Exception as e:
            get_application_logger().error(f"Failed to load {section_name} of UG: {e}")
