from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QWidget, QComboBox, QFormLayout, QDateEdit

from anylabeling.views.labeling.utils.duckdb_utils import DuckDB


class FilesFilterWidget(QWidget):
    def __init__(self, db: DuckDB, parent=None):
        super(FilesFilterWidget, self).__init__(parent)

        self.db = db
        self.was_reviewed_combo_box = QComboBox()
        self.was_reviewed_combo_box.addItems(["All", "Yes", "No"])
        self.was_reviewed_combo_box.currentIndexChanged.connect(
            parent.filter_files_in_file_list
        )
        self.sample_id_combo_box = QComboBox()
        self.objective_combo_box = QComboBox()

        self.reload_combo_boxes_values()
        self.sample_id_combo_box.currentIndexChanged.connect(
            parent.filter_files_in_file_list
        )
        self.objective_combo_box.currentIndexChanged.connect(
            parent.filter_files_in_file_list
        )
        self.from_date_edit = QDateEdit(calendarPopup=True, displayFormat="yyyy-MM-dd", date=QDate(2024, 1, 1))
        self.from_date_edit.dateChanged.connect(parent.filter_files_in_file_list)
        self.until_date_edit = QDateEdit(calendarPopup=True, displayFormat="yyyy-MM-dd", date=QDate(2025, 12, 31))
        self.until_date_edit.dateChanged.connect(parent.filter_files_in_file_list)

        layout = QFormLayout()
        layout.addRow("Reviewed", self.was_reviewed_combo_box)
        layout.addRow("From", self.from_date_edit)
        layout.addRow("Until", self.until_date_edit)
        layout.addRow("Sample ID", self.sample_id_combo_box)
        layout.addRow("Objective", self.objective_combo_box)
        self.setLayout(layout)

    def reload_combo_boxes_values(self):
        self.sample_id_combo_box.clear()
        self.sample_id_combo_box.addItems([""] + self.db.get_column_values("sample_id"))
        self.objective_combo_box.clear()
        self.objective_combo_box.addItems([""] + self.db.get_column_values("objective"))

    def get_was_reviewed_filter_value(self) -> bool | None:
        if self.was_reviewed_combo_box.currentText() == "All":
            return None
        return self.was_reviewed_combo_box.currentText() == "Yes"

    def get_sample_id_filter_value(self) -> str | None:
        if self.sample_id_combo_box.currentText() == "":
            return None
        return self.sample_id_combo_box.currentText()

    def get_objective_filter_value(self) -> str | None:
        if self.objective_combo_box.currentText() == "":
            return None
        return self.objective_combo_box.currentText()

    def get_dates_range_filter_value(self) -> (str, str):
        return self.from_date_edit.date().toString("yyyy-MM-dd"), self.until_date_edit.date().toString("yyyy-MM-dd")
