import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGroupBox, QVBoxLayout, QListView, QStandardItemModel
from PyQt5.QtCore import Qt

# Sample list of months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

app = QApplication(sys.argv)

# Create the main window widget
window = QWidget()
window.setWindowTitle("Filter Example")

# Create the main layout for the window
layout = QVBoxLayout()

# Create a QGroupBox to hold the filters
filter_group = QGroupBox("Filters")
filter_layout = QVBoxLayout()

# Create a custom combo box filter for months with multiple selections
month_filter = QListView()
month_filter.setSelectionMode(QListView.MultiSelection)

# Create a model for the combo box
model = QStandardItemModel()
for month in months:
    item = QStandardItem(month)
    item.setCheckable(True)
    item.setCheckState(Qt.Checked)
    model.appendRow(item)

month_filter.setModel(model)

# Function to update the combo box text based on selected items
def update_combo_text():
    selected_months = [model.itemFromIndex(index).text() for index in month_filter.selectedIndexes()]
    month_filter.setEditText(", ".join(selected_months))

# Connect the itemChanged signal to update the combo box text
model.itemChanged.connect(update_combo_text)

# Initially update the combo box text
update_combo_text()

# Add the combo box filter to the main layout
filter_layout.addWidget(month_filter)
filter_group.setLayout(filter_layout)

# Add the filter_group to the main layout of the window
layout.addWidget(filter_group)

# Set the layout of the main window
window.setLayout(layout)

window.show()
sys.exit(app.exec_())
