import sys

import pandas
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QFileDialog,
    QTabWidget, QComboBox, QGroupBox, QFormLayout, QScrollArea,QDialog, QHBoxLayout, QLabel,\
    QSpacerItem, QSizePolicy, QTextBrowser,QCheckBox
)
from PyQt5.QtGui import QPixmap

from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Create global data variable
data = None

# Default filter value
DEFAULT_FILTER = "ALL"

def load_excel():
    global data
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, "Open Excel File", "", "Excel Files (*.xlsx *.xls)")

    if file_path:
        try:
            data = pd.read_excel(file_path)

            # Set default filter values to "ALL"
            for combo in [brand_filter_combo, geo_filter_combo, campus_filter_combo, Department_filter_combo]:
                combo.clear()
                combo.addItem(DEFAULT_FILTER)

            # Populate filter combo boxes with unique values
            unique_filters = {
                'Brand': brand_filter_combo,
                "GEO": geo_filter_combo,
                'Campus': campus_filter_combo,
                'Department': Department_filter_combo
            }

            for column, combo in unique_filters.items():
                unique_values = data[column].unique()
                combo.addItems(unique_values)

            visualize_data()
        except Exception as e:
            print(f"Error loading Excel file: {str(e)}")


def visualize_data():
    selected_viz = visualization_combo.currentText()
    current_page_index = tab_widget.currentIndex()

    canvas = canvases[current_page_index]
    fig = canvas.figure
    fig.clear()

    filtered_data = data  # Start with unfiltered data

    # Apply filters based on selected options
    selected_filters = {
        'Brand': brand_filter_combo.currentText(),
        "GEO'": geo_filter_combo.currentText(),
        'Campus': campus_filter_combo.currentText(),
        'Department': Department_filter_combo.currentText()
    }

    for column, value in selected_filters.items():
        if value != DEFAULT_FILTER:
            filtered_data = filtered_data[filtered_data[column] == value]

    ax = fig.add_subplot(111)

    if selected_viz == "Audit Finding Categorizations":
        audit_categories = filtered_data['Audit Finding Categorisation (Observations/NC/OFI)'].value_counts()

        # Set a pleasant Seaborn style
        sns.set(style="ticks")

        # Create a bar chart with values displayed on top of each bar
        bars = sns.barplot(x=audit_categories.index, y=audit_categories.values, ax=ax, palette="Blues_d")

        for bar, count in zip(bars.patches, audit_categories.values):
            ax.text(bar.get_x() + bar.get_width() / 2, count + 1, str(count), ha='center', va='bottom', fontsize=12)

        ax.set_title("Audit Finding Categorizations", fontsize=16)
        ax.set_xlabel("Category", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.tick_params(axis='x', labelrotation=30, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
    elif selected_viz == "Status Open-Close":
        status_counts = filtered_data['Status'].value_counts()
        ax.pie(status_counts, labels=status_counts.index,
               autopct=lambda p: f'{p:.1f}%\n{int(p * sum(status_counts) / 100)}', startangle=90)
        ax.set_title("Status of Audit findings Open/Closed distributions")
        ax.legend(status_counts.index, title="Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    elif selected_viz in ["Ageing (from Report Date)", "Deviation (from target closure date)"]:
        # Set plot labels and title
        ax.set_title(
            "Ageing (from Report Date)" if selected_viz == "Ageing (from Report Date)" else
            "Total Findings Deviation (against target closure date)")
        ax.set_xlabel("Ageing (days)")
        ax.set_ylabel("Count of Audit findings")

        # Define the threshold values
        thresholds = [30, 60, 90, 120, 150, 180, 210, 240]

        # Count data points above each threshold
        counts = [np.sum(filtered_data['Ageing (from Report Date)'] > threshold) if selected_viz == "Ageing (from Report Date)"
                  else np.sum(filtered_data['Deviation (against target closure date)'] > threshold) for threshold in
                  thresholds]

        # Choose a pleasant color palette (e.g., 'Blues' for Ageing, 'Greens' for Deviation)
        palette = 'Blues' if selected_viz == "Ageing (from Report Date)" else 'Greens'

        # Create a bar plot with explicit x positions for the bars
        x_positions = np.arange(len(thresholds))
        ax.bar(x_positions, counts, width=0.8, alpha=0.8, color=sns.color_palette(palette, len(thresholds)),
               label='Count Above Threshold')

        # Set the x-axis tick positions and labels to indicate thresholds
        ax.set_xticks(x_positions)
        ax.set_xticklabels(thresholds)

        # Add labels to the bars
        for x, count in zip(x_positions, counts):
            ax.text(x, count + 10, str(count), ha='center', va='bottom', fontsize=10)

        # Set the x-axis limits and add padding
        ax.set_xlim(-0.5, len(thresholds) - 0.5)

        # Add a legend
        ax.legend()

    canvas.draw()

# Create the PyQt5 application
app = QApplication(sys.argv)
app.setStyle("Fusion")

# Create the main window
main_window = QMainWindow()
main_window.setWindowTitle("GIIS Audit Data Analytics")
main_window.setGeometry(100, 100, 1000, 600)

# Create the central widget and layout
central_widget = QWidget()
main_window.setCentralWidget(central_widget)
layout = QHBoxLayout(central_widget)

# Left Sidebar
left_sidebar = QGroupBox("")
left_layout = QVBoxLayout(left_sidebar)

# Create Load Excel button
load_excel_button = QPushButton("Load Audit tracking File")
load_excel_button.setStyleSheet(
    "QPushButton {"
    "background-color: #007ACC;"
    "color: white;"
    "border: none;"
    "border-radius: 4px;"
    "padding: 10px 20px;"
    "font-size: 16px;"
    "}"
    "QPushButton:hover {"
    "background-color: #005E9C;"
    "}"
    "QPushButton:pressed {"
    "background-color: #004A80;"
    "}"
)

# Create Visualization combo box
visualization_combo = QComboBox()
visualization_combo.addItems(["Audit Finding Categorizations", "Status Open-Close", \
                              "Ageing (from Report Date)","Deviation (from target closure date)"])
visualization_combo.setStyleSheet(
    "QComboBox {"
    "background-color: #F0F0F0;"
    "color: #007ACC;"
    "border: 1px solid #007ACC;"
    "border-radius: 5px;"
    "padding: 5px;"
    "}"
)

info_label = QLabel()
info_pixmap = QPixmap("logo.jpg")  # Provide the correct path to your image
info_label.setPixmap(info_pixmap)
info_label.setAlignment(Qt.AlignCenter)

# Add Load Excel button and Visualization combo box to left layout
left_layout.addWidget(info_label)
left_layout.addWidget(load_excel_button)
left_layout.addWidget(visualization_combo)

# Filters
filter_group = QGroupBox("Filters")
filter_layout = QFormLayout(filter_group)

filter_groupPV2 = QGroupBox("Pv2 Filters")
filter_layoutPV2 = QFormLayout(filter_groupPV2)


# Create filter combo boxes
brand_filter_combo = QComboBox()
geo_filter_combo = QComboBox()
campus_filter_combo = QComboBox()
Department_filter_combo = QComboBox()

# Create Apply Filters button
apply_filter_button = QPushButton("Apply Filters")
apply_filter_button.setStyleSheet(
    "QPushButton {"
    "background-color: #007ACC;"
    "color: white;"
    "border: none;"
    "border-radius: 4px;"
    "padding: 10px 20px;"
    "font-size: 16px;"
    "}"
    "QPushButton:hover {"
    "background-color: #005E9C;"
    "}"
    "QPushButton:pressed {"
    "background-color: #004A80;"
    "}"
)


apply_filter_buttonPV2 = QPushButton("Apply Filters")
apply_filter_buttonPV2.setStyleSheet(
    "QPushButton {"
    "background-color: #007ACC;"
    "color: white;"
    "border: none;"
    "border-radius: 4px;"
    "padding: 10px 20px;"
    "font-size: 16px;"
    "}"
    "QPushButton:hover {"
    "background-color: #005E9C;"
    "}"
    "QPushButton:pressed {"
    "background-color: #004A80;"
    "}"
)

# Add filter combo boxes and Apply Filters button to filter layout
filter_layout.addRow("Brand:", brand_filter_combo)
filter_layout.addRow("GEO:", geo_filter_combo)
filter_layout.addRow("Campus:", campus_filter_combo)
filter_layout.addRow("Department:", Department_filter_combo)
filter_layout.addRow(apply_filter_button)



def ArbitratePV2():
    global df,canvas2

    canvas2.figure.clear()

    main_window.update()

    if visualization_combo2.currentText() == "Total Monthly Pv2 Scores":
        plot_total_all_monthly()
    elif visualization_combo2.currentText() == "Parameters wise Pv2 Scores":
        filtered_rows = df[df['Unit of measurement'] == 'Sub Total']
        # filtered_rows.to_csv("a.csv")
        plot_monthly_stats_stacked(filtered_rows , "Criteria of assessment",filtered_months , canvas2)
    elif visualization_combo2.currentText() == "Criteria-Wise Monthly Total Scores":
            # filtered_rows = df[df['Criteria of assessment'] == AssessmentCriterias.currentText()]
            # plot_monthly_stats_pie()
            filtered_rows = df[df['Unit of measurement'] == 'Sub Total'].copy()
            filtered_rows = filtered_rows.set_index("Criteria of assessment")
            # filtered_rows.to_csv("a.csv")
            print(filtered_rows)
            plot_monthly_stats_stacked(filtered_rows, "Criteria of assessment", filtered_months, canvas2)

def UpdateAssParameters():
    Criteria = AssessmentCriterias.currentText()
    AssessmentParameters.addItems(set(df[df["Criteria of assessment"] == Criteria]["Parameters"].unique().tolist()+["All"]))
    main_window.update()

AssessmentCriterias = QComboBox()
AssessmentParameters = QComboBox()
AssessmentParameters.currentIndexChanged.connect(UpdateAssParameters)

# Initialize the list to store selected months
selected_months = []
filtered_months = []

def open_month_selection_dialog():
    # Create a dialog to select months
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Select Months")
    # Apply the Fusion style to the dialog
    dialog.setStyleSheet("QDialog { background-color: white; }")

    # Set the width of the dialog (adjust as needed)
    dialog.setGeometry(100, 100, 400, 300)

    # Create checkboxes for each month
    months = {
        "January": -13,
        "February": -12,
        "March": -11,
        "April": -10,
        "May": -9,
        "June": -8,
        "July": -7,
        "August": -6,
        "September": -5,
        "October": -4,
        "November": -3,
        "December": -2
    }
    checkboxes = []

    for month in months.keys():
        checkbox = QCheckBox(month)
        checkbox.setChecked(month in selected_months)  # Check the boxes based on the initial selection
        checkbox.stateChanged.connect(lambda state, month=month: update_selected_months(month, state))
        checkboxes.append(checkbox)

    # Check the boxes based on the previous selection
    for checkbox in checkboxes:
        if checkbox.text() in selected_months:
            checkbox.setChecked(True)

    # Create a layout for the checkboxes
    layout = QVBoxLayout()
    for checkbox in checkboxes:
        layout.addWidget(checkbox)

    def save_and_close_dialog():
        filtered_months.clear()
        # Update the selected_months list and print it
        selected_months.clear()
        for checkbox in checkboxes:
            if checkbox.isChecked():
                selected_months.append(checkbox.text())
                filtered_months.append(months[checkbox.text()])
        dialog.accept()

    # Create a "Save" button to save the selection
    save_button = QPushButton("Save")
    save_button.clicked.connect(save_and_close_dialog)

    layout.addWidget(save_button)
    dialog.setLayout(layout)

    # Show the dialog
    dialog.exec_()

def update_selected_months(month, state):
    if state == 2:  # 2 corresponds to Checked state
        selected_months.append(month)
    else:
        selected_months.remove(month)

select_months_button = QPushButton("Select Months")
select_months_button.clicked.connect(open_month_selection_dialog)

CriteriaParameters = QComboBox()

filter_layoutPV2.addRow("Select Months:", select_months_button)
filter_layoutPV2.addRow("Assessment Criteria:", AssessmentCriterias)
filter_layoutPV2.addRow("Criteria Parameters:", AssessmentParameters)
# filter_layoutPV2.addRow("Months:", campus_filter_combo)
# filter_layoutPV2.addRow("Department:", Department_filter_combo)
filter_layoutPV2.addRow(apply_filter_buttonPV2)
apply_filter_buttonPV2.clicked.connect(ArbitratePV2)
# Add filter group to left layout

left_layout.addWidget(filter_group)
left_layout.addWidget(filter_groupPV2)

# Spacer to create space between the widgets
spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
left_layout.addItem(spacer)

# Info label
info_label = QLabel("Welcome to the Data Visualization Notebook")
info_label.setAlignment(Qt.AlignCenter)
info_label.setStyleSheet(
    "QLabel { color: #007ACC; font-size: 18px; }"
)

# Visualization Area
visualization_scroll_area = QScrollArea()

# Tab Widget
tab_widget = QTabWidget()
tab_widget.setStyleSheet(
    "QTabBar::tab {"
    "background-color: #007ACC;"
    "color: white;"
    "border: 1px solid #007ACC;"
    "padding: 8px 12px;"
    "}"
    "QTabBar::tab:selected {"
    "background-color: #004A80;"
    "border-bottom: 2px solid #FFA500;"
    "}"
)

# Add left sidebar and tab widget to the main layout
layout.addWidget(left_sidebar)
layout.addWidget(tab_widget)

# Function to create a visualization page
def create_visualization_page(pname):
    page = QWidget()
    tab_widget.addTab(page, pname)
    return page

# Create visualization pages
page1 = create_visualization_page("1.0 Audit Tracking_GSF")
page2 = create_visualization_page("2.0 Pv2 Reports")

canvas1 = FigureCanvas(Figure(figsize=(8, 6)))
canvas2 = FigureCanvas(Figure(figsize=(8, 6)))

canvases = [canvas1, canvas2]

# Add canvases to visualization pages
page1_layout = QVBoxLayout(page1)
page1_layout.addWidget(canvas1)

page2_layout = QVBoxLayout(page2)
page2_layout.addWidget(canvas2)

# Connect signals to functions
load_excel_button.clicked.connect(load_excel)
visualization_combo.currentIndexChanged.connect(lambda x: visualize_data())
apply_filter_button.clicked.connect(lambda: visualize_data())



df=pandas.DataFrame()


def plot_total_all_monthly():
    global canvas2
    # canvas2 = FigureCanvas(Figure(figsize=(8, 6)))
    canvas2.figure.clear()
    main_window.update()
    canvas2.update()
    total_rows = df[df['Unit of measurement'] == 'Total']

    # Extract the last row containing totals for each month
    totals_row = total_rows.iloc[-1, filtered_months]  # Assuming the month columns start from index 8 to 20

    # Create a more beautiful and descriptive bar plot

    fig = canvas2.figure
    ax = fig.add_subplot(111)

    # Customize the color palette for the bars
    colors = sns.color_palette("Set2", len(totals_row))

    # Create the bar plot
    ax.bar(range(len(totals_row)), totals_row, color=colors, alpha=0.7)
    ax.set_xlabel('Months')
    ax.set_ylabel('Overall Total')
    ax.set_title('Monthly Totals for 2023', fontsize=16)

    # Add data labels on top of the bars
    for i, total in enumerate(totals_row):
        ax.text(i, total, f'{total:.0f}', ha='center', va='bottom', fontsize=12)

    # Customize the x-axis labels
    ax.set_xticks(range(len(totals_row)))
    ax.set_xticklabels(selected_months, rotation=45)

    # Show the plot with a grid
    sns.despine(left=True, bottom=True)  # Remove spines on the left and bottom
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    main_window.update()
    canvas2.update()

def plot_monthly_stats_pie():

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")  # Use Seaborn for an attractive style

    # Define a custom color palette for the bars
    month_colors = sns.color_palette("Paired", n_colors=len(filtered_rows))

    # Width of each bar group
    bar_width = 0.15
    bar_positions = range(len(selected_months))

    for index, row in enumerate(filtered_rows.iterrows()):
        _, row = row
        totals_row = row.iloc[:,filtered_months]
        bar_positions_grouped = [pos + index * bar_width for pos in bar_positions]

        # Create the grouped bar
        plt.bar(bar_positions_grouped, totals_row, label=row["Parameters"], width=bar_width, color=month_colors[index])

        # Add values on top of the bars
        for pos, value in zip(bar_positions_grouped, totals_row):
            plt.text(pos, value, f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Months')
    plt.ylabel('Total Score')
    plt.title(f'Monthly Totals for {Campus_name} in 2023', fontsize=14)

    # Customize the legend
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right')

    # Customize the x-axis labels
    # ax.set_xticks([pos for pos in bar_positions], df.columns[filtered_months], rotation=45)

    # Get the current axes
    ax = plt.gca()

    # Create a legend and move it outside the graph
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Display the plot
    plt.show()

def plot_monthly_stats_stacked(df, category_column, columns_to_display, canvas):
        # Set 'Category' as the index

        # Create a Figure and set the canvas as its figure
        fig = Figure(figsize=(8, 6))
        FigureCanvas(fig)
        canvas2.figure = fig
        ax = fig.add_subplot(111)

        # Select the columns to display based on the given negative column numbers
        selected_columns = df[df.columns[columns_to_display]]
        print(selected_columns)
        # Transpose the DataFrame for grouped bar plotting
        selected_columns = selected_columns.T

        # Get the number of categories and months
        num_categories = len(selected_columns.index)
        num_months = len(selected_columns.columns)

        # Set the width of the bars to create space between them
        bar_width = 0.2

        # Calculate the positions for the bars
        positions = range(num_months)

        # Plot the bars for each category
        for i, category in enumerate(selected_columns.index):
            x = [p + i * bar_width for p in positions]
            bar = ax.bar(x, selected_columns.loc[category], bar_width, label=category)

            # Add labels displaying values at the bottom of the bars
            for rect in bar:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                            textcoords="offset points", ha='center', va='bottom')

        # Customize the plot if needed
        ax.set_ylabel('Pv2 Scores')
        ax.set_xlabel('Assessment Categories')
        ax.set_title('Total monthly Category wise Pv2 scores')
        ax.set_xticks([p + 0.3 * bar_width * (num_categories - 1) for p in positions])
        ax.set_xticklabels(selected_columns.columns)
        # Rotate the x-axis tick labels
        ax.set_xticklabels(selected_columns.columns, rotation=10, ha='right')  # Adjust the rotation angle as needed

        ax.legend(title='Categories')

        # Draw the plot on the canvas
        canvas.draw()

# Create Load Excel button
load_excel_Pv2_button = QPushButton("Load Pv2 Campus File")
load_excel_Pv2_button.setStyleSheet(
    "QPushButton {"
    "background-color: #007ACC;"
    "color: white;"
    "border: none;"
    "border-radius: 4px;"
    "padding: 10px 20px;"
    "font-size: 16px;"
    "}"
    "QPushButton:hover {"
    "background-color: #005E9C;"
    "}"
    "QPushButton:pressed {"
    "background-color: #004A80;"
    "}"
)

Campus_name = ''
filtered_rows = pandas.DataFrame()
def load_pv2_excel():
    global data , df, Campus_name,filtered_rows

    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, "Open Excel File", "", "Excel Files (*.xlsx *.xls)")
    if not("pv2" in file_path.lower()):
        print("Error: Wrong file selected")
        return
    # Read the Excel file into a Pandas DataFrame
    df = pd.read_excel(file_path, header=0)
    df.ffill(inplace=True)
    df.iloc[0, :7] = df.columns.values[:7]
    Campus_name = df.columns[8]

    # Set the first row as the column headers
    df.columns = df.iloc[0]

    # Drop the first row, which is now the column headers
    df = df.iloc[1:].reset_index(drop=True)
    AssessmentCriterias.addItems(df["Criteria of assessment"].unique().tolist())
    Criteria = AssessmentCriterias.currentText()
    AssessmentParameters.addItems(set(df[df["Criteria of assessment"] == Criteria]["Parameters"].unique().tolist()+["All"]))


left_layout.addWidget(load_excel_Pv2_button)
left_layout.addWidget(load_excel_Pv2_button)
load_excel_Pv2_button.clicked.connect(load_pv2_excel)

# Create Visualization combo box
visualization_combo2 = QComboBox()

visualization_combo2.addItems(["Total Monthly Pv2 Scores", \
                               "Criteria-Wise Monthly Total Scores", \
                              "Parameters wise Pv2 Scores",\
                               "Deviation (from target closure date)",\
                               "Reserved1","Reserved1"])

visualization_combo2.setStyleSheet(
    "QComboBox {"
    "background-color: #F0F0F0;"
    "color: #007ACC;"
    "border: 1px solid #007ACC;"
    "border-radius: 5px;"
    "padding: 5px;"
    "}"
)
left_layout.addWidget(visualization_combo2)


# Show the main window and start the application event loop
main_window.show()
sys.exit(app.exec_())
