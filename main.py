import sys

import pandas
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QFileDialog,
    QTabWidget, QComboBox, QGroupBox, QFormLayout, QScrollArea, QHBoxLayout, QLabel, QSpacerItem, QSizePolicy, QTextBrowser
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
load_excel_button = QPushButton("Load File")
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

# Add filter combo boxes and Apply Filters button to filter layout
filter_layout.addRow("Brand:", brand_filter_combo)
filter_layout.addRow("GEO:", geo_filter_combo)
filter_layout.addRow("Campus:", campus_filter_combo)
filter_layout.addRow("Department:", Department_filter_combo)
filter_layout.addRow(apply_filter_button)

# Add filter group to left layout
left_layout.addWidget(filter_group)

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
    total_rows = df[df['Unit of measurement'] == 'Total']

    # Extract the last row containing totals for each month
    totals_row = total_rows.iloc[-1, len(df.columns)-12 : -1]  # Assuming the month columns start from index 8 to 20

    # Create a more beautiful and descriptive bar plot
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")  # Use Seaborn for a more attractive style

    # Customize the color palette for the bars
    colors = sns.color_palette("Set2", len(totals_row))

    # Create the bar plot
    plt.bar(range(len(totals_row)), totals_row, color=colors, alpha=0.7)
    plt.xlabel('Months')
    plt.ylabel('Overall Total')
    plt.title('Monthly Totals for 2023', fontsize=16)

    # Add data labels on top of the bars
    for i, total in enumerate(totals_row):
        plt.text(i, total, f'{total:.0f}', ha='center', va='bottom', fontsize=12)

    # Customize the x-axis labels
    plt.xticks(range(len(totals_row)), df.columns[len(df.columns)-12 : -1], rotation=45)
    plt.tight_layout()  # Ensure the labels are not cut off

    # Show the plot with a grid
    sns.despine(left=True, bottom=True)  # Remove spines on the left and bottom
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.show()

def plot_monthly_stats_pie():
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")  # Use Seaborn for an attractive style

    # Define a custom color palette for the bars
    month_colors = sns.color_palette("Paired", n_colors=len(filtered_rows))

    # Width of each bar group
    bar_width = 0.15
    bar_positions = range(len(df.columns[len(df.columns) - 13: -1]))

    for index, row in enumerate(filtered_rows.iterrows()):
        _, row = row
        totals_row = row[len(df.columns) - 13:-1]
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
    plt.xticks([pos + (len(filtered_rows) - 1) * bar_width / 2 for pos in bar_positions],
               df.columns[len(df.columns) - 13: -1], rotation=45)

    # Get the current axes
    ax = plt.gca()

    # Create a legend and move it outside the graph
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Display the plot
    plt.show()
def plot_monthly_stats_stacked():

    # Create a grouped bar plot with values on top
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")  # Use Seaborn for an attractive style

    # Define a custom color palette for the bars
    month_colors = sns.color_palette("Set2", n_colors=len(filtered_rows))

    # Width of each bar group
    bar_width = 0.15
    bar_positions = range(len(df.columns[len(df.columns)-13 : -1]))

    for index, row in enumerate(filtered_rows.iterrows()):
        _, row = row
        totals_row = row[len(df.columns) - 13:-1]
        bar_positions_grouped = [pos + index * bar_width for pos in bar_positions]

        # Create the grouped bar
        plt.bar(bar_positions_grouped, totals_row, label=row[0], width=bar_width, color=month_colors[index])

        # Add values on top of the bars
        for pos, value in zip(bar_positions_grouped, totals_row):
            plt.text(pos, value, f'{value:.0f}', ha='center', va='bottom', fontsize=10)

    plt.xlabel('Months')
    plt.ylabel('Total Score')
    plt.title(f'Monthly Totals for {Campus_name} in 2023', fontsize=14)

    # Customize the legend
    # plt.legend(loc='upper right',bbox_to_anchor=(1, 1))

    # Customize the x-axis labels
    plt.xticks([pos + (len(filtered_rows) - 1) * bar_width / 2 for pos in bar_positions],\
               df.columns[len(df.columns)-13 : -1], rotation=45)

    # Get the current axes
    ax = plt.gca()

    # Create a legend and move it outside the graph
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    # Display the plot
    plt.show()

# Filter rows that meet a certain criteria (adjust this condition as needed)

# plot_total_all_monthly()
# unique_criterias = df['Criteria of assessment'].unique().tolist()
# for criteria in unique_criterias :
#     filtered_rows = df[df['Criteria of assessment'] == criteria]
#     plot_monthly_stats_pie()


# filtered_rows = df[df['Unit of measurement'] == 'Sub Total']
# plot_monthly_stats_stacked()


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

    if visualization_combo2.currentText()=="Total Monthly Pv2 Scores":
        plot_total_all_monthly()
    elif  visualization_combo2.currentText()=="Parameters Wise Scores":
        filtered_rows = df[df['Unit of measurement'] == 'Sub Total']
        plot_monthly_stats_stacked()
    elif  visualization_combo2.currentText()=="Criteria-Wise Monthly Total Scores":
            unique_criterias = df['Criteria of assessment'].unique().tolist()
            for criteria in unique_criterias :
                filtered_rows = df[df['Criteria of assessment'] == criteria]
                plot_monthly_stats_pie()

left_layout.addWidget(load_excel_Pv2_button)
left_layout.addWidget(load_excel_Pv2_button)
load_excel_Pv2_button.clicked.connect(load_pv2_excel)

# Create Visualization combo box
visualization_combo2 = QComboBox()
visualization_combo2.addItems(["Total Monthly Pv2 Scores", "Criteria-Wise Monthly Total Scores", \
                              "Parameters Wise Scores","Deviation (from target closure date)","Reserved1","Reserved1"])
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
