import sys
import os
import random
import pandas
import pandas as pd
import textwrap
import seaborn as sns
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QFileDialog,
    QTabWidget, QComboBox, QGroupBox, QFormLayout, QScrollArea,QDialog, QHBoxLayout, QLabel,\
    QSpacerItem, QSizePolicy, QTextBrowser,QCheckBox,QStackedLayout,QMessageBox,QTableWidget,QTableWidgetItem
)
from tkinter import filedialog

from PyQt5.QtGui import QPixmap
import hashlib
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import warnings
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QColor, QPainter, QBrush
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re
import textwrap
import matplotlib.style as mplstyle

# Suppress all warnings
warnings.filterwarnings("ignore")

# Create global data variable
data = ""

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

            load_excel_button.setStyleSheet(
                "QPushButton {"
                "background-color: #097969;"
                "color: white;"
                "border: none;"
                "border-radius: 4px;"
                "padding: 10px 20px;"
                "font-size: 16px;"
                "}"
                "QPushButton:hover {"
                "background-color: #50C878;"
                "}"
                "QPushButton:pressed {"
                "background-color: #50C878;"
                "}"
            )

        except Exception as e:
            print(f"Error loading Excel file: {str(e)}")



def show_warning(message):
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Warning")
    msg_box.setText(message)

    # Set a custom stylesheet to make the warning box more modern
    msg_box.setStyleSheet(
        """
        QMessageBox {
            background-color: dodgerblue;
            border: 1px solid #007acc;
            color: white;
        }
        QMessageBox QLabel {
            color: white;
        }
        """
    )

    msg_box.exec_()


def plot_unique_value_counts_pie_on_canvas(df, column_name, canvas):
    # Clear the canvas before plotting
    canvas.figure.clear()

    # Count unique values in the specified column
    value_counts = df[column_name].value_counts()


    # Create a pie chart with modern styling
    fig = canvas.figure
    ax = fig.add_subplot(111)
    colors = plt.cm.Paired(range(len(value_counts)))

    # Calculate percentages
    total = sum(value_counts)
    percentages = [count / total * 100 for count in value_counts]

    # Create the pie chart with percentage and absolute counts
    wedges, texts, autotexts = ax.pie(value_counts, labels=None, startangle=140, colors=colors,
                                      autopct=lambda p: f'{p:.1f}% ')

    # Set custom labels with both labels and values
    labels = [f'{label} ({count})' for label, count in zip(value_counts.index, value_counts)]
    ax.legend(wedges, labels, title=column_name, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)

    # Make the pie chart circular
    ax.axis('equal')

    # Set a pleasant title
    ax.set_title('Audit findings categorisations', fontsize=16)

    # Adjust layout
    fig.tight_layout()

    # Display the pie chart on the canvas
    canvas.draw()


# Example usage (without actual canvas and dataframe)
# plot_unique_value_counts_pie_on_canvas(df, 'column_name', canvas)

# Example usage (without actual canvas and dataframe)
# plot_unique_value_counts_pie_on_canvas(df, 'column_name', canvas)


def visualize_data():
    # global data
    # if data == "":
    #     show_warning("Please load Audit tracking file into the system.")
    #     return
    try:
        selected_viz = visualization_combo.currentText()
        current_page_index = tab_widget.currentIndex()

        fig = canvas1.figure
        fig.clear()

        filtered_data = data  # Start with unfiltered data

        # Apply filters based on selected options
        selected_filters = {
            'Brand': brand_filter_combo.currentText(),
            "GEO": geo_filter_combo.currentText(),
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
            fig.tight_layout()
            canvas1.draw()
        elif selected_viz == "Status Open-Close":
            status_counts = filtered_data['Status'].value_counts()
            ax.pie(status_counts, labels=status_counts.index,
                   autopct=lambda p: f'{p:.1f}%\n{int(p * sum(status_counts) / 100)}', startangle=90)
            ax.set_title("Status of Audit findings Open/Closed distributions")
            ax.legend(status_counts.index, title="Status", bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            canvas1.draw()
        elif selected_viz == "Audit Alerts":
            plot_unique_value_counts_pie_on_canvas(filtered_data, "Audit Alerts", canvas1)
            filtered_data = filtered_data[filtered_data["Audit Alerts"] == "Negative"]
            create_table(filtered_data)
            tab_widget.setCurrentIndex(3)  # Switch to the tab with the table

        elif selected_viz in ["Ageing (from Report Date)", "Deviation (from target closure date)"]:
            # Set plot labels and title
            ax.set_title(
                "Ageing (from Report Date)" if selected_viz == "Ageing (from Report Date)" else
                "Total Findings Deviation (against target closure date)", fontsize=14)
            ax.set_xlabel("Ageing (days)", fontsize=12)
            ax.set_ylabel("Count of Audit findings", fontsize=12)

            # Define the threshold values
            thresholds = [30, 60, 90, 120, 150, 180, 210, 240]

            # Count data points above each threshold
            counts = [np.sum(
                filtered_data['Ageing (from Report Date)'] > threshold) if selected_viz == "Ageing (from Report Date)"
                      else np.sum(filtered_data['Deviation (against target closure date)'] > threshold) for threshold in
                      thresholds]

            # Choose a pleasant color palette (e.g., 'Blues' for Ageing, 'Greens' for Deviation)
            palette = 'Blues' if selected_viz == "Ageing (from Report Date)" else 'Greens'

            # Determine color intensity based on counts for visual emphasis
            colors = sns.color_palette(palette,
                                       n_colors=max(counts) + 1)  # Adjust +1 to ensure at least one color is selected

            # Create a bar plot with explicit x positions for the bars
            x_positions = np.arange(len(thresholds))
            bars = ax.bar(x_positions, counts, width=0.8, alpha=0.8,
                          color=[colors[count] for count in counts],  # Apply dynamic color based on count
                          label='Count of Findings Above Each Threshold')

            # Set the x-axis tick positions and labels to indicate thresholds
            ax.set_xticks(x_positions)
            ax.set_xticklabels(thresholds)

            # Add labels to the bars for exact counts
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

            # Add gridlines for better readability
            ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

            # Remove fixed aspect ratio to optimize space usage
            # ax.set_aspect('equal')  # Commented out to allow dynamic adjustment

            # Adjust x-axis limits to add some padding for visual appeal
            ax.set_xlim(-0.5, len(thresholds) - 0.5)

            # Add a legend with an improved description
            ax.legend(title='Audit Findings')

            fig.tight_layout()
            canvas1.draw()

        if tab_widget.currentIndex() !=0 :
            tab_widget.setCurrentIndex(0)
    except Exception as error:
        show_warning(f"Could not Visualize. Ensure Audit tracker file is loaded / Filters are valid\n {Exception}")

# Create the PyQt5 application
app = QApplication(sys.argv)
app.setStyle("Fusion")

# Create the main window
main_window = QMainWindow()
main_window.setWindowTitle("GSF Data Analytics")
main_window.setGeometry(100, 100, 1000, 600)

# Create the central widget and layout
central_widget = QWidget(main_window)
main_window.setCentralWidget(central_widget)
layout = QHBoxLayout(central_widget)

# Left Sidebar
left_sidebar = QGroupBox("")
left_sidebar.setMaximumWidth(400)
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
                              "Ageing (from Report Date)","Deviation (from target closure date)" , "Audit Alerts"])
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


# Define your custom function to be triggered
def on_analysis_type_changed(index):
    selected_value = analysis_type.itemText(index)

    if selected_value == "Audit Tracking analysis":
        analysis_layout_filters_disp.setCurrentIndex(0)
    elif selected_value == "Pv2 Scores analysis":
        analysis_layout_filters_disp.setCurrentIndex(1)
    elif selected_value == "PSAT-SSAT analysis":
        analysis_layout_filters_disp.setCurrentIndex(2)

    left_layout.update()


analysis_layout_filters_disp = QStackedLayout()

analysis_type = QComboBox()
analysis_type.addItem("Audit Tracking analysis")
analysis_type.addItem("Pv2 Scores analysis")
analysis_type.addItem("PSAT-SSAT analysis")

# Connect the signal to the custom function
analysis_type.currentIndexChanged.connect(on_analysis_type_changed)


# Add Load Excel button and Visualization combo box to left layout
left_layout.addWidget(info_label)
left_layout.addWidget(analysis_type)
left_layout.addLayout(analysis_layout_filters_disp)
# Filters
filter_group = QGroupBox("Filters")
filter_layout = QFormLayout(filter_group)

filter_groupPV2 = QGroupBox("Pv2 Filters")
filter_layoutPV2 = QFormLayout(filter_groupPV2)

filter_groupCSAT = QGroupBox("CSAT Filters")
filter_layoutCSAT = QFormLayout(filter_groupCSAT)

# Create filter combo boxes
brand_filter_combo = QComboBox()
geo_filter_combo = QComboBox()
campus_filter_combo = QComboBox()
Department_filter_combo = QComboBox()

# Create Execute button
apply_filter_button = QPushButton("Execute")
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


# Add filter combo boxes and Execute button to filter layout
filter_layout.addRow(load_excel_button)
filter_layout.addRow(visualization_combo)
filter_layout.addRow("Brand:", brand_filter_combo)
filter_layout.addRow("GEO:", geo_filter_combo)
filter_layout.addRow("Campus:", campus_filter_combo)
filter_layout.addRow("Department:", Department_filter_combo)
filter_layout.addRow(apply_filter_button)


def plot_total_weightage(grouped):
    # Create a new figure and axis
    fig = canvas2.figure
    ax = fig.add_subplot(111)

    # Create a pie chart
    ax.pie(grouped, labels=grouped.index, autopct='%1.1f%%', startangle=140)

    # Set a title for the pie chart
    ax.set_title('Weightage of Criteria of Assessments')

    # Show the canvas with the pie chart
    fig.tight_layout()
    canvas2.draw()

def ArbitratePV2():
    global df,canvas2
    if df.empty :
        show_warning("Please load valid Pv2 data file for any campus.")
    else:
        canvas2.figure.clear()

        main_window.update()

        if visualization_combo2.currentText() == "Total Monthly Pv2 Scores":
            plot_total_all_monthly()
        elif visualization_combo2.currentText() == "Criteria-Wise Monthly Total Scores":
            filtered_rows = df[df['Unit of measurement'] == 'Sub Total']
            plot_monthly_stats_stacked(filtered_rows ,filtered_months, "Criteria of assessment")
        elif visualization_combo2.currentText() == "Criteria's parameters scores":
                filtered_rows = None
                filtered_rows = df[df['Criteria of assessment'] == AssessmentCriterias.currentText()]
                # print(filtered_rows)
                plot_Criteria_parameters_line(filtered_rows, filtered_months, "Parameters")
        elif visualization_combo2.currentText() == "Total Weightage":
                filtered_rows = df
                filtered_rows["Total Weightage"].fillna(method='ffill', inplace=True)
                grouped = filtered_rows.groupby("Criteria of assessment")["Total Weightage"].mean()
                plot_total_weightage(grouped)
        elif visualization_combo2.currentText() == "Audit Alerts":
                # filtered_rows = df[df["Audit Alerts"]]
                plot_unique_value_counts_pie_on_canvas(df, "Audit Alerts",canvas2)
                filtered_data = df[df["Audit Alerts"] == "Negative"]
                create_table(filtered_data)
                tab_widget.setCurrentIndex(3)  # Switch to the tab with the table

        if tab_widget.currentIndex() !=1 :
            tab_widget.setCurrentIndex(1)

def UpdateAssParameters():
    Criteria = AssessmentCriterias.currentText()
    AssessmentParameters.addItems(set(df[df["Criteria of assessment"] == Criteria]["Parameters"].unique().tolist()+["All"]))
    main_window.update()

AssessmentCriterias = QComboBox()
AssessmentParameters = QComboBox()
AssessmentParameters.currentIndexChanged.connect(UpdateAssParameters)

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

# Initialize the list to store selected months
selected_months = [key for key in months.keys()]
filtered_months = [n for n in range(-13,-1,1)]

def open_month_selection_dialog():
    # Create a dialog to select months
    dialog = QDialog(main_window)
    dialog.setWindowTitle("Select Months")
    # Apply the Fusion style to the dialog
    dialog.setStyleSheet("QDialog { background-color: white; }")

    # Set the width of the dialog (adjust as needed)
    dialog.setGeometry(100, 100, 400, 300)

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

apply_filter_buttonPV2 = QPushButton("Execute")
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

# Create Visualization combo box
visualization_combo2 = QComboBox()

visualization_combo2.addItems(["Total Monthly Pv2 Scores", \
                               "Criteria-Wise Monthly Total Scores", \
                              "Criteria's parameters scores",\
                               "Total Weightage",\
                               "Deviation (from target closure date)",\
                               "Audit Alerts","Reserved1"])

visualization_combo2.setStyleSheet(
    "QComboBox {"
    "background-color: #F0F0F0;"
    "color: #007ACC;"
    "border: 1px solid #007ACC;"
    "border-radius: 5px;"
    "padding: 5px;"
    "}"
)

filter_layoutPV2.addRow(load_excel_Pv2_button)
filter_layoutPV2.addRow(visualization_combo2)
filter_layoutPV2.addRow("Select Months:", select_months_button)
filter_layoutPV2.addRow("Assessment Criteria:", AssessmentCriterias)
filter_layoutPV2.addRow(apply_filter_buttonPV2)
apply_filter_buttonPV2.clicked.connect(ArbitratePV2)
#__________________________________________________________________

load_excel_CSAT_button = QPushButton("Load SSAT/PSAT File")
load_excel_CSAT_button.setStyleSheet(
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

apply_comparison_buttonCSAT = QPushButton("Select Files")
apply_comparison_buttonCSAT.setStyleSheet(
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

AnalyseNPSButton = QPushButton("Analyse NPS")
AnalyseNPSButton.setStyleSheet(
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


apply_filter_buttonCSAT = QPushButton("Execute")
apply_filter_buttonCSAT.setStyleSheet(
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


Class_groups = []
def find_tables_with_headings(excel_file, headings):
    # List to store the table names
    table_names = []

    # List to store the tables
    tables = []

    # Load the Excel file
    xls = pd.ExcelFile(excel_file)

    # Iterate over each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame, without headers initially
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)

        # Iterate over each row
        for i, row in df.iterrows():
            # Check if the row contains the headings (in any order)
            if all(h in row.values for h in headings):
                # This row might be the start of a table
                # Capture the table name from the previous row

                # Find the end of the table
                end_row = i + 1
                while end_row < len(df) and not df.iloc[end_row].isna().all():
                    end_row += 1

                # Extract the table
                table = pd.read_excel(xls, sheet_name=sheet_name, header=i, nrows=end_row - i - 1)
                table_names.append(str(table['Category'].unique()[0]))

                table = table.round(2)
                tables.append(table)

                # Skip rows till end of this table
                i = end_row
    return tables , table_names


def plot_satisfaction_analysis(df, canvas, graph_type):
    """
    Plot a specific type of satisfaction analysis on the given canvas.

    :param df: DataFrame with the satisfaction data.
    :param canvas: The canvas on which to draw the plots.
    :param graph_type: The type of graph to plot ('bar', 'stacked', 'line').
    """
    # Clear the current figure
    canvas.figure.clear()

    # Create a subplot
    fig = canvas.figure
    ax = fig.add_subplot(111)

    def on_hover(event, bars, questions, ax, fig, canvas):
        hover_annotation = None

        # Check if there is already an annotation on the plot
        for child in ax.get_children():
            if isinstance(child, plt.Annotation):
                hover_annotation = child
                break

        if not hover_annotation:
            # Create a new annotation
            hover_annotation = ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                                           bbox=dict(boxstyle="round", fc="w"),
                                           arrowprops=dict(arrowstyle="->"))
            hover_annotation.get_bbox_patch().set_alpha(0.4)
            fig.canvas.draw_idle()

        for bar in bars:
            if bar.contains(event)[0]:
                # Get the index of the bar
                index = bars.index(bar)
                wrapped_text = textwrap.fill(questions[index], width=50)
                # Set the text and position for the annotation
                hover_annotation.set_text(wrapped_text)
                hover_annotation.xy = (event.xdata, bar.get_y() + bar.get_height() / 2)
                hover_annotation.set_visible(True)
                break
        else:
            # If not on a bar, hide the annotation
            hover_annotation.set_visible(False)

        # Redraw only the annotation
        hover_annotation.figure.canvas.draw_idle()

    def launch_question_window(df):
        new_window = tk.Tk()
        new_window.title('List of Questions')

        frame = ttk.Frame(new_window)
        frame.pack(fill='both', expand=True)

        tv = ttk.Treeview(frame, columns=('Question Number', 'Question Text'), show='headings')
        tv.heading('Question Number', text='Question Number')
        tv.heading('Question Text', text='Question Text')
        tv.column('Question Number', anchor='center', width=100)
        tv.column('Question Text', anchor='w', width=400)

        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tv.yview)
        tv.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')

        for i, question in enumerate(df['Questions']):
            tv.insert('', 'end', values=('Q' + str(i + 1), question))

        tv.pack(fill='both', expand=True)

        new_window.mainloop()

    if graph_type == 'pie':
        ax.clear()  # Clear the axis
        ax.set_title('Average Satisfaction % by Driver', pad=20)

        # Grouping the data by 'Drivers' and calculating the average satisfaction
        avg_satisfaction_by_driver = df.groupby('Drivers')['Satisfaction %'].mean()

        # Sort the values in descending order
        sorted_satisfaction = avg_satisfaction_by_driver.sort_values(ascending=False)

        # Creating a color palette from green to red (reversed RdYlGn)
        cmap = plt.cm.RdYlGn_r  # Use the reversed colormap (green to red)

        # Normalize satisfaction scores to ensure proper color assignment
        norm = plt.Normalize(vmin=sorted_satisfaction.min(), vmax=sorted_satisfaction.max())
        colors = [cmap(norm(value)) for value in sorted_satisfaction][::-1]

        # Creating a pie chart for average satisfaction by driver category
        wedges, texts, autotexts = ax.pie(sorted_satisfaction, autopct='%1.2f%%',
                                          # Show percentage annotations with 2 decimal places
                                          startangle=90,
                                          colors=colors,
                                          shadow=True)

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Improve the legend
        ax.legend(wedges, sorted_satisfaction.index,
                  title="Drivers",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))


    elif graph_type == 'stacked':
        # Visualization 2: Enhanced Stacked Bar Chart for Risk Profiles
        if 'Accept/At Risk' in df.columns:
            risk_profile_counts = df.groupby(['Drivers', 'Accept/At Risk']).size().unstack().fillna(0)
        elif "At Risk/Accepted" in df.columns :
            risk_profile_counts = df.groupby(['Drivers', 'At Risk/Accepted']).size().unstack().fillna(0)
        else:
            print("Error , At Risk/Accepted not found")

        colors = {'Accepted': '#33FF57', 'At Risk': '#FF5733'}  # Green for Accepted, Red for At Risk
        x = np.arange(len(risk_profile_counts.index))  # x-axis positions
        bar_width = 0.35  # Bar width

        ax.clear()
        ax.set_title('Risk Profile by Driver')

        # Stacked bar plot with color coding
        bottoms = np.zeros(len(risk_profile_counts))
        for i, (name, values) in enumerate(risk_profile_counts.items()):
            ax.bar(x, values, bottom=bottoms, label=name, color=colors[name], width=bar_width)
            bottoms += values

            # Add annotations
            for j, value in enumerate(values):
                if value > 0:  # Only annotate non-zero values
                    ax.text(j, bottoms[j] - (value / 2), f'{value:.0f}', ha='center', va='center', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(risk_profile_counts.index, rotation=45)
        ax.set_xlabel('Driver')
        ax.set_ylabel('Count')
        ax.legend(fontsize=7)

    elif graph_type == 'line':
        # Line Plot for Satisfaction Percentages
        sns.lineplot(data=df, x='Drivers', y='Satisfaction %', ax=ax, marker='o')
        ax.set_title('Satisfaction % Trend by Driver')
        ax.set_xlabel('Driver')
        ax.set_ylabel('Satisfaction %')
        ax.tick_params(axis='x', rotation=70)


    elif graph_type == 'question-wise':

        # Clear the axis for a fresh plot

        ax.clear()

        # Apply seaborn style for a modern look

        sns.set(style="whitegrid")

        # Sort the DataFrame by 'Satisfaction %'

        plt_df = df.sort_values(by='Satisfaction %', ascending=False)

        # Replace questions with 'Q1', 'Q2', etc.

        question_labels = ['Q' + str(i + 1) for i in range(len(plt_df))]

        # Set up the horizontal bar graph with seaborn color palette

        colors = sns.color_palette('coolwarm', len(plt_df))

        bars = ax.barh(question_labels, plt_df['Satisfaction %'], color=colors)

        # Connect the hover function to the motion_notify_event

        fig.canvas.mpl_connect('motion_notify_event', lambda event: on_hover(event, bars, plt_df['Questions'].tolist(), ax, fig, canvas))

        ax.set_xlabel('Satisfaction %')

        ax.set_title('Satisfaction % by Question')

        # Inverting the y-axis to have the highest satisfaction at the top

        ax.invert_yaxis()

        # Adding a dotted line for the 80% threshold across all bars

        ax.axvline(80, color='green', linestyle='--', linewidth=1)

        # Adding data labels

        for bar in bars:
            width = bar.get_width()

            label_x_pos = width - 5 if width > 5 else width + 1

            ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width}%', va='center',

                    ha='right' if width > 5 else 'left')

        # Adjust the layout to account for the bar labels and the new line

        fig.tight_layout()

        canvas.draw()

    else:
        raise ValueError("Invalid graph type. Choose 'bar', 'stacked', or 'line'.")

    # Adjust layout and draw the canvas
    fig.tight_layout()
    canvas.draw()



def create_window_with_canvases(canvas_list):
    global page3 , page3_layout
    
    def clear_layout(layout):
        """Remove all widgets and items from a layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
            elif child.layout():
                clear_layout(child.layout())
            del child

    clear_layout(page3_layout)


    for canvas in canvas_list:
            scroll_area = QScrollArea()
            scroll_area.setWidget(canvas)
            scroll_area.setWidgetResizable(True)
            page3_layout.addWidget(scroll_area)



    page3.update()



CSAT_data = []
def Load_SSAT() :
    global CSAT_data
    CSAT_data.clear()
    
    required_headings = ["Drivers"]
    if toggleSwitch.isChecked():
        # Get common table names from both paths
        common_table_names = None
        for path in SATFile2ComparePaths:

            data, table_names = find_tables_with_headings(path, required_headings)
            CSAT_data.append(data)

            if common_table_names is None:
                common_table_names = set(table_names)
            else:
                common_table_names.intersection_update(table_names)

        table_names = list(common_table_names)

        # Clear and fill the criteria combo box
        CSAT_class_criteria.clear()
        CSAT_class_criteria.addItems(table_names)
        return

    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(None, "Open Excel File", "",\
                                               "Excel Files (*.xlsx *.xls)")
    if not("AT_" in file_path.upper()):
        print("Error: Wrong file selected")
        return

    required_headings = [ "Drivers"]

    # Find the tables
    CSAT_data,table_names = find_tables_with_headings(file_path, required_headings)


    CSAT_class_criteria.clear()
    CSAT_class_criteria.addItems(table_names)


def UpdateAnalysis_CSAT():
    if not CSAT_data:
        return

    canvases = []
    if toggleSwitch.isChecked():
        Criteria = CSAT_Analysis_criteria.currentText()
        class_chosen = CSAT_class_criteria.currentText()
        index = CSAT_class_criteria.findText(class_chosen)

        for data in CSAT_data:
            df = data[index]
            canvas = FigureCanvas()

            if Criteria == "Avg Satisfaction % by Driver":
                plot_satisfaction_analysis(df, canvas, 'pie')
                tab_widget.setCurrentIndex(2)
            elif Criteria == "Risk Profiles":
                plot_satisfaction_analysis(df, canvas, 'stacked')
                tab_widget.setCurrentIndex(2)
            elif Criteria == "Satisfaction % trend":
                plot_satisfaction_analysis(df, canvas, 'line')
                tab_widget.setCurrentIndex(2)
            elif Criteria == "question-wise":
                plot_satisfaction_analysis(df, canvas, "question-wise")
                tab_widget.setCurrentIndex(2)
            elif Criteria == "Audit Alerts":
                if "OFI/Strength" in df.columns:
                    filtered_data = df[df["OFI/Strength"] == "OFI"]
                elif "Strength/OFI" in df.columns:
                    filtered_data = df[df["Strength/OFI"] == "OFI"]

                create_table(filtered_data)
                tab_widget.setCurrentIndex(3)  # Switch to the tab with the table

            canvas.figure.tight_layout()
            canvases.append(canvas)


        def add_canvases_to_existing_app():
            global  canvas_widget
            create_window_with_canvases(canvases)
            main_window.update()

        add_canvases_to_existing_app()
    else:
        Criteria = CSAT_Analysis_criteria.currentText()
        class_chosen = CSAT_class_criteria.currentText()
        index = CSAT_class_criteria.findText(class_chosen)
        df = CSAT_data[index]

        if Criteria == "Avg Satisfaction % by Driver":
            plot_satisfaction_analysis(df, canvas3, 'pie')
            tab_widget.setCurrentIndex(2)
        elif Criteria == "Risk Profiles":
            plot_satisfaction_analysis(df, canvas3, 'stacked')
            tab_widget.setCurrentIndex(2)
        elif Criteria == "Satisfaction % trend":
            plot_satisfaction_analysis(df, canvas3, 'line')
            tab_widget.setCurrentIndex(2)
        elif Criteria == "question-wise":
            plot_satisfaction_analysis(df, canvas3, "question-wise")
            tab_widget.setCurrentIndex(2)
        elif Criteria == "Audit Alerts":
            if "OFI/Strength" in df.columns:
                filtered_data = df[df["OFI/Strength"] == "OFI"]
            elif "Strength/OFI" in df.columns:
                filtered_data = df[df["Strength/OFI"] == "OFI"]

            create_table(filtered_data)
            tab_widget.setCurrentIndex(3)  # Switch to the tab with the table


        canvas3.figure.tight_layout()
        create_window_with_canvases([canvas3])
        main_window.update()


CSATOptions = ["Avg Satisfaction % by Driver",\
                          "Risk Profiles",
                          "Satisfaction % trend",
                          "Audit Alerts",'question-wise']

CSAT_Analysis_criteria = QComboBox()
CSAT_Analysis_criteria.addItems(CSATOptions)
filter_layoutCSAT.addRow(load_excel_CSAT_button)

CSAT_class_criteria = QComboBox()
CSAT_class_criteria.addItems(Class_groups)
filter_layoutCSAT.addRow(load_excel_CSAT_button)

load_excel_CSAT_button.clicked.connect(Load_SSAT)
filter_layoutCSAT.addRow( apply_comparison_buttonCSAT)
filter_layoutCSAT.addRow( AnalyseNPSButton)
filter_layoutCSAT.addRow("Class Group", CSAT_class_criteria)
filter_layoutCSAT.addRow("Analysis Criteria:", CSAT_Analysis_criteria)

filter_layoutCSAT.addRow( apply_filter_buttonCSAT)
apply_filter_buttonCSAT.clicked.connect(UpdateAnalysis_CSAT)
apply_comparison_buttonCSAT.setVisible(False)


analysis_layout_filters_disp.addWidget(filter_group)
analysis_layout_filters_disp.addWidget(filter_groupPV2)
analysis_layout_filters_disp.addWidget(filter_groupCSAT)


# Spacer to create space between the widgets
spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
left_layout.addItem(spacer)

# Info label
info_label = QLabel("Welcome to the Data Visualization Notebook")
info_label.setAlignment(Qt.AlignCenter)
info_label.setStyleSheet(
    "QLabel { color: #007ACC; font-size: 18px; }"
)



# Tab Widget
tab_widget = QTabWidget(main_window)
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

def create_visualization_page2(pname):
    page = QWidget()
    tab_widget.addTab(page, pname)
    return page

def create_visualization_page(pname, canvas):
    page = QWidget()
    layout = QVBoxLayout(page)

    # Create a QScrollArea for the canvas
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setWidget(canvas)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    # Add the scroll area to the page layout
    layout.addWidget(scroll_area)
    page.setLayout(layout)

    # Add the page to the tab widget
    tab_widget.addTab(page, pname)
    return page


canvas1 = FigureCanvas()
canvas2 = FigureCanvas()
canvas3 = FigureCanvas()
# Create visualization pages
page1 = create_visualization_page("1.0 Audit Tracking_GSF",canvas1)
page2 = create_visualization_page("2.0 Pv2 Reports",canvas2)

page3 = QWidget()  # Initialize page3 without canvas
page3_layout = QHBoxLayout(page3)
tab_widget.addTab(page3, "3.0 SAT Analysis")

page4 = create_visualization_page2("4.0 Audit Alerts")





scrollArea = QScrollArea(page4)  # The scroll area will be added to 'page4'
scrollArea.setWidgetResizable(True)
scrollAreaWidgetContents = QWidget()  # This is the container for your tables
scrollArea.setWidget(scrollAreaWidgetContents)  # Add the container to the scroll area
tablesLayout = QVBoxLayout(scrollAreaWidgetContents)  # The layout for your tables
scrollAreaWidgetContents.setLayout(tablesLayout)


# Add the scroll area to the page4's layout
page4_layout = QVBoxLayout(page4)  # This assumes you have a layout set for page4 already
page4_layout.addWidget(scrollArea)

existing_table_hashes = set()
all_tables = []  # To store all the tables created


def create_table(df, heading="Sample heading"):
    global existing_table_hashes

    # Round numeric values to 2 decimal places
    df = df.applymap(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

    # Generate a hash for the new dataframe
    df_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    # Check if the hash is in the set of existing hashes
    if df_hash in existing_table_hashes:
        print("Table already exists. Skipping.")
        return
    else:
        existing_table_hashes.add(df_hash)

    # If the code reaches here, it means the table is new and should be added
    table = QTableWidget()
    num_rows, num_cols = df.shape
    table.setRowCount(num_rows)
    table.setColumnCount(num_cols)
    columns = list(df.columns)
    table.setHorizontalHeaderLabels([str(x) for x in columns])

    for i in range(num_rows):
        for j in range(num_cols):
            item = QTableWidgetItem(str(df.iloc[i, j]))
            table.setItem(i, j, item)

    # Add the new table to the tablesLayout, which is inside the scrollAreaWidgetContents
    tablesLayout.addWidget(table)

    # Store the table, its columns, and the heading for HTML generation
    all_tables.append((table, columns, heading))

    # After adding the table, adjust its minimum height to ensure it doesn't shrink smaller than its content
    table.setMinimumHeight(table.verticalHeader().length() + table.horizontalHeader().height() + 10)


def generate_html_report():
    # Open a dialog to choose where to save the HTML report
    save_path, _ = QFileDialog.getSaveFileName(None, "Save HTML Report", "", "HTML Files (*.html)")
    if not save_path:
        return

    # Start writing the HTML content with a modern look
    html_content = """
    <html>
    <head>
    <title>GSF Audit Alerts</title>
    <style>
        body {font-family: Arial, sans-serif; margin: 20px;}
        h1 {text-align: center; color: #333;}
        h2 {color: #555; margin-top: 30px;}
        table {width: 100%; border-collapse: collapse; margin-top: 10px;}
        th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}
        th {background-color: #f2f2f2; color: #333;}
        tr:nth-child(even) {background-color: #f9f9f9;}
        tr:hover {background-color: #f1f1f1;}
    </style>
    </head>
    <body>
    <h1>GSF Audit Alerts</h1>
    """

    for idx, (table, columns, heading) in enumerate(all_tables):
        html_content += f"<h2>{heading}</h2>"
        html_content += "<table><tr>"

        # Add column headers
        for col in columns:
            html_content += f"<th>{col}</th>"
        html_content += "</tr>"

        # Add table data
        for row in range(table.rowCount()):
            html_content += "<tr>"
            for col in range(table.columnCount()):
                item = table.item(row, col)
                html_content += f"<td>{item.text() if item else ''}</td>"
            html_content += "</tr>"

        html_content += "</table><br>"

    html_content += "</body></html>"

    # Write the HTML content to a file
    with open(save_path, "w") as html_file:
        html_file.write(html_content)

    print(f"HTML report generated and saved to {save_path}")


# Example usage: Hook up the generate_html_report to a button click
generate_report_button = QPushButton("Generate HTML Report")
generate_report_button.clicked.connect(generate_html_report)

# Assume tablesLayout is a QVBoxLayout and you add the button to it
tablesLayout.addWidget(generate_report_button)

class DummyGraph:
    def __init__(self, figure_canvas, width=0.4, height=0.2):
        self.figure_canvas = figure_canvas
        self.fig = self.figure_canvas.figure
        self.ax = self.fig.add_subplot(111)

        self.create_logo()
        self.resize(width, height)

    def create_logo(self):
        # Create some dummy data for different types of graphs
        x = list(range(1, 21))
        y1 = [random.randint(50, 150) for _ in range(20)]
        y2 = [random.randint(0, 200) for _ in range(20)]
        y3 = [random.randint(20, 200) for _ in range(20)]

        sns.set(style="whitegrid")  # Use a white grid style
        colors = sns.color_palette("Set2")

        # Create different types of plots and overlay them
        sns.lineplot(x=x, y=y1, label="Sample A", ax=self.ax, color=colors[0])
        sns.scatterplot(x=x, y=y2, label="Sample B", ax=self.ax, color=colors[0])
        sns.barplot(x=x, y=y3, label="Sample C", ax=self.ax, color=colors[0])

        title_font = {'size': 20}  # Increase title size and make it bold
        self.ax.axis("off")
        self.ax.set_title("GSF Data Analytics", fontdict=title_font,
                          color=colors[0])  # Customize title

        self.ax.legend(frameon=False, loc="upper right")  # Customize the legend
        sns.despine()  # Remove top and right spines

    def resize(self, width, height):
        box = self.ax.get_position()
        self.ax.set_position([0.5 - width / 2, 0.5 - height / 2, width, height])



DummyGraph(canvas1)
DummyGraph(canvas2)
# Add canvases to visualization pages
page1_layout = QVBoxLayout(page1)
page1_layout.addWidget(canvas1)

page2_layout = QVBoxLayout(page2)
page2_layout.addWidget(canvas2)


# page3_layout = QVBoxLayout(page3)
# page3_layout.addWidget(canvas3)

# Connect signals to functions
load_excel_button.clicked.connect(load_excel)
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
    fig.tight_layout()
    canvas2.draw()
    main_window.update()

def plot_Criteria_parameters_line(df, months_columns, index_column):
    df.reset_index(inplace=True, drop=True)
    index_to_delete = df[df['Unit of measurement'] == 'Sub Total'].index

    fig = canvas2.figure
    ax = fig.add_subplot(111)

    # Create a new figure
    ax.clear()

    # Filter the DataFrame to keep only the specified columns of interest
    df = df.replace("NA", 0)
    df = df.fillna(0)
    df = df.iloc[0:index_to_delete[0]:]

    # Transpose the DataFrame to have months as the index and parameters as columns
    df = df.set_index(index_column)
    columns_of_interest = months_columns
    df = df.iloc[:, columns_of_interest]
    df = df.T
    
    # Plot a line for each column on the subplot with different line styles and colors
    line_styles = ['-', '--', '-.', ':']  # Define line styles for variety
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define colors for variety

    for i, column in enumerate(df.columns):
        line_style = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]

        ax.plot(df.index, df[column], marker='o', linestyle=line_style, label=column, color=color)

        # Annotate data points with their values
        for x, y in zip(df.index, df[column]):
            ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=color)

    # Customize the plot
    ax.set_title('Line Plot Example')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True)

    # Display the legend with a smaller font size
    ax.legend(loc='upper right', prop={'size': 8})

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Add grid lines to the plot
    ax.grid(True, linestyle='--', alpha=0.6)

    # Add a background color to the plot
    ax.set_facecolor('#f0f0f0')

    # Customize the plot's appearance further as needed
    fig.tight_layout()
    # Finally, redraw the canvas
    canvas2.draw()

def plot_monthly_stats_stacked(df, months_columns, index_column):
    # Filter the DataFrame to keep only the specified columns of interest
    df = df.replace("NA", 0)
    df = df.fillna(0)

    # Transpose the DataFrame to have months as the index and parameters as columns
    df = df.set_index(index_column)
    columns_of_interest = months_columns
    df = df.iloc[:, columns_of_interest]
    df = df.T

    # Define colors for different assessment categories
    colors = ['#FF5733', '#33FF57', '#3398FF', '#FF33C2']

    # Get the figure and axis from the canvas
    fig = canvas2.figure
    ax = fig.add_subplot(111)

    # Create a new figure
    ax.clear()
    ax.set_title('Monthly total of score in each Criteria')

    # Set the bar width
    bar_width = 0.1

    # Set the x-axis positions for bars
    x = np.arange(len(df.index))

    # Define spacing between groups
    group_spacing = 1.2

    # Set font size for bar values
    font_size = 7  # Adjust the font size as needed


    # Loop through the parameters and create the grouped bars with spacing
    for i, parameter in enumerate(df.columns):
        ax.bar(x + (i - 1.5) * bar_width * group_spacing, df[parameter], width=bar_width, label=parameter)


        # Add bar values on the bars with reduced font size
        for j, value in enumerate(df[parameter]):
            ax.text(x[j] + (i - 1.5) * bar_width * group_spacing, value, "{:.1f}".format(value), ha='center', va='bottom',
                    fontsize=font_size)

    # Set x-axis labels based on negative column indices
    months = list(df.index)
    ax.set_xlabel('Months')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(months)

    # Create the legend
    ax.legend(fontsize = 7)
    fig.tight_layout()
    # Show the plot
    canvas2.draw()




Campus_name = ''
filtered_rows = pandas.DataFrame()
df= pandas.DataFrame()
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

    load_excel_Pv2_button.setStyleSheet(
        "QPushButton {"
        "background-color: #097969;"
        "color: white;"
        "border: none;"
        "border-radius: 4px;"
        "padding: 10px 20px;"
        "font-size: 16px;"
        "}"
        "QPushButton:hover {"
        "background-color: #50C878;"
        "}"
        "QPushButton:pressed {"
        "background-color: #50C878;"
        "}"
    )



load_excel_Pv2_button.clicked.connect(load_pv2_excel)
SATFile2ComparePaths = []

class SATFile2Compare(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel File Selector")
        self.setGeometry(100, 100, 450, 600)

        self.central_widget = QWidget(main_window)
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()

        # Create horizontal layout for two sections
        self.sections_layout = QHBoxLayout()

        # Create layout for two sections
        self.section1_layout = QVBoxLayout()
        self.section2_layout = QVBoxLayout()

        # Labels and ComboBoxes for file components for both sections
        self.labels = ["Analysis Type", "Brand", "Campus", "Country", "Month", "Year"]
        self.comboboxes1 = {label: QComboBox() for label in self.labels}
        self.comboboxes2 = {label: QComboBox() for label in self.labels}

        # Populate layouts for both sections
        for label in self.labels:
            self.section1_layout.addWidget(QLabel(label))
            self.section1_layout.addWidget(self.comboboxes1[label])
            self.section2_layout.addWidget(QLabel(label))
            self.section2_layout.addWidget(self.comboboxes2[label])

        self.sections_layout.addLayout(self.section1_layout)
        self.sections_layout.addLayout(self.section2_layout)

        self.main_layout.addLayout(self.sections_layout)

        # Select Folder button for common folder
        self.select_folder_button = QPushButton("Select Folder")
        self.main_layout.addWidget(self.select_folder_button)

        # Button to migrate to the next screen
        self.next_screen_button = QPushButton("Save")
        self.main_layout.addWidget(self.next_screen_button)

        self.central_widget.setLayout(self.main_layout)

        self.folder_path = ""
        self.saved_comboboxes1 = {}
        self.saved_comboboxes2 = {}

        self.select_folder_button.clicked.connect(self.select_folder)
        self.next_screen_button.clicked.connect(self.check_files_and_migrate)

        # Apply modern black and off-white theme using stylesheets
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: Arial;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
            }
            QComboBox {
                background-color: #ffffff;
                border: 1px solid #333333;
                padding: 5px;
                color: #333333;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
            QPushButton:pressed {
                background-color: #111111;
            }
        """)

    def select_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(None, "Select Folder")
        if self.folder_path:
            file_list = [f for f in os.listdir(self.folder_path) if f.endswith('.xlsx')]
            components = {"Analysis Type": set(), "Brand": set(), "Campus": set(), "Country": set(), "Month": set(), "Year": set()}

            for file in file_list:
                try:
                    analysis_type, brand, campus, country, month, year_with_ext = file.split('_')
                    year = year_with_ext.split('.')[0]
                    components["Analysis Type"].add(analysis_type)
                    components["Brand"].add(brand)
                    components["Campus"].add(campus)
                    components["Country"].add(country)
                    components["Month"].add(month)
                    components["Year"].add(year)
                except ValueError:
                    continue

            for label in self.labels:
                self.comboboxes1[label].clear()
                self.comboboxes1[label].addItems(sorted(components[label]))
                if label in self.saved_comboboxes1:
                    self.comboboxes1[label].setCurrentText(self.saved_comboboxes1[label])

                self.comboboxes2[label].clear()
                self.comboboxes2[label].addItems(sorted(components[label]))
                if label in self.saved_comboboxes2:
                    self.comboboxes2[label].setCurrentText(self.saved_comboboxes2[label])

    def check_files_and_migrate(self):
        global SATFile2ComparePaths

        selected_components1 = [self.comboboxes1[label].currentText() for label in self.labels]
        file_name1 = "_".join(selected_components1) + ".xlsx"

        selected_components2 = [self.comboboxes2[label].currentText() for label in self.labels]
        file_name2 = "_".join(selected_components2) + ".xlsx"

        file1_exists = os.path.isfile(os.path.join(self.folder_path, file_name1))
        file2_exists = os.path.isfile(os.path.join(self.folder_path, file_name2))

        if file1_exists and file2_exists:
            SATFile2ComparePaths = []
            print(f"Both files exist: {file_name1} and {file_name2}")
            SATFile2ComparePaths = [os.path.join(self.folder_path, file_name1), os.path.join(self.folder_path, file_name2)]
            Load_SSAT()

            # Save the current selections
            for label in self.labels:
                self.saved_comboboxes1[label] = self.comboboxes1[label].currentText()
                self.saved_comboboxes2[label] = self.comboboxes2[label].currentText()

            # Close the window
            self.close()
        else:
            error_message = "The following files do not exist:\n"
            if not file1_exists:
                error_message += f"{file_name1}\n"
            if not file2_exists:
                error_message += f"{file_name2}\n"
            QMessageBox.critical(self, "Error", error_message)


def open_file_selector():
        global file_selector_window
        file_selector_window = SATFile2Compare()
        file_selector_window.show()


apply_comparison_buttonCSAT.clicked.connect(open_file_selector)

class ToggleSwitch(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(QSize(150, 30))
        self._checked = False
        self.setCursor(Qt.PointingHandCursor)

    def sizeHint(self):
        return QSize(60, 30)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked):
        self._checked = checked
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._checked = not self._checked
            self.update()
            self.clicked.emit(self._checked)

    def paintEvent(self, event):
        rect = self.rect()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the background
        if self._checked:
            painter.setBrush(QBrush(QColor(144, 238, 144).darker(120)))
            painter.drawRoundedRect(0, 0, rect.width(), rect.height(), rect.height() / 2, rect.height() / 2)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(QRect(rect.width() - rect.height(), 0, rect.height(), rect.height()))
            painter.drawText(rect, Qt.AlignCenter, "COMPARISON")
        else:
            painter.setBrush(QBrush(QColor(211, 211, 211)))
            painter.drawRoundedRect(0, 0, rect.width(), rect.height(), rect.height() / 2, rect.height() / 2)
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(QRect(0, 0, rect.height(), rect.height()))
            painter.drawText(rect, Qt.AlignCenter, "SINGLE")

    def sizeHint(self):
        return QSize(60, 30)

def Launch_NPS_analysis():
    file_path = filedialog.askopenfilename(
        title="Select a raw survey response file",
        filetypes=(("Excel files", "*.xlsx;*.xls"), ("CSV files", "*.csv"), ("All files", "*.*"))
    )
    # Read the Excel file
    dfNPS = pd.read_excel(f"{file_path}")

    # Apply a modern style to the plots
    mplstyle.use('ggplot')

    # Function to filter questions containing the word "how"
    def filter_questions(questions):
        return [q for q in questions if re.search(r'\bhow\b', q, re.IGNORECASE)]

    # Function to wrap the title if it's too long
    def wrap_title(title, width=60):
        return "\n".join(textwrap.wrap(title, width))

    # Function to update the statistics and plot based on the selected question
    def update_stats_and_plot(selected_question):
        # Clear the previous plots and table
        for widget in plot_area_frame.winfo_children():
            widget.destroy()

        for widget in stats_area_frame.winfo_children():
            widget.destroy()

        for widget in table_area_frame.winfo_children():
            widget.destroy()

        # Select specific columns and slice the DataFrame
        dfNPS_filtered = dfNPS[["Please select your child's year level?", selected_question]].iloc[1:, :]

        # Transform all cells to uppercase and strip whitespace
        dfNPS_filtered = dfNPS_filtered.applymap(lambda x: str(x).upper().strip())

        # Define the common mapping for replacement
        replacement_mapping = {
            'VERY UNLIKELY': 1,
            'UNLIKELY': 2,
            'NEITHER LIKELY NOR UNLIKELY': 3,
            'LIKELY': 4,
            'VERY LIKELY': 5,
            'VERY DISSATISFIED': 1,
            'DISSATISFIED': 2,
            'NEITHER SATISFIED NOR DISSATISFIED': 3,
            'SATISFIED': 4,
            'VERY SATISFIED': 5
        }

        # Replace the values in the DataFrame
        dfNPS_filtered[selected_question] = dfNPS_filtered[selected_question].replace(replacement_mapping)

        # Group by the first column and count the occurrences of each value in the second column
        counts = dfNPS_filtered.groupby("Please select your child's year level?")[selected_question].value_counts().unstack(
            fill_value=0)

        # Add a total column
        counts['Total'] = counts.sum(axis=1)

        # Add a grand total row
        counts.loc['Grand Total'] = counts.sum()

        # Rename the index for clarity
        counts.index.name = 'Grade'
        counts.columns.name = 'Rating'

        # Rename columns to 1, 2, 3, 4, 5
        counts.columns = [1, 2, 3, 4, 5, 'Total']

        # Calculate Total Opportunity
        counts['Total Opportunity'] = counts['Total'] * 5

        # Calculate weighted scores for each grade
        weighted_scores = counts.apply(lambda row: row[1] * 1 + row[2] * 2 + row[3] * 3 + row[4] * 4 + row[5] * 5, axis=1)

        #counts['Weighted scores'] = weighted_scores
        # Calculate Satisfaction Rate
        counts['Satisfaction Rate'] = round((weighted_scores / counts['Total Opportunity']) * 100, 2)

        # Extract numeric part from the grade and sort based on that
        counts = counts.reset_index()
        counts['Grade Number'] = counts['Grade'].apply(
            lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'))
        counts = counts.sort_values('Grade Number').set_index('Grade').drop(columns=['Grade Number'])

        print(counts)

        # Get the grand total counts
        grand_totals = counts.loc['Grand Total']

        # Calculate weighted counts for NPS
        weighted_promoters = grand_totals[4] * 4 + grand_totals[5] * 5
        weighted_detractors = grand_totals[1] * 1 + grand_totals[2] * 2
        passives = grand_totals[3] * 3

        # Calculate individual counts for promoters and detractors
        individual_promoters = grand_totals[4] + grand_totals[5]
        individual_detractors = grand_totals[1] + grand_totals[2]

        # Distribute passives equally between promoters and detractors
        weighted_promoters += passives // 2
        weighted_detractors += passives // 2

        # In case of an odd number of passives, add the remaining 1 to promoters
        if passives % 2 != 0:
            weighted_promoters += 3

        # Calculate the Net Promoter Score (NPS)
        net_promoter_score = weighted_promoters - weighted_detractors

        # Calculate the NPS percentage
        total_responses = grand_totals['Total'] * 5
        nps_percentage = round((net_promoter_score / total_responses) * 100, 2)


        # Recreate the labels to display the statistics
        global weighted_promoters_label, weighted_detractors_label, passives_label, net_promoter_score_label, nps_percentage_label

        weighted_promoters_label = tk.Label(stats_area_frame, text=f"Weighted Promoters: {weighted_promoters} (Count: {individual_promoters})")
        weighted_promoters_label.pack()

        weighted_detractors_label = tk.Label(stats_area_frame, text=f"Weighted Detractors: {weighted_detractors} (Count: {individual_detractors})")
        weighted_detractors_label.pack()

        passives_label = tk.Label(stats_area_frame, text=f"Passives: {passives}")
        passives_label.pack()

        net_promoter_score_label = tk.Label(stats_area_frame, text=f"Net Promoter Score: {net_promoter_score}")
        net_promoter_score_label.pack()

        nps_percentage_label = tk.Label(stats_area_frame, text=f"NPS: {nps_percentage}%")
        nps_percentage_label.pack()

        # Plot the weighted statistics
        fig_stats_weighted, ax_stats_weighted = plt.subplots(figsize=(8, 6))
        weighted_stats_data = {
            'Weighted Promoters': weighted_promoters,
            'Weighted Detractors': weighted_detractors,
            'Passives': passives,
            'Net Promoter Score %': nps_percentage
        }
        weighted_bars = ax_stats_weighted.bar(weighted_stats_data.keys(), weighted_stats_data.values(),
                                              color=['green', 'red', 'yellow', 'blue'])

        # Set the title to the selected question
        ax_stats_weighted.set_title(wrap_title(f'Weighted Survey Statistics for: {selected_question}'))
        ax_stats_weighted.set_ylabel('Values')

        # Annotate the weighted statistics bars
        for bar in weighted_bars:
            yval = bar.get_height()
            ax_stats_weighted.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        # Add the weighted statistics plot
        canvas_stats_weighted = FigureCanvasTkAgg(fig_stats_weighted, master=stats_area_frame)
        canvas_stats_weighted.draw()
        canvas_stats_weighted.get_tk_widget().pack(expand=True, fill='both')

        # Plot the individual counts
        fig_stats_individual, ax_stats_individual = plt.subplots(figsize=(8, 6))
        individual_stats_data = {
            'Individual Promoters': individual_promoters,
            'Individual Detractors': individual_detractors
        }
        individual_bars = ax_stats_individual.bar(individual_stats_data.keys(), individual_stats_data.values(),
                                                  color=['lightgreen', 'lightcoral'])

        # Set the title to the selected question
        ax_stats_individual.set_title(f'Individual Counts for: {selected_question}')
        ax_stats_individual.set_ylabel('Counts')

        # Annotate the individual counts bars
        for bar in individual_bars:
            yval = bar.get_height()
            ax_stats_individual.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

        # Add the individual counts plot
        canvas_stats_individual = FigureCanvasTkAgg(fig_stats_individual, master=stats_area_frame)
        canvas_stats_individual.draw()
        canvas_stats_individual.get_tk_widget().pack(expand=True, fill='both')

        # Plot the data as a grouped bar chart with satisfaction rate
        fig, ax1 = plt.subplots(figsize=(14, 8))
        ax1.plot(counts.index, counts['Satisfaction Rate'], color='blue', marker='o', label='Satisfaction Rate')
        ax1.set_ylabel('Satisfaction Rate (%)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.spines['left'].set_color('blue')
        wrapped_title = wrap_title(f'Responses by Grade for: {selected_question}')
        ax1.set_title(wrapped_title)
        plt.xticks(rotation=45)
        # Annotate the points on the Satisfaction Rate line
        for i, value in enumerate(counts['Satisfaction Rate']):
            ax1.annotate(f'{value:.2f}', (counts.index[i], counts['Satisfaction Rate'][i]),
                         textcoords="offset points", xytext=(0, 10), ha='center', color='blue')

        # Plot the Count on the right y-axis
        ax2 = ax1.twinx()

        # Drop unnecessary columns and plot the bar chart
        counts.drop(['Total', 'Total Opportunity', 'Satisfaction Rate'], axis=1).plot(kind='bar', ax=ax2,
                                                                                      color=['red', 'orange', 'yellow',
                                                                                             'lightgreen', 'green'])

        # Set the y-axis label and ensure the scale is properly aligned
        ax2.set_ylabel('Count', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Ensure ax2 scale fits the data range
        ax2.set_ylim(0, counts.drop(['Total', 'Total Opportunity', 'Satisfaction Rate'], axis=1).values.max() * 1.1)

        # Annotate the bars with improved clarity
        for container in ax2.containers:
            ax2.bar_label(container)

        # Set the x-label and adjust the x-ticks
        ax2.set_xlabel('Grade')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

        # Add the new plot
        canvas = FigureCanvasTkAgg(fig, master=plot_area_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill='both')

        # Insert a column for the row headers (index)
        tree = ttk.Treeview(table_area_frame, columns=['Index'] + list(counts.columns), show='headings')

        # Define the index column heading and set its width
        tree.heading('Index', text='Index')
        tree.column('Index', anchor='center', width=50)

        # Define column headings and set column widths
        for col in counts.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor='center', width=100)  # Set a default width of 100 pixels for each column

        # Insert DataFrame rows with the index as the first column
        for index, row in counts.iterrows():
            tree.insert("", "end", values=[index] + list(row))

        # Pack the Treeview widget
        tree.pack(expand=True, fill='both')

    # Create the main window
    root = tk.Tk()
    root.title("Grade wise Survey statistics")

    # Maximize the window
    #root.state('zoomed')

    # Create a horizontal frame at the top of the window
    top_frame = tk.Frame(root)
    top_frame.pack(fill='x', padx=10, pady=5)

    # Create frames for different content areas
    main_frame = tk.Frame(root)
    main_frame.pack(expand=True, fill='both')

    notebook = ttk.Notebook(main_frame)
    notebook.pack(expand=True, fill='both')

    plot_frame = ttk.Frame(notebook)
    stats_frame = ttk.Frame(notebook)
    table_frame = ttk.Frame(notebook)

    notebook.add(plot_frame, text='Visualizations ')
    notebook.add(stats_frame, text='Satisfaction KPIs')
    notebook.add(table_frame, text='Grade wise satisfaction Rate')

    # Create frames within each tab for specific content
    plot_area_frame = tk.Frame(plot_frame)
    plot_area_frame.pack(expand=True, fill='both')

    stats_area_frame = tk.Frame(stats_frame)
    stats_area_frame.pack(expand=True, fill='both')

    table_area_frame = tk.Frame(table_frame)
    table_area_frame.pack(expand=True, fill='both')

    # Create a dropdown to select the question in the top frame
    question_label = tk.Label(top_frame, text="Select Question:")
    question_label.pack(side='left')

    # Filter questions that contain the word "how" (case-insensitive)
    questions = filter_questions(dfNPS.columns[1:])  # Exclude the first column
    question_var = tk.StringVar()
    question_dropdown = ttk.Combobox(top_frame, textvariable=question_var, values=list(questions), width=90)
    question_dropdown.pack(side='left')

    # Bind the update function to the dropdown selection
    question_dropdown.bind("<<ComboboxSelected>>", lambda event: update_stats_and_plot(question_dropdown.get()))

    # Start the Tkinter event loop
    root.mainloop()


def toggleFrame( event):
        global file_selector_window

        if event.button() == Qt.LeftButton:
            toggleSwitch._checked = not toggleSwitch._checked
            toggleSwitch.update()

            if not(apply_comparison_buttonCSAT.isVisible()):
                apply_comparison_buttonCSAT.setVisible(True)
                load_excel_CSAT_button.setVisible(False)

            else:
                apply_comparison_buttonCSAT.setVisible(False)
                load_excel_CSAT_button.setVisible(True)
                try:
                    file_selector_window
                    if file_selector_window is not None:
                        file_selector_window.close()
                        file_selector_window = None
                except : pass

AnalyseNPSButton.clicked.connect(Launch_NPS_analysis)


# Create the toggle switch
toggleSwitch = ToggleSwitch()
toggleSwitch.mousePressEvent = toggleFrame
filter_layoutCSAT.addWidget(toggleSwitch)
# Show the main window and start the application event loop
main_window.show()
sys.exit(app.exec_())
