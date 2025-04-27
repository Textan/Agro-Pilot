# Import Libraries Pre-Installation
import os
import sys
import subprocess
import importlib.metadata

# Dependencies Check and Install Function
required_packages = ['pandas', 'scikit-learn', 'joblib', 'seaborn', 'matplotlib', 'torch', 'transformers', 'datasets', 'accelerate', 'dotenv', 'google-generativeai']
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def check_and_install_packages():
    installed_packages = {pkg.metadata['Name'].lower() for pkg in importlib.metadata.distributions()}
    missing_packages = [pkg for pkg in required_packages if pkg not in installed_packages]
    for package in missing_packages:
        install(package)
check_and_install_packages()

# Load Required Libraries Post-Installation
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from concurrent.futures import ThreadPoolExecutor
from tkinter import Tk, Frame, Label, Entry, Button, StringVar, OptionMenu, messagebox, Text, Scrollbar, Canvas, END
from tkinter.font import Font
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import google.generativeai as genai
import re
from module import messagebox
from tooltips import createToolTipAtGivenPos, deleteToolTip
from dotenv import load_dotenv
import shutil
import pathlib

# Clean Up __pycache__ and .pyc Files
def clean(path="."):
    root = pathlib.Path(path)
    [shutil.rmtree(p) for p in root.rglob("__pycache__")]
    [os.remove(p) for p in root.rglob("*.pyc")]

# Gemini Configuration
load_dotenv()
genai.configure(api_key=os.getenv('google_gen_api_key'))
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(model_name="gemini-1.5-pro-002", generation_config=generation_config)

# Load Dataset
os.chdir(os.path.dirname(os.path.abspath(__file__)))
file_path = 'Data_for_Agro_Pilot.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Error: The file '{file_path}' does not exist")
data = pd.read_csv(file_path)

# Data Preprocessing
month_mapping = {month: index + 1 for index, month in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])}
data['Months'] = data['Months'].map(month_mapping)
data['Avg_pH'] = data[['pH_1', 'pH_2', 'pH_3']].mean(axis=1)
data['Avg_Soil'] = data[['Soil_1', 'Soil_2', 'Soil_3']].apply(lambda x: ','.join(x.unique()), axis=1)
data['Avg_Water'] = data[['Water_1', 'Water_2', 'Water_3']].mean(axis=1)

features = data[['Months', 'Temp', 'Avg_pH', 'Avg_Soil', 'Avg_Water']]
target_crop_1, target_crop_2, target_crop_3 = data['Crop_1'], data['Crop_2'], data['Crop_3']
soil_types = ['Loamy', 'Sandy', 'Clayey', 'Black', 'Red']

# Column Transformer for Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Months', 'Temp', 'Avg_pH', 'Avg_Water']),
        ('soil', OneHotEncoder(categories=[soil_types], handle_unknown='ignore'), ['Avg_Soil'])
    ])

# Define Hyperparameter Grid
rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
}

gb_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.1, 0.01],
    'classifier__max_depth': [3, 5, 7],
}

# Hyperparameter Tuning for Base Models
def tune_base_estimators(features, target):
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])
    rf_random_search = RandomizedSearchCV(rf_pipeline, param_distributions=rf_param_grid, n_iter=10, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1, random_state=42)
    rf_random_search.fit(features, target)
    best_rf = rf_random_search.best_estimator_.named_steps['classifier']

    gb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier(random_state=42))])
    gb_random_search = RandomizedSearchCV(gb_pipeline, param_distributions=gb_param_grid, n_iter=10, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1, random_state=42)
    gb_random_search.fit(features, target)
    best_gb = gb_random_search.best_estimator_.named_steps['classifier']

    return best_rf, best_gb

# Get Best Estimators for Each Crop
best_rf_1, best_gb_1 = tune_base_estimators(features, target_crop_1)
best_rf_2, best_gb_2 = tune_base_estimators(features, target_crop_2)
best_rf_3, best_gb_3 = tune_base_estimators(features, target_crop_3)

# Define and Train Stacking Models
stacking_model_1 = StackingClassifier(estimators=[('rf', best_rf_1), ('gb', best_gb_1)], final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))
stacking_model_2 = StackingClassifier(estimators=[('rf', best_rf_2), ('gb', best_gb_2)], final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))
stacking_model_3 = StackingClassifier(estimators=[('rf', best_rf_3), ('gb', best_gb_3)], final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))

# Define Model Training and Saving Function
def train_and_save_model(features, target, model_name, stacking_model):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', stacking_model)])
    pipeline.fit(features, target)
    joblib.dump(pipeline, os.path.join(pkl_storage_path, model_name), compress=3)
    return pipeline

# Training Models
pkl_storage_path = 'PKL_Storage'
os.makedirs(pkl_storage_path, exist_ok=True)

best_model_pipeline_crop_1 = train_and_save_model(features, target_crop_1, 'Best_Model_Crop_1.pkl', stacking_model_1)
best_model_pipeline_crop_2 = train_and_save_model(features, target_crop_2, 'Best_Model_Crop_2.pkl', stacking_model_2)
best_model_pipeline_crop_3 = train_and_save_model(features, target_crop_3, 'Best_Model_Crop_3.pkl', stacking_model_3)

# Load Models Concurrently
with ThreadPoolExecutor() as executor:
    model_1 = executor.submit(joblib.load, os.path.join(pkl_storage_path, 'Best_Model_Crop_1.pkl'))
    model_2 = executor.submit(joblib.load, os.path.join(pkl_storage_path, 'Best_Model_Crop_2.pkl'))
    model_3 = executor.submit(joblib.load, os.path.join(pkl_storage_path, 'Best_Model_Crop_3.pkl'))
    best_model_pipeline_crop_1 = model_1.result()
    best_model_pipeline_crop_2 = model_2.result()
    best_model_pipeline_crop_3 = model_3.result()

# Crop Prediction Function
def predict_best_crop_with_soil_dependency(month_int, temp, avg_pH, avg_soil, avg_water):
    input_data = pd.DataFrame({
        'Months': [month_int],
        'Temp': [temp],
        'Avg_pH': [avg_pH],
        'Avg_Soil': [avg_soil],
        'Avg_Water': [avg_water]
    })

    prob_crop_1 = best_model_pipeline_crop_1.predict_proba(input_data)[0]
    prob_crop_2 = best_model_pipeline_crop_2.predict_proba(input_data)[0]
    prob_crop_3 = best_model_pipeline_crop_3.predict_proba(input_data)[0]

    confidence_factor = 0.3 * (50 - abs(temp - 25)) / 50 + 0.3 * min(avg_water / 1000, 1) + 0.2 * (10 - abs(avg_pH - 7)) / 10 + 0.2
    prob_crop_1 = [p * confidence_factor for p in prob_crop_1]
    prob_crop_2 = [p * confidence_factor for p in prob_crop_2]
    prob_crop_3 = [p * confidence_factor for p in prob_crop_3]

    crop_list = [(best_model_pipeline_crop_1.classes_[i], prob_crop_1[i]) for i in range(len(best_model_pipeline_crop_1.classes_))]
    crop_list += [(best_model_pipeline_crop_2.classes_[i], prob_crop_2[i]) for i in range(len(best_model_pipeline_crop_2.classes_))]
    crop_list += [(best_model_pipeline_crop_3.classes_[i], prob_crop_3[i]) for i in range(len(best_model_pipeline_crop_3.classes_))]
    
    crop_list.sort(key=lambda x: x[1], reverse=True)
    best_crop, best_prob = crop_list[0]
    second_best_crop, second_best_prob = crop_list[1]

    return best_crop, best_prob * 100, second_best_crop, second_best_prob * 100

# Farmer's Forest TK GUI
class AgroPilot(Tk):
    # Initial Configuration
    def __init__(self):
        super().__init__()
        self.title("Agro-Pilot")
        self.state(newstate="zoomed")
        self.configure(bg="#2e2e2e")
        self.create_widgets()
    
    # Tooltip Utility Method
    def add_tooltip(self, widget, tooltip_id, message):
        widget.bind(
            "<Enter>",
            lambda event: createToolTipAtGivenPos(tooltip_id, self, message, None, event)
        )
        widget.bind(
            "<Leave>",
            lambda event: deleteToolTip(tooltip_id, self)
        )

    # Menu Widgets
    def create_widgets(self):
        header_bg = "#3b3b3b"
        button_bg = "#0F8079"
        text_fg = "white"
        title_font = Font(family="Yu Gothic", size=16, weight="bold")

        sidebar = Frame(self, bg=header_bg, width=200)
        sidebar.pack(side="left", fill="y")
        
        Label(sidebar, text="Menu", font=title_font, fg=text_fg, bg=header_bg).pack(pady=10)
        buttons = ["Crop Prediction", "Data Visualization", "Ask the Expert"]
        for text in buttons:
            b = Button(sidebar, text=text, bg=button_bg, fg="white", font=title_font, width=20, 
                   relief="flat", command=lambda t=text: self.switch_view(t))
            b.pack(pady=5)
        Label(sidebar, text="Credits", font=title_font, fg=text_fg, bg=header_bg, pady=10).pack()
        
        credits_text = [
            "Srijan: Project Head",
            "Atul: UI Design",
            "Viswas: UI Improvements",
            "From: The Hindu Senior Secondary School"
        ]
        
        for text in credits_text:
            Label(sidebar, text=text, font=("Futura", 12, "bold"), fg=text_fg, bg=header_bg).pack(anchor="w")

        header = Frame(self, bg=header_bg, height=50)
        header.pack(side="top", fill="x")
        Label(header, text="Agro-Pilot - Dashboard", font=title_font, fg=text_fg, bg=header_bg).pack(pady=10)
        
        self.content = Frame(self, bg="#2e2e2e")
        self.content.pack(side="right", expand=True, fill="both", padx=10, pady=10)
        self.switch_view("Crop Prediction")
    
    # Function Options
    def switch_view(self, view_name,):  
        for widget in self.content.winfo_children():
            widget.destroy()
        if view_name == "Crop Prediction":
            self.show_crop_prediction()
        elif view_name == "Data Visualization":
            self.show_data_visualization()
        elif view_name == "Ask the Expert":
            self.show_chat_interface()
    # Crop Prediction GUI
    def show_crop_prediction(self):
        title_font = Font(family="Yu Gothic", size=16, weight="bold")
        label_font = Font(family="Yu Gothic", size=12)
        entry_font = Font(family="Yu Gothic", size=12)
        button_font = Font(family="Yu Gothic", size=14, weight="bold")
        result_font = Font(family="Yu Gothic", size=14, weight="bold")
        label_fg = "white"
        entry_bg = "#f0f0f0"
        entry_fg = "black"
        button_bg = "#0F8079"
        button_fg = "white"

        title_frame = Frame(self.content, bg='#3b3b3b')
        title_frame.pack(pady=9, padx=9, fill='x')
        title_label = Label(title_frame, text="Crop Prediction", font=title_font, fg='white', bg='#3b3b3b')
        title_label.pack(pady=9)

        frame = Frame(self.content, bg='#2e2e2e')
        frame.pack(pady=10)

        # Location
        label_location = Label(frame, text="Location:", font=label_font, bg="#2e2e2e", fg="white")
        label_location.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        location_var = StringVar(value="Thanjavur")
        dropdown_location = OptionMenu(frame, location_var, "Thanjavur")
        dropdown_location.config(font=entry_font)
        dropdown_location.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.add_tooltip(dropdown_location, "location_tooltip", "Select your location.")

        # Month
        label_month = Label(frame, text="Month:", font=label_font, bg="#2e2e2e", fg=label_fg)
        label_month.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.month_var = StringVar(self)
        self.month_var.set("Select Month")
        months_list = list(month_mapping.keys())
        dropdown_month = OptionMenu(frame, self.month_var, *months_list)
        dropdown_month.config(font=entry_font)
        dropdown_month.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.add_tooltip(dropdown_month, "month_tooltip", "Select the month for prediction.")

        # Temperature
        label_temp = Label(frame, text="Temperature (C):", font=label_font, bg="#2e2e2e", fg=label_fg)
        label_temp.grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.entry_temp = Entry(frame, font=entry_font, bg=entry_bg, fg=entry_fg)
        self.entry_temp.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        self.add_tooltip(self.entry_temp, "temp_tooltip", "Enter the average temperature in Celsius.")

        # Soil pH Level
        label_pH = Label(frame, text="Soil pH Level:", font=label_font, bg="#2e2e2e", fg=label_fg)
        label_pH.grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.entry_pH = Entry(frame, font=entry_font, bg=entry_bg, fg=entry_fg)
        self.entry_pH.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.add_tooltip(self.entry_pH, "ph_tooltip", "Enter the soil pH level from 4 to 9(e.g., 6.5).")

        # Soil Type
        label_soil = Label(frame, text="Soil Type:", font=label_font, bg="#2e2e2e", fg=label_fg)
        label_soil.grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.soil_var = StringVar(self)
        self.soil_var.set("Select Soil")
        dropdown_soil = OptionMenu(frame, self.soil_var, *soil_types)
        dropdown_soil.config(font=entry_font)
        dropdown_soil.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.add_tooltip(dropdown_soil, "soil_tooltip", "Select the type of soil.")

        # Water Availability
        label_water = Label(frame, text="Water Availability (mm):", font=label_font, bg="#2e2e2e", fg=label_fg)
        label_water.grid(row=5, column=0, sticky="e", padx=5, pady=5)
        self.entry_water = Entry(frame, font=entry_font, bg=entry_bg, fg=entry_fg)
        self.entry_water.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.add_tooltip(self.entry_water, "water_tooltip", "Enter water availability in millimeters.")

        # Buttons
        button_predict = Button(frame, text="Predict Best Crop", command=self.predict_crop, font=button_font, bg=button_bg, fg=button_fg)
        button_predict.grid(row=6, column=0, padx=5, pady=5, sticky="e")

        try_again_button = Button(frame, text="Reset Inputs", command=self.reset_inputs, font=button_font, bg=button_bg, fg=button_fg)
        try_again_button.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        button_exit = Button(frame, text="Exit", command=self.quit, font=button_font, bg=button_bg, fg=button_fg)
        button_exit.grid(row=7, column=0, columnspan=2, pady=10)

    # Error Handling_1
    def validate_inputs(self, month, temp, pH, water):
        if not month or not temp or not pH or not water:
            messagebox.showerror(header="Error", msg="Please Fill All Fields", root=self)
            return False
        if not 0 <= temp <= 50:
            messagebox.showerror(header="Error", msg="Temperature Must be Between 0 and 50°C", root=self)
            return False
        if not 4 <= pH <= 9:
            messagebox.showerror(header="Error", msg="pH Value Must be Between 4 and 9", root=self)
            return False
        if not 0 <= water <= 300:
            messagebox.showerror(header="Error", msg="Water Availability Must be Between 0 and 300 mm", root=self)
            return False
        return True

    # Error Handling_2 and Bar for Crop Scores
    def predict_crop(self):
        month = self.month_var.get()
        temp = self.entry_temp.get()
        pH = self.entry_pH.get()
        soil = self.soil_var.get()
        water = self.entry_water.get()

        if not temp or not pH or not water or month == "Select Month" or soil == "Select Soil":
            messagebox.showerror(header="Error", msg="Please Provide All Required Inputs", root=self)
            return
        try:
            month_int = month_mapping[month]
            temp = float(temp)
            pH = float(pH)
            water = float(water)
        except ValueError:
            messagebox.showerror(header="Error", msg="Invalid Input. Please Enter Numerical Values for Temperature, pH, and Water", root=self)
            return

        if not self.validate_inputs(month, temp, pH, water):
            return

        best_crop, best_prob, second_best_crop, second_best_prob = predict_best_crop_with_soil_dependency(month_int, temp, pH, soil, water)

        canvas_width = 200
        canvas_height = 30
        bar_padding = 5

        if hasattr(self, "try_again_button"):
            self.try_again_button.destroy()
        if hasattr(self, "label_best_crop"):
            self.label_best_crop.destroy()
        if hasattr(self, "label_second_best_crop"):
            self.label_second_best_crop.destroy()
        if hasattr(self, "best_canvas"):
            self.best_canvas.destroy()
        if hasattr(self, "second_best_canvas"):
            self.second_best_canvas.destroy()

        self.label_best_crop = Label(self.content, text=f"Best Crop: {best_crop} ({best_prob:.2f}%)", fg="white", bg="#2e2e2e", font=("Arial", 14))
        self.label_best_crop.pack(pady=(10, 5))

        self.best_canvas = Canvas(self.content, width=canvas_width, height=canvas_height, bg="#2e2e2e", highlightthickness=0)
        self.best_canvas.pack(pady=5)
        best_color = "red" if best_prob < 40 else "green"
        best_bar_width = int((best_prob / 100) * (canvas_width - bar_padding * 2))
        self.best_canvas.create_rectangle(bar_padding, bar_padding, canvas_width - bar_padding, canvas_height - bar_padding, outline="black", width=2)
        self.best_canvas.create_rectangle(bar_padding, bar_padding, best_bar_width + bar_padding, canvas_height - bar_padding, fill=best_color)
        self.best_canvas.create_text(canvas_width / 2, canvas_height / 2, text=f"{best_prob:.2f}%", fill="white", font=("Arial", 12))

        self.label_second_best_crop = Label(self.content, text=f"Second Best Crop: {second_best_crop} ({second_best_prob:.2f}%)", fg="white", bg="#2e2e2e", font=("Arial", 14))
        self.label_second_best_crop.pack(pady=(10, 5))

        self.second_best_canvas = Canvas(self.content, width=canvas_width, height=canvas_height, bg="#2e2e2e", highlightthickness=0)
        self.second_best_canvas.pack(pady=5)
        second_best_color = "red" if second_best_prob < 40 else "green"
        second_best_bar_width = int((second_best_prob / 100) * (canvas_width - bar_padding * 2))
        self.second_best_canvas.create_rectangle(bar_padding, bar_padding, canvas_width - bar_padding, canvas_height - bar_padding, outline="black", width=2)
        self.second_best_canvas.create_rectangle(bar_padding, bar_padding, second_best_bar_width + bar_padding, canvas_height - bar_padding, fill=second_best_color)
        self.second_best_canvas.create_text(canvas_width / 2, canvas_height / 2, text=f"{second_best_prob:.2f}%", fill="white", font=("Arial", 12))

    # Input Reset for Crop Predictor
    def reset_inputs(self):
        self.entry_temp.delete(0, END)
        self.entry_pH.delete(0, END)
        self.entry_water.delete(0, END)
        
        self.month_var.set("Select Month")
        self.soil_var.set("Select Soil")
        
        self.label_best_crop.config(text="")
        self.label_second_best_crop.config(text="")
        
        if hasattr(self, "best_canvas") and self.best_canvas:
            self.best_canvas.destroy()
        if hasattr(self, "second_best_canvas") and self.second_best_canvas:
            self.second_best_canvas.destroy()

    # TK for Graph Options
    def show_data_visualization(self):
        title_font = Font(family="Yu Gothic", size=16, weight="bold")
        title_frame = Frame(self.content, bg='#3b3b3b')
        title_frame.pack(pady=9, padx=9, fill='x')
        title_label = Label(title_frame, text="Data Visualization", font=title_font, fg='white', bg='#3b3b3b')
        title_label.pack(pady=9)

        graph_options = ["Monthly Temperature Trends", "Water Availability by Month", "Soil pH Distribution"]
        self.graph_var = StringVar(value="Select Graph")
        dropdown = OptionMenu(self.content, self.graph_var, *graph_options)
        dropdown.config(bg="#0F8079", fg="white", font=title_font, relief="flat")
        dropdown.pack(pady=10)

        display_button = Button(self.content, text="Display Graph", command=self.display_graph, bg="#0F8079", fg="white", font=title_font)
        display_button.pack(pady=10)

        self.canvas = None

    # Data Visualisation Function using Matplotlib and Seaborn
    def display_graph(self):
        selected_graph = self.graph_var.get()
        if selected_graph == "Select Graph":
            messagebox.showerror(header="Error", msg="Please Select a Graph to Display", root=self)
            return

        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#2e2e2e")
        ax.set_facecolor("#3b3b3b")
        ax.tick_params(axis='x', rotation=45, colors="white")
        ax.tick_params(axis='y', colors="white")

        if selected_graph == "Monthly Temperature Trends":
            ax.set_title("Monthly Temperature Trends", color="white")
            ax.set_xlabel("Month", color="white")
            ax.set_ylabel("Temperature (°C)", color="white")
            sns.lineplot(data=data, x='Months', y='Temp', marker="o", color='coral', ax=ax)

        elif selected_graph == "Water Availability by Month":
            ax.set_title("Water Availability by Month", color="white")
            ax.set_xlabel("Month", color="white")
            ax.set_ylabel("Water Availability (mm)", color="white")
            sns.barplot(data=data, x='Months', y='Avg_Water', palette='Blues', ax=ax)

        elif selected_graph == "Soil pH Distribution":
            ax.set_title("Soil pH Distribution", color="white")
            ax.set_xlabel("Soil pH", color="white")
            ax.set_ylabel("Frequency", color="white")
            sns.histplot(data=data, x='Avg_pH', bins=10, color='lime', kde=True, ax=ax)

        self.canvas = FigureCanvasTkAgg(fig, master=self.content)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # TK for Ask the Expert Function
    def show_chat_interface(self):
        title_font = Font(family="Yu Gothic", size=16, weight="bold")
        label_font = Font(family="Yu Gothic", size=12)
        entry_font = Font(family="Yu Gothic", size=12)
        button_font = Font(family="Yu Gothic", size=14, weight="bold")

        title_frame = Frame(self.content, bg='#3b3b3b')
        title_frame.pack(pady=9, padx=9, fill='x')
        title_label = Label(title_frame, text="Ask the Expert", font=title_font, fg='white', bg='#3b3b3b')
        title_label.pack(pady=9)

        question_frame = Frame(self.content, bg="#2e2e2e")
        question_frame.pack(pady=10, fill="x")
        question_label = Label(question_frame, text="Ask a question:", font=label_font, bg="#2e2e2e", fg='white')
        question_label.pack(anchor="w", padx=10, pady=5)
        
        self.entry_question = Entry(question_frame, font=entry_font, width=70)
        self.entry_question.pack(padx=10, pady=5, fill="x")

        ask_button = Button(question_frame, text="Get Answer", command=self.get_answer, font=button_font, bg="#0F8079", fg="white")
        ask_button.pack(pady=10, padx=10)

        answer_frame = Frame(self.content, bg="#2e2e2e", bd=2, relief="sunken")
        answer_frame.pack(pady=10, fill="both", expand=True)

        answer_label = Label(answer_frame, text="Answer:", font=label_font, bg="#2e2e2e", fg='white')
        answer_label.pack(anchor="w", padx=10, pady=5)

        answer_text_frame = Frame(answer_frame, bg="#2e2e2e")
        answer_text_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.answer_text = Text(answer_text_frame, wrap="word", font=entry_font, bg="#2e2e2e", fg='white', height=10)
        self.answer_text.pack(side="left", fill="both", expand=True)
        
        scrollbar = Scrollbar(answer_text_frame, command=self.answer_text.yview, bg="#2e2e2e")
        scrollbar.pack(side="right", fill="y")
        self.answer_text.config(yscrollcommand=scrollbar.set)
        self.answer_text.config(state="disabled")
    
    # Gemini Answer Formatting
    def insert_formatted_text(self, text):
        self.answer_text.tag_configure("bold", font="Arial 10 bold")
        self.answer_text.tag_configure("italic", font="Arial 10 italic")
        self.answer_text.tag_configure("header", font="Arial 12 bold", underline=1)
        self.answer_text.tag_configure("bullet", font="Arial 10", lmargin1=10, lmargin2=30)
        lines = text.split('\n')

        for line in lines:
            if line.startswith("# "):
                self.answer_text.insert("end", line[2:] + "\n", "header")
            elif line.startswith("• "):
                self.answer_text.insert("end", "• " + line[2:] + "\n", "bullet")
            elif "**" in line:
                segments = re.split(r'(\*\*.*?\*\*)', line)
                for segment in segments:
                    if segment.startswith("**") and segment.endswith("**"):
                        self.answer_text.insert("end", segment[2:-2], "bold")
                    else:
                        self.answer_text.insert("end", segment)
                self.answer_text.insert("end", "\n")
            elif "*" in line:
                segments = re.split(r'(\*.*?\*)', line)
                for segment in segments:
                    if segment.startswith("*") and segment.endswith("*"):
                        self.answer_text.insert("end", segment[1:-1], "italic")
                    else:
                        self.answer_text.insert("end", segment)
                self.answer_text.insert("end", "\n")
            else:
                self.answer_text.insert("end", line + "\n")
        self.answer_text.config(state="disabled")
    
    # Answer Generation and Error Handling for AI
    def get_answer(self):
        question = self.entry_question.get()
        if question:
            self.answer_text.config(state="normal")
            self.answer_text.delete(1.0, "end")
            
            try:
                chat_session = model.start_chat(history=[])
                response = chat_session.send_message(question)
                answer = response.text if hasattr(response, 'text') else "No Response Available."
                
                self.insert_formatted_text(answer)
            except Exception as e:
                self.answer_text.insert("end", f"Error Communicating with AI: {str(e)}")
            self.answer_text.config(state="disabled")
        else:
            self.answer_text.config(state="normal")
            self.answer_text.delete(1.0, "end")
            self.answer_text.insert("end", "Please Enter a Question or Try Again Later.")
            self.answer_text.config(state="disabled")

if __name__ == "__main__":
    clean()
    app = AgroPilot()
    app.mainloop()
