from flask import Flask, render_template, url_for, send_file, request, redirect, session, jsonify
from flask_sqlalchemy  import SQLAlchemy
from flask_session import Session
from datetime import datetime
from werkzeug.utils import secure_filename
from common.load_data import load_and_check_data
import os
import subprocess
import json
import pandas as pd
import pickle
import shutil
import sys

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['UPLOAD_FOLDER'] = 'datasets'
app.config['MAX_CONTENT_PATH'] = 16 * 1000 * 1000
app.config["IMAGE_UPLOADS"] = "static"
app.config["DATASET_FOLDER"] = 'datasets'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)


db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id
    
def get_datasets():
    dataset_folder = app.config["DATASET_FOLDER"]
    datasets = []
    if os.path.exists(dataset_folder) and os.path.isdir(dataset_folder):
        datasets = [folder for folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, folder))]
        if "userdata" in datasets:
            datasets.remove("userdata") 
            datasets.insert(0, "userdata")  # Insert 'userdata' at the beginning of the list

    return datasets

def load_file_extension(data_file_path):
    file_extension = os.path.splitext(data_file_path)[-1]
    return file_extension

def locate_python_bin():
    """
    returns the location of the python_bin file depending on the user's OS
    """
    if os.getenv('VIRTUAL_ENV'):
        virtual_env_path = os.environ.get('VIRTUAL_ENV')
        python_bin = os.path.relpath(sys.executable, virtual_env_path)
        python_bin= os.path.join(os.path.basename(virtual_env_path), python_bin)
    else:
        print("Not in a virtual environment.")
        python_bin = sys.executable
    
    return python_bin

def locate_dataset(selected_dataset):

    dataset_folder = app.config['DATASET_FOLDER']
    dataset_path = os.path.join(dataset_folder, selected_dataset)

    # List files in the dataset folder
    # List all subdirectories within the dataset folder
    subdirectories = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    selected_file = None
    for subdir in subdirectories:
        # List files in each subdirectory
        files_in_subdir = [f for f in os.listdir(os.path.join(dataset_folder, subdir)) if os.path.isfile(os.path.join(dataset_folder, subdir, f))]

        # Filter the file with a name that matches the selected dataset
        selected_file = next((f for f in files_in_subdir if selected_dataset in f), None)

        if selected_file:
            break
    return os.path.join(dataset_path, selected_file)

python_bin = locate_python_bin()




@app.route('/', methods=['GET'])
def index():
    if not session.get("name"):
        # if not there in the session then redirect to the login page
        return redirect("/login")
    
    if "userdata" not in app.config:
        #Initializing the userdata object if nonexistent 
        app.config["userdata"] = {} 
    
    if session["name"] not in app.config["userdata"]:
        #Initializing the userdata object for the current user
        app.config["userdata"][session["name"]] = {}

    # tasks = Todo.query.order_by(Todo.date_created).all()
    datasets = get_datasets()
    is_image = False
    if os.path.isfile(os.path.join(app.config["IMAGE_UPLOADS"], 'image.png')):
        is_image = True
    #print(is_image)
    return render_template('index.html', uploaded_image = os.path.join(app.config["IMAGE_UPLOADS"], 'image.png'), is_image = is_image, datasets = datasets)

@app.route('/delete/<int:id>')
def delete(id):
    task_to_delete = Todo.query.get_or_404(id)

    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect('/')
    except:
        return 'There was a problem deleting that data'

@app.route('/update/<int:id>', methods=['GET', 'POST'])
def update(id):
    task = Todo.query.get_or_404(id)

    if request.method == 'POST':
        task.content = request.form['content']

        try:
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue updating your data'

    else:
        return render_template('update.html', task=task)

@app.route('/upload')
def upload():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        popup_message_upload = ""
        f = request.files['file']
        # Get the original filename
        original_filename = secure_filename(f.filename)
        if original_filename == "":
            print("No file detected, please try again")
            popup_message_upload = "No file detected, please try again" # for error popup
            session["popup_message_upload"] = popup_message_upload
            return redirect("/")
        # Get the file extension
        file_extension = load_file_extension(original_filename) 
        print("FILE EXTENSION", file_extension)
        if file_extension != ".csv" and file_extension !=".txt":
            print("Only csv and txt files are supported")
            popup_message_upload = "Only csv and txt files are supported" # for error popup
            session["popup_message_upload"] = popup_message_upload
            return redirect("/")
        userdata_folder = os.path.join(app.config['UPLOAD_FOLDER'], "userdata")
        if not os.path.exists(userdata_folder):
            os.makedirs(userdata_folder)
        else:
            # Clear the contents of the 'userdata' folder
            for file in os.listdir(userdata_folder):
                file_path = os.path.join(userdata_folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        # Rename the uploaded file to 'userdata' with the original extension
        new_filename = 'userdata' + file_extension
        filepath = os.path.join(userdata_folder, secure_filename(new_filename))
        f.save(filepath)
        return redirect("/")

def optional_parameters(form_value):
    try : 
        if form_value == "pc_stable" or form_value == "pc_mvpc":
            value = request.form.get(form_value)
        else:
            value = request.form[form_value]
        if value == '':
            return 'null'
        else:
            return value
    except Exception as e:
        return 'null'

# Load graph_operations from a pickle file
def load_graph_operations():
    try:
        with open('graph_operations.pkl', 'rb') as pickle_file:
            return pickle.load(pickle_file)
    except FileNotFoundError:
        return []

# Write graph_operations to a pickle file
def write_graph_operations(graph_operations):
    with open('graph_operations.pkl', 'wb') as pickle_file:
        pickle.dump(graph_operations, pickle_file)


@app.route('/run_script', methods=['GET', 'POST'])
def run_script():
    
    if request.method == 'GET':
        is_image = False
        if os.path.isfile(os.path.join(app.config["IMAGE_UPLOADS"], 'image.png')):
            is_image = True
        return render_template("graph.html", is_image = is_image)
    
    if "userdata" not in app.config:
        app.config["userdata"] = {} 

    if session["name"] not in app.config["userdata"]:
        #Initializing the userdata object for the current user
        app.config["userdata"][session["name"]] = {}
    
    userdata = app.config["userdata"][session["name"]]

    if 'datasets' in request.form:
        # if on index.html, read dataset chosen
        selected_dataset = request.form['datasets']
        userdata["dataset"] = selected_dataset
    else:
        # if on graph.html, use saved dataset for user
        selected_dataset = userdata["dataset"]

    if 'algorithm' in request.form:
        # if on index.html, read algorithm chosen
        selected_algorithm = request.form['algorithm']
        userdata["algorithm"] = selected_algorithm
    else:
        # if on graph.html, use saved algorithm for user
        selected_algorithm = userdata["algorithm"]

    if "graph_operations" not in userdata:
        userdata["graph_operations"] = []
        write_graph_operations(userdata["graph_operations"])
    
    print("selected dataset : ",userdata["dataset"] )

    popup_message_graph = '' # for error popup

    selected_starting_edge_add = ''
    selected_ending_edge_add = ''
    selected_starting_edge_delete = ''
    selected_ending_edge_delete = ''

    if  'add_edge_start' in request.form:   
        selected_starting_edge_add = request.form['add_edge_start']
        
    if  'add_edge_end' in request.form: 
        selected_ending_edge_add = request.form['add_edge_end']

    if  'delete_edge_start' in request.form:   
        selected_starting_edge_delete = request.form['delete_edge_start']
        
    if  'delete_edge_end' in request.form: 
        selected_ending_edge_delete = request.form['delete_edge_end']
        

    if selected_starting_edge_add != '' and selected_ending_edge_add != '':
        graph_operations = load_graph_operations()
        # Modify the data (add a new operation)
        new_operation = {"op": "add", "start": selected_starting_edge_add, "end": selected_ending_edge_add}
        graph_operations.append(new_operation)
        # Write the modified data back to the pickle file
        write_graph_operations(graph_operations)
    elif selected_starting_edge_add == '' and selected_ending_edge_add != '':
        popup_message_graph = 'Missing starting node to add path'
    elif selected_starting_edge_add != '' and selected_ending_edge_add == '':
        popup_message_graph = 'Missing ending node to add path'

    if selected_starting_edge_delete != '' and selected_ending_edge_delete != '':
        graph_operations = load_graph_operations()
        # Modify the data (add a new operation)
        new_operation = {"op": "delete", "start": selected_starting_edge_delete, "end": selected_ending_edge_delete}
        graph_operations.append(new_operation)
        # Write the modified data back to the pickle file
        write_graph_operations(graph_operations)

    elif selected_starting_edge_delete == '' and selected_ending_edge_delete != '':
        popup_message_graph = 'Missing starting node to delete path'
    elif selected_starting_edge_delete != '' and selected_ending_edge_delete == '':
        popup_message_graph = 'Missing ending node to delete path'

    userdata["graph_operations"] = load_graph_operations()

    dataset_path = locate_dataset(selected_dataset)

    if userdata["algorithm"] == "pc_gcastle":  
        pc_tiers = optional_parameters("pc_gcastle_tiers")
        subprocess.Popen([python_bin, 'generate_graph.py', dataset_path, "pc_gcastle", json.dumps(userdata["graph_operations"]), optional_parameters("pc_variant"), optional_parameters("pc_alpha"), optional_parameters("pc_ci_test"), pc_tiers]).wait()


    elif userdata["algorithm"] == "pc_causal":

        pc_tiers = optional_parameters("pc_causal_tiers")
        subprocess.Popen([python_bin, 'generate_graph.py', dataset_path, "pc_causal", json.dumps(userdata["graph_operations"]), optional_parameters("pc_alpha_cl"), optional_parameters("pc_indep_test"), json.dumps(optional_parameters("pc_stable")), optional_parameters("pc_uc_rule"), optional_parameters("pc_uc_priority"), json.dumps(optional_parameters("pc_mvpc")), optional_parameters("pc_correction_name"), pc_tiers]).wait()

    elif userdata["algorithm"] == "ges_gcastle":
        subprocess.Popen([python_bin, 'generate_graph.py', dataset_path, "ges_gcastle", json.dumps(userdata["graph_operations"]), optional_parameters("ges_criterion"), optional_parameters("ges_method"), optional_parameters("ges_k"), optional_parameters("ges_N")]).wait()

    elif userdata["algorithm"] == "ges_causal":

        ges_maxP = optional_parameters("ges_maxP")
        ges_parameters_kfold = optional_parameters("ges_parameters_kfold")
        ges_parameters_lambda = optional_parameters("ges_parameters_lambda")
        ges_parameters_dlabel = optional_parameters("ges_parameters_dlabel")
        subprocess.Popen([python_bin, 'generate_graph.py', dataset_path , "ges_causal", json.dumps(userdata["graph_operations"]), optional_parameters("ges_score_func"), ges_maxP, ges_parameters_kfold, ges_parameters_lambda, ges_parameters_dlabel]).wait()

    else:
        print("Option doesn't exist.")
    
    #Error handling 
    
    with open("error.txt","r") as error_file: 
        popup_message_graph += "".join(error_file.readlines())
    # Check if "error.txt" file exists
    if os.path.exists("error.txt"):
        # Open "error.txt" in write mode to clear its content
        with open("error.txt", "w"):
            pass
    session["popup_message_graph"] = popup_message_graph

    return redirect("/run_script")

@app.route('/run_metrics', methods=['POST', 'GET'])
def run_metrics():
    if request.method == 'GET':
        print_message_metrics = None
        #checking if estimates.txt is not empty
        with open("estimates.txt","r") as estimates_file: 
            print_message_metrics = "<br>".join(estimates_file.readlines())

        if os.path.exists("estimates.txt"):
        # Open "estimates.txt" in write mode to clear its content
            with open("estimates.txt", "w"):
                pass
        return render_template('metrics.html', print_message_metrics = print_message_metrics)
        
    
    if "userdata" not in app.config:
        app.config["userdata"] = {} 
    
    userdata = app.config["userdata"][session["name"]]
    selected_library_metrics = ''
    selected_treatment = ''
    selected_outcome = ''
    selected_estimator_NDE_NIE = ''
    selected_method_name_ATE_ATC_ATT = ''

    if 'library_metrics' in request.form:
        # if on index.html, read dataset chosen
        selected_library_metrics = request.form['library_metrics']
        userdata["library_metrics"] = selected_library_metrics
        
    if  'treatment' in request.form:   
        selected_treatment = request.form['treatment']
        userdata["treatment"] = selected_treatment
        
    if  'outcome' in request.form: 
        selected_outcome = request.form['outcome']
        userdata["outcome"] = selected_outcome
        
    if  'estimator' in request.form:   
        selected_estimator_NDE_NIE = request.form['estimator']
        userdata["estimator"] = selected_estimator_NDE_NIE
        
    if  'method_name' in request.form:   
        selected_method_name_ATE_ATC_ATT = request.form['method_name']
        userdata["method_name"] = selected_method_name_ATE_ATC_ATT
        
    graph_operations = load_graph_operations()
    # Modify the data (add a new operation)
    new_operation = {"op": "metrics", "start": selected_treatment, "end": selected_outcome, "estimator": selected_estimator_NDE_NIE}
    graph_operations.append(new_operation)
    # Write the modified data back to the pickle file
    write_graph_operations(graph_operations)
    userdata["graph_operations"] = load_graph_operations()
    
    popup_message_metrics = ''
    dataset_path = locate_dataset(userdata["dataset"])
    if userdata["library_metrics"] == "dowhy":    
        proc = subprocess.Popen([python_bin, 'dowhy_file.py', dataset_path, userdata["treatment"], userdata["outcome"], userdata["estimator"], userdata["method_name"]])
        proc.wait()
        with open("error.txt","r") as error_file: 
            popup_message_metrics = "".join(error_file.readlines())
        # Check if "error.txt" file exists
        if os.path.exists("error.txt"):
            # Open "error.txt" in write mode to clear its content
            with open("error.txt", "w"):
                pass
    else:
        print("Option doesn't exist.")
    session["popup_message"] = popup_message_metrics

    return redirect("/run_metrics")


@app.route('/run_data', methods = ['POST', 'GET'])
def run_data():
    userdata = app.config["userdata"][session["name"]]

    if request.method == 'POST' and 'datasets' in request.form:
        # if on index.html, read dataset chosen
        selected_dataset = request.form['datasets']
        userdata["dataset"] = selected_dataset
    
    else:
        selected_dataset = userdata["dataset"]

    dataset = locate_dataset(selected_dataset)
    df, _, _ = load_and_check_data(dataset, dropna = False, drop_objects = False)
    summary = df.describe(include="all")
    types = df.dtypes.rename('data_type').to_frame().T
    missing_values = df.isnull().any().rename('missing_values').to_frame().T
    result = pd.concat([summary, types, missing_values], axis=0, join="outer")
    summary_stats = result.to_html(classes='table table-striped table-bordered table-sm')
    # summary_stats = pd.concat([df.describe(include="all").T, df.dtypes.rename('data_type')], axis=1)
    # summary_stats= summary_stats.to_html(classes='table table-striped table-bordered table-sm')
    return render_template('summary.html', data_name = selected_dataset, summary_stats=summary_stats)  

@app.route('/image', methods=['GET'])
def get_image():
    return send_file(os.path.join(app.config["IMAGE_UPLOADS"], 'image.png'), mimetype='image/png')

@app.route("/login", methods=["POST", "GET"])
def login():
  # if form is submited
    if request.method == "POST":
        # record the user name
        session["name"] = request.form.get("name")
        #Initializing the userdata object if nonexistent 
        if "userdata" not in app.config:
            app.config["userdata"] = {} 
        #Initializing the userdata object for the current user
        app.config["userdata"][session["name"]] = {}
        # redirect to the main page
        return redirect("/")
    return render_template("login.html")
    #return 'User registered'


@app.route("/logout")
def logout():
    session["name"] = None
    session["popup_message_graph"] = None
    session["popup_message"] = None
    session["popup_message_upload"] = None
    graph_generated_by_user = 'graph_list.pkl'
    graph_hyper_parameters = "graph_hyper_parameters.pkl"
    graph_operations = "graph_operations.pkl"
    data_by_user = "data_file_path.pkl"

    try:
        if os.path.exists(graph_generated_by_user):
            os.remove(graph_generated_by_user)
    except FileNotFoundError:
        pass

    try:
        if os.path.exists(graph_hyper_parameters):
            os.remove(graph_hyper_parameters)
    except FileNotFoundError:
        pass

    try:
        if os.path.exists(graph_hyper_parameters):
            os.remove(graph_hyper_parameters)
    except FileNotFoundError:
        pass    

    try:
        if os.path.exists(data_by_user):
            os.remove(data_by_user)
    except FileNotFoundError:
        pass   

    try:
        if os.path.exists(graph_operations):
            os.remove(graph_operations)
    except FileNotFoundError:
        pass  
    userdata_folder = os.path.join(app.config['UPLOAD_FOLDER'], "userdata")
    if os.path.exists(userdata_folder):
        shutil.rmtree(userdata_folder)

    

    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)

