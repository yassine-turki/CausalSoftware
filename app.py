from flask import Flask, render_template, url_for, send_file, request, redirect, session
from flask_sqlalchemy  import SQLAlchemy
from flask_session import Session
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import subprocess
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_PATH'] = 16 * 1000 * 1000
app.config["IMAGE_UPLOADS"] = "static/"
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
    is_image = False
    if os.path.isfile(app.config["IMAGE_UPLOADS"] + "image.png"):
        is_image = True
    #print(is_image)
    return render_template('index.html',uploaded_image = app.config["IMAGE_UPLOADS"] + "image.png", is_image = is_image)

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
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
      return 'file uploaded successfully'
   
@app.route('/run_script', methods=['GET', 'POST'])
def run_script():
    if request.method == 'GET':
        is_image = False
        if os.path.isfile(app.config["IMAGE_UPLOADS"] + "image.png"):
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

    selected_starting_edge_add = ''
    selected_ending_edge_add = ''
    selected_starting_edge_delete = ''
    selected_ending_edge_delete = ''

    if  'add_edge_start' in request.form:   
        selected_starting_edge_add = request.form['add_edge_start']
        print(selected_starting_edge_add)
    if  'add_edge_end' in request.form: 
        selected_ending_edge_add = request.form['add_edge_end']
    if  'delete_edge_start' in request.form:   
        selected_starting_edge_delete = request.form['delete_edge_start']
        print(selected_starting_edge_delete)
    if  'delete_edge_end' in request.form: 
        selected_ending_edge_delete = request.form['delete_edge_end']
        print(selected_ending_edge_delete)

    if selected_starting_edge_add != '' and selected_ending_edge_add != '':
        userdata["graph_operations"].append({"op": "add", "start": selected_starting_edge_add, "end": selected_ending_edge_add})

    if selected_starting_edge_delete != '' and selected_ending_edge_delete != '':
        userdata["graph_operations"].append({"op": "delete", "start": selected_starting_edge_delete, "end": selected_ending_edge_delete})

    print(userdata)

    python_bin = "env\Scripts\python"
    dataset_path = app.config['DATASET_FOLDER'] + "\\" + selected_dataset + "\\"
    print(dataset_path)

    if userdata["algorithm"] == "pc_gcastle":       
        subprocess.Popen([python_bin, 'generate_graph.py', dataset_path + userdata["dataset"] + ".csv", "pc_gcastle", json.dumps(userdata["graph_operations"])]).wait()
    elif userdata["algorithm"] == "pc_causal":
        subprocess.Popen([python_bin, 'generate_graph.py', dataset_path + userdata["dataset"] + ".csv", "pc_causal", json.dumps(userdata["graph_operations"])]).wait()
    elif userdata["algorithm"] == "ges_gcastle":
        subprocess.Popen([python_bin, 'generate_graph.py', dataset_path + userdata["dataset"] + ".csv", "ges_gcastle", json.dumps(userdata["graph_operations"])]).wait()
    elif userdata["algorithm"] == "ges_causal":
        subprocess.Popen([python_bin, 'generate_graph.py', dataset_path + userdata["dataset"] + ".csv", "ges_causal", json.dumps(userdata["graph_operations"])]).wait()
    else:
        print("Option doesn't exist.")

    return redirect("/run_script")

@app.route('/run_metrics', methods=['POST', 'GET'])
def run_metrics():
    if request.method == 'GET':
        return render_template('metrics.html')
    
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
        print(selected_library_metrics)
    if  'treatment' in request.form:   
        selected_treatment = request.form['treatment']
        userdata["treatment"] = selected_treatment
        print(selected_treatment)
    if  'outcome' in request.form: 
        selected_outcome = request.form['outcome']
        userdata["outcome"] = selected_outcome
        print(selected_outcome)
    if  'estimator' in request.form:   
        selected_estimator_NDE_NIE = request.form['estimator']
        userdata["estimator"] = selected_estimator_NDE_NIE
        print(selected_estimator_NDE_NIE)
    if  'method_name' in request.form:   
        selected_method_name_ATE_ATC_ATT = request.form['method_name']
        userdata["method_name"] = selected_method_name_ATE_ATC_ATT
        print(selected_method_name_ATE_ATC_ATT)

    userdata["graph_operations"].append({"op": "metrics", "start": selected_treatment, "end": selected_outcome, "estimator": selected_estimator_NDE_NIE})
    print(userdata)
    python_bin = "env\Scripts\python"
    dataset_path = app.config['DATASET_FOLDER'] + "\\" + userdata["dataset"] + "\\"
    
    popup_message_metrics = ''

    if userdata["library_metrics"] == "dowhy":    
        proc = subprocess.Popen([python_bin, 'dowhy_file.py', dataset_path + userdata["dataset"] + ".csv", userdata["treatment"], userdata["outcome"], userdata["estimator"], userdata["method_name"]])
        proc.wait()
        with open("error.txt","r") as error_file: 
            popup_message_metrics = "".join(error_file.readlines())
    else:
        print("Option doesn't exist.")
    session["popup_message"] = popup_message_metrics

    return redirect("/run_metrics")

@app.route('/image', methods=['GET'])
def get_image():
    return send_file(app.config["IMAGE_UPLOADS"] + "image.png", mimetype='image/png')

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
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)

