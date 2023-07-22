from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy  import SQLAlchemy
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import subprocess

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_PATH'] = 16 * 1000 * 1000
app.config["IMAGE_UPLOADS"] = "generated_img/"
app.config["DATASET_FOLDER"] = 'datasets'

db = SQLAlchemy(app)

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return '<Task %r>' % self.id

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        task_content = request.form['content']
        new_task = Todo(content=task_content)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect('/')
        except:
            return 'There was an issue adding your data'

    else:
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template('index.html',uploaded_image = app.config["IMAGE_UPLOADS"] + "image.png" )

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
   
@app.route('/run_script', methods=['POST'])
def run_script():
    selected_option1 = request.form['datasets']
    selected_option2 = request.form['algorithm']
    print(selected_option1)
    print(selected_option2)
    python_bin = "env\Scripts\python"
    dataset_path = app.config['DATASET_FOLDER'] + "\\" + selected_option1 + "\\"
    print(dataset_path)
    if selected_option2 == "pc_gcastle":    
        subprocess.Popen([python_bin, 'test_gcastle.py', dataset_path + selected_option1 + ".csv"])
    elif selected_option2 == "pc_causal":
        subprocess.Popen([python_bin, 'test_causallearn_pc.py', dataset_path + selected_option1 + ".csv"])
    elif selected_option2 == "ges_gcastle":
        subprocess.Popen([python_bin, 'test_gcastle_ges.py', dataset_path + selected_option1 + ".csv"])
    elif selected_option2 == "ges_causal":
        subprocess.Popen([python_bin, 'test_causallearn_ges.py'])
    else:
        print("Option doesn't exist.")
    return 'Script execution initiated.'


if __name__ == "__main__":
    app.run(debug=True)