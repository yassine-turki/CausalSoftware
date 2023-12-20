# CausalSoftware


## Getting started
### Using virtualenv

1. Install the virtual environment

We use a module named virtualenv, which is a tool to create isolated Python environments. virtualenv creates a folder that contains all the necessary executables to use the packages that a Python project would need.

```
pip install virtualenv
```

2. Go to the local directory where we want to create the app

```
virtualenv virtualenv_name
```

Then, run the command:

For Windows:
```
.\virtualenv_name\Scripts\activate
```

For Mac and Linux : 
```
source virtualenv_name/bin/activate
```

3. Lastly, install the dependencies

```
(env) pip install -r requirements.txt
```


### Using conda:

1. First we create a virtual environment using conda (if conda not installed, use this link https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

```
conda create env_name python=3.10
```
2. We activate the environment:
```
conda activate env_name
```

3. Lastly, install the dependencies using either

```
conda install --yes --file requirements.txt
```

or 

```
pip install -r requirements.txt
```

4. To launch the app, run the command:

```
(env) python app.py
```

