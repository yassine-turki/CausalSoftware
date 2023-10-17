# CausalSoftware


## Getting started

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

