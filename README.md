# Data analysis
- Document here the project: AMEX_default_prediction
- Description: the project is a graduation project of Le wagon data science program. It aims to use the history data of American express to predict who might default when any customers want to apply a credit card in American express. You can predict whether a customer might default [here](https://amexoracle.herokuapp.com/).

- Data Source: [American Express - Default Prediction](https://www.kaggle.com/competitions/amex-default-prediction/data)

Please document the project the better you can.

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for AMEX_default_prediction in github.com/{group}. If your project is not set please add it:

Create a new project on github.com/{group}/AMEX_default_prediction
Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "AMEX_default_prediction"
git remote add origin git@github.com:{group}/AMEX_default_prediction.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
AMEX_default_prediction-run
```

# Install

Go to `https://github.com/{group}/AMEX_default_prediction` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/AMEX_default_prediction.git
cd AMEX_default_prediction
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
AMEX_default_prediction-run
```
