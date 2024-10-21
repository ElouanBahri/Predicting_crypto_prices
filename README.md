
# ☑ DATA challenge ENS
This project uses **Python == 3.11**.

## 1. Installation

### 1.1. Virtual environment
```bash
conda env create -f src/environment/conda_dependencies.yml
conda activate challenge_ENS_env_CFM

### 1.2. Dev guidelines

1. To update your environment, make sure to run :
```bash
pip install -r src/environment/requirements.txt
```

2. To format your code, you can run :
```bash
invoke format
```
## 2. Run streamlit

1. To launch streamlit , you can run : 

```bash
streamlit run src/streamlit_app/Accueil.py
```
