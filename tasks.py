import os

from invoke import task

PATHS_TO_FORMAT = "."
REQUIRMENTS_PATH = "src/environment"
NOTEBOOKS_PATH = "src/notebooks"


@task
def format(c):
    c.run(f"isort --profile black {PATHS_TO_FORMAT}")
    c.run(f"black {PATHS_TO_FORMAT}")


@task
def formatcheck(c):
    c.run(f"isort --profile black {PATHS_TO_FORMAT} -c")
    c.run(f"black --check {PATHS_TO_FORMAT}")


@task
def update(c):
    # c.run(f"pip-compile {REQUIRMENTS_PATH}/requirements.in")
    # c.run(f"pip-compile {REQUIRMENTS_PATH}/dev-requirements.in")
    c.run(f"pip3 install -r {REQUIRMENTS_PATH}/requirements.txt")
    # c.run(f"pip3 install -r {REQUIRMENTS_PATH}/dev-requirements.txt")
