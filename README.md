
# ☑ DATA challenge ENS

# Challenge goals

The aim of this challenge is to attempt to identify from which stock a piece of tick-by-tick exchange data belongs. The problem is thus a classification task. The exchange data includes each atomic update of the order-book giving information about the best bid and offer prices, any trades that may have occurred, orders that have been placed in the book or cancelled. The order-book is also an aggregated order-book that is built out of multiple exchanges from where we can buy and sell the same shares. Although the data seems very non-descript and anonymous, we expect there to be clues in the data that will give away from which stock a piece of data belongs. This might be through the average spread, the typical quantities of shares at the bid or ask, the frequency with which trades occur, the distribution of how trades are split amongst the venues on which the stock is traded etc. there is a lot of information to aid the participant.

You'll find information on the challenge here : https://challengedata.ens.fr/challenges/146

I scored 20/96 and I made 5% better than the benchmark. 

This project uses **Python == 3.11**.

## 1. Installation

### 1.1. Virtual environment
```bash
conda env create -f src/environment/conda_dependencies.yml
conda activate challenge_ENS_env_CFM
```

### 1.2. Dev guidelines

1. To update your environment, make sure to run :
```bash
pip install -r src/environment/requirements.txt
```

2. To format your code, you can run :
```bash
invoke format
```

