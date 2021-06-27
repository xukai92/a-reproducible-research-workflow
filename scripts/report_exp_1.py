#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os, sys
PROJ_DIR = os.path.expanduser("~/projects/a-reproducible-research-workflow")
RESULTS_DIR = os.path.join(PROJ_DIR, "results", "exp_1")


# In[3]:


import torch

import pandas as pd

from matplotlib import pyplot as plt
plt.style.use("bmh")
plt.xkcd()


# In[4]:


def collect_results():
    df = pd.DataFrame(columns=["Learning rate", "Batch size", "Accuracy"])

    for (i, name) in enumerate(os.listdir(RESULTS_DIR)):
        hps = torch.load(os.path.join(RESULTS_DIR, name, "hps.pt"))
        metrics = torch.load(os.path.join(RESULTS_DIR, name, "metrics.pt"))
        df.loc[name] = [
            hps.lr, hps.batch_size, metrics["accuracy"]
        ]

    return df

if __name__ == "__main__":
    df = collect_results()
    print(df)


# In[5]:


def process_df(df):
    return df.groupby(['Learning rate'])["Accuracy"].mean().reset_index()

# For LaTeX
#print(process_df(df).to_latex(float_format="%#.3g"))


# In[8]:


def make_plot(df):
    df_processed = process_df(df)
    
    fig, ax = plt.subplots()
    ax.plot(df_processed['Learning rate'], df_processed["Accuracy"])
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Accuracy")

    return fig

if __name__ == "__main__":
    make_plot(df)


# You can convert the notebook to an executable script in the end after tuning.
# - You can't use `-` if you want to import it later.

# In[ ]:




