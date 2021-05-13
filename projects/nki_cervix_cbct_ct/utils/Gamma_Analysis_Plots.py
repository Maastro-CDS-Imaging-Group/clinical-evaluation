#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylustrator


# In[2]:


# pylustrator.start()


# In[3]:


# Read csv file with Gamma Analysis results
gamma_results = pd.read_csv("/home/suraj/Workspace/results/NKI/media_results/dose_check/dosimetrics.csv", index_col=0)


# In[4]:


gamma_results.head()


# In[5]:


processed_gamma_results = pd.DataFrame()

gamma_rate_labels = [tag for tag in gamma_results.columns if 'rate' in tag]

processed_gamma_results["gamma"] = pd.concat([gamma_results[tag] for tag in gamma_rate_labels])
processed_gamma_results["config"] = pd.concat([pd.Series([tag.split("_", 3)[-1].replace("_", "/")] * len(gamma_results)) for tag in gamma_rate_labels])
processed_gamma_results["modality"]  = ["CBCT"] * (len(processed_gamma_results["config"])//2) + ["sCT"] * (len(processed_gamma_results["config"])//2)


# In[6]:


processed_gamma_results.head()


# In[7]:


# Plot the gamma values in a box plot
f = plt.figure(figsize=(10, 10))
sns.set_theme(style="ticks", palette="vlag")
sns.boxplot(data=processed_gamma_results, x="config", y="gamma", hue="modality", showmeans=True)
# plt.xticks(rotation=45)
sns.despine(offset=10, trim=True)

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(25.000000/2.54, 21.100000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.126832, 0.144673, 0.781000, 0.843848])
plt.figure(1).axes[0].xaxis.labelpad = 19.200000
plt.figure(1).axes[0].get_legend()._set_loc((0.856390, 0.028566))
plt.figure(1).axes[0].patches[0].set_facecolor("#7891b5")
plt.figure(1).axes[0].patches[1].set_facecolor("#aeb9cb")
plt.figure(1).axes[0].get_xaxis().get_label().set_text("Gamma DD/DTA")
plt.figure(1).axes[0].get_yaxis().get_label().set_text("Pass rates")
#% end: automatic generated code from pylustrator
plt.show()
# In[ ]:




