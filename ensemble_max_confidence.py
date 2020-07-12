import numpy as np
from numpy import genfromtxt
import pandas as pd

# if __name__ == "__main__":

#     # ARGS
#     parser = argparse.ArgumentParser(description="Training")
#     parser.add_argument("-f1", "--datapath", type=str, default="./alaska2-image-steganalysis", help="Path to root data dir. Default: %(default)s")
#     parser.add_argument("-f2", "--checkpoint", type=str, help="Resume from checkpoint.")
#     parser.add_argument("-s", "--skip_training", action='store_true', help="Skip training and evaluate test set.")
#     options = parser.parse_args()

# file1 = 'submission_1_0_3.csv'
# file2 = 'submission_model_2_0_1_epoch45.csv'
file1 = 'ensemble_submission_1_0_3_and_submission_model_2_0_1_epoch45.csv'
file2 = 'submission_4_1_1_a.csv'

df1 = pd.read_csv(file1)
print(df1.head())
df2 = pd.read_csv(file2)
print(df2.head())

df_joint = pd.DataFrame()
df_joint['Id'] = df1['Id']

labels = np.hstack([df1['Label'], df2['Label']]).reshape((2, -1)).T
dist_from_half = np.abs(0.5 - labels)
idxs = np.argmax(dist_from_half, axis=1)
df_joint['Label'] = labels[np.arange(len(labels)), idxs]
print(df_joint.head())

df_joint.to_csv("ensemble_{}_and_{}.csv".format(file1.split(".")[0], file2.split(".")[0]), index=False)
print("The percentage of file2 being chosen", float(sum(idxs)) / len(idxs))

agreements = 1 - np.logical_xor(np.round(df1['Label']), np.round(df2['Label']))
print("The agreement ratio betweeen the models is:", float(sum(agreements)) / len(agreements))