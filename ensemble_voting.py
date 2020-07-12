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

file1 = 'submission_1_0_3.csv'
file2 = 'submission_model_2_0_1_epoch45.csv'
file3 = 'submission_4_1_1_a.csv'
files = [file1, file2, file3]

df_joint = pd.DataFrame()
label_columns = []
for f in files:
    df = pd.read_csv(f)
    print(df.head(10))
    df_joint['Id'] = df['Id']
    label_columns.append(df['Label'])
label_columns = np.hstack(label_columns).reshape((len(files), -1)).T
label_columns = np.round(label_columns)
# 0 0 1 -> 0.33 -> 0
# 1 0 1 -> 0.66 -> 1
mean = np.mean(label_columns, axis=1)
print(mean[:10])
vote_outcome = np.round(mean)
df_joint['Label'] = vote_outcome

print(df_joint.head(10))
filename = "".join([f.split(".")[0] + "_and_" for f in files])
df_joint.to_csv("ensemble_vote_{}.csv".format(filename), index=False)

# print("The percentage of file2 being chosen", float(sum(idxs)) / len(idxs))
# agreements = 1 - np.logical_xor(np.round(df1['Label']), np.round(df2['Label']))
# print("The agreement ratio betweeen the models is:", float(sum(agreements)) / len(agreements))