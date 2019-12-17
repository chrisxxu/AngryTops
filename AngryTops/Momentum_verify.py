import pandas as pd
import numpy as np
import sys
import features as at
import matplotlib
from matplotlib import pyplot as plt

# pxpypzEM representation
jets_pxpypzEM = [
"jet0 P_x",  "jet0 P_y",  "jet0 P_z",  "jet0 E",  "jet0 M",
"jet1 P_x",  "jet1 P_y",  "jet1 P_z",  "jet1 E",  "jet1 M",
"jet2 P_x",  "jet2 P_y",  "jet2 P_z",  "jet2 E",  "jet2 M",
"jet3 P_x",  "jet3 P_y",  "jet3 P_z",  "jet3 E",  "jet3 M",
"jet4 P_x",  "jet4 P_y",  "jet4 P_z",  "jet4 E",  "jet4 M"]

# Lepton info (cartesian) + Energy + missing transverse energy info
lep_cartE = ["lep Px", "lep Py", "lep Pz", "lep E", "met_met", "met_phi"]

output_columns_pxpypzE = [
"target_W_had_Px", "target_W_had_Py", "target_W_had_Pz", "target_W_had_E",
"target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz", "target_W_lep_E",
"target_b_had_Px", "target_b_had_Py", "target_b_had_Pz", "target_b_had_E",
"target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz", "target_b_lep_E",
"target_t_had_Px", "target_t_had_Py", "target_t_had_Pz", "target_t_had_E",
"target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz", "target_t_lep_E"]

column_names = ["runNumber", "eventNumber", "weight", "jets_n", "bjets_n",
"lep Px", "lep Py", "lep Pz", "lep E", "met_met", "met_phi",
"jet0 P_x",  "jet0 P_y",  "jet0 P_z",  "jet0 E",  "jet0 M",  "jet0 BTag",
"jet1 P_x",  "jet1 P_y",  "jet1 P_z",  "jet1 E",  "jet1 M",  "jet1 BTag",
"jet2 P_x",  "jet2 P_y",  "jet2 P_z",  "jet2 E",  "jet2 M",  "jet2 BTag",
"jet3 P_x",  "jet3 P_y",  "jet3 P_z",  "jet3 E",  "jet3 M",  "jet3 BTag",
"jet4 P_x",  "jet4 P_y",  "jet4 P_z",  "jet4 E",  "jet4 M",  "jet4 BTag",
"target_W_had_Px", "target_W_had_Py", "target_W_had_Pz", "target_W_had_E", "target_W_had_M",
"target_W_lep_Px", "target_W_lep_Py", "target_W_lep_Pz", "target_W_lep_E", "target_W_lep_M",
"target_b_had_Px", "target_b_had_Py", "target_b_had_Pz", "target_b_had_E", "target_b_had_M",
"target_b_lep_Px", "target_b_lep_Py", "target_b_lep_Pz", "target_b_lep_E", "target_b_lep_M",
"target_t_had_Px", "target_t_had_Py", "target_t_had_Pz", "target_t_had_E", "target_t_had_M",
"target_t_lep_Px", "target_t_lep_Py", "target_t_lep_Pz", "target_t_lep_E", "target_t_lep_M",
"lep Pt", "lep Eta", "lep Phi",
"jet0 Pt", "jet0 Eta", "jet0 Phi",
"jet1 Pt", "jet1 Eta", "jet1 Phi",
"jet2 Pt", "jet2 Eta", "jet2 Phi",
"jet3 Pt", "jet3 Eta", "jet3 Phi",
"jet4 Pt", "jet4 Eta", "jet4 Phi",
"target_W_had_Pt", "target_W_had_Eta", "target_W_had_Phi",
"target_W_lep_Pt", "target_W_lep_Eta", "target_W_lep_Phi",
"target_b_had_Pt", "target_b_had_Eta", "target_b_had_Phi",
"target_b_lep_Pt", "target_b_lep_Eta", "target_b_lep_Phi",
"target_t_had_Pt", "target_t_had_Eta", "target_t_had_Phi",
"target_t_lep_Pt", "target_t_lep_Eta", "target_t_lep_Phi",
"Event HT", "Closest b Index", "DeltaPhi", "Invariant Mass"
]

# def p_verify(input_filename, **kwargs):

input_filename = 'test.csv'

# Cartesian; eg. we dont use p_psi here
rep = 'pxpypzEM'

# Read the data from the csv file
representations = [lep_cartE, jets_pxpypzEM, output_columns_pxpypzE]
df = pd.read_csv(input_filename, names=column_names)
df_outliers = pd.DataFrame(columns = column_names)

# Initalize the array for data storing
mass_actual = np.zeros(df.shape[0])
mass_compute = np.zeros(df.shape[0])
mass_diff = np.zeros(df.shape[0])

# Loops through df(Data Frame, read from csv)
# Leptonics for now
for index, event in df.iterrows():

    # compute m_t  = sqrt[(p_w+p_b)^2 + (E_w+E_b)^2], here p is 3-momentum
    mass_compute[index] = np.sqrt((event['target_W_lep_E']  + event['target_b_lep_E'])**2 \
        - (event['target_W_lep_Px'] + event['target_b_lep_Px'])**2 \
        - (event['target_W_lep_Py'] + event['target_b_lep_Py'])**2 \
        - (event['target_W_lep_Pz'] + event['target_b_lep_Pz'])**2 )

    mass_actual[index] = event['target_t_lep_M']
    mass_diff[index] = mass_actual[index] - mass_compute[index]

    if(abs(mass_diff[index]) > 20.):
        df_outliers = df_outliers.append(event)

print(df_outliers)

plt.figure(1)
plt.hist(mass_compute, alpha=0.5, label = 'Computed mass')
plt.hist(mass_actual, alpha=0.5, label = 'Target mass')
plt.yticks(np.arange(0, 150, 10))
plt.xticks(np.arange(80, 200, 10))
plt.legend()
plt.savefig('Mass.pdf', dpi = 300)
plt.show()
plt.close()

plt.figure(2)
plt.hist(mass_diff, bins=range(-10, 90, 10), label = 'Mass difference')
plt.legend()
plt.savefig('Mass_diff.pdf', dpi = 300)
plt.show()
plt.close()


# if __name__=='__main__':
#   p_verify(input_filename=sys.argv[1])
