import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

def calculate_jsd(p_distribution, q_distribution):
    """
    Calculate the Jensen-Shannon Divergence (JSD) between two probability distributions.

    :param p_distribution: First probability distribution as a list or numpy array.
    :param q_distribution: Second probability distribution as a list or numpy array.
    :return: JSD value between the two distributions.
    """
    # Convert the distributions to numpy arrays if they aren't already
    p_distribution = np.array(p_distribution)
    q_distribution = np.array(q_distribution)

    # Calculate the mean distribution
    m_distribution = (p_distribution + q_distribution) / 2

    # Calculate the Kullback-Leibler Divergence for each distribution against the mean distribution
    kld_p = entropy(p_distribution, m_distribution)
    kld_q = entropy(q_distribution, m_distribution)

    # Calculate the Jensen-Shannon Divergence
    jsd = (kld_p + kld_q) / 2

    return jsd

# #Scenario 1
print()
p_general_11 = [.4717, .5283]
q_general_11 = [0.961, 0.039]
jsd_value = calculate_jsd(p_general_11, q_general_11)
print(f"Scenario 1.1 {jsd_value}")

p_general_12 = [.9245, .0755]
q_general_12 = [0.131, 0.869]
jsd_value = calculate_jsd(p_general_12, q_general_12)
print(f"Scenario 1.2 {jsd_value}")

p_general_13 = [.673, .327]
q_general_13 = [0.997, 0.003]
jsd_value = calculate_jsd(p_general_13, q_general_13)
print(f"Scenario 1.3 {jsd_value}")

p_general_14 = [.9811, .0189]
q_general_14 = [0.975, 0.025]
jsd_value = calculate_jsd(p_general_14, q_general_14)
print(f"Scenario 1.4 {jsd_value}")

p_general_15 = [.9623, .0377]
q_general_15 = [0.9637, 0.0363]
jsd_value = calculate_jsd(p_general_15, q_general_15)
print(f"Scenario 1.5 {jsd_value}")



#Scenario 2
print()
p_general_21 = [.9811, .0189]
q_general_21 = [0.986, 0.014]
jsd_value = calculate_jsd(p_general_21, q_general_21)
print(f"Scenario 2.1 {jsd_value}")

p_general_22 = [.5849, .4151]
q_general_22 = [0.002, 0.998]
jsd_value = calculate_jsd(p_general_22, q_general_22)
print(f"Scenario 2.2 {jsd_value}")

p_general_23 = [.4403, .5597]
q_general_23 = [0.732, 0.268]
jsd_value = calculate_jsd(p_general_23, q_general_23)
print(f"Scenario 2.3 {jsd_value}")

p_general_24 = [.5472, .4528]
q_general_24 = [0.443, 0.557]
jsd_value = calculate_jsd(p_general_24, q_general_24)
print(f"Scenario 2.4 {jsd_value}")

p_general_25 = [.1321, .8679]
q_general_25 = [0.971, 0.029]
jsd_value = calculate_jsd(p_general_25, q_general_25)
print(f"Scenario 2.5 {jsd_value}")


#Scenario 3
print()
p_general_31 = [.9932, .0068]
q_general_31 = [0.98, 0.02]
jsd_value = calculate_jsd(p_general_31, q_general_31)
print(f"Scenario 3.1 {jsd_value}")

p_general_32 = [.8514, .1486]
q_general_32 = [0.62, 0.38]
jsd_value = calculate_jsd(p_general_32, q_general_32)
print(f"Scenario 3.2 {jsd_value}")

p_general_33 = [.1757, .8243]
q_general_33 = [0.99, 0.01]
jsd_value = calculate_jsd(p_general_33, q_general_33)
print(f"Scenario 3.3 {jsd_value}")

p_general_34 = [.9324, .0676]
q_general_34 = [0.96, 0.04]
jsd_value = calculate_jsd(p_general_34, q_general_34)
print(f"Scenario 3.4 {jsd_value}")

p_general_35 = [.0608, .9392]
q_general_35 = [0.6, 0.4]
jsd_value = calculate_jsd(p_general_35, q_general_35)
print(f"Scenario 3.5 {jsd_value}")



#Scenario 4
print()
p_general_41 = [.1222, .8778]
q_general_41 = [0.07, 0.93]
jsd_value = calculate_jsd(p_general_41, q_general_41)
print(f"Scenario 4.1 {jsd_value}")

p_general_42 = [.8556, .1444]
q_general_42 = [0.95, 0.05]
jsd_value = calculate_jsd(p_general_42, q_general_42)
print(f"Scenario 4.2 {jsd_value}")

p_general_43 = [.6889, .3111]
q_general_43 = [0.91, 0.09]
jsd_value = calculate_jsd(p_general_43, q_general_43)
print(f"Scenario 4.3 {jsd_value}")

p_general_44 = [.6778, .3222]
q_general_44 = [0.94, 0.06]
jsd_value = calculate_jsd(p_general_44, q_general_44)
print(f"Scenario 4.4 {jsd_value}")

p_general_45 = [.3111, .6889]
q_general_45 = [0.33, 0.67]
jsd_value = calculate_jsd(p_general_45, q_general_45)
print(f"Scenario 4.5 {jsd_value}")



#Scenario 5
print()
p_general_51 = [.618, .382]
q_general_51 = [0.9386, 0.0614]
jsd_value = calculate_jsd(p_general_51, q_general_51)
print(f"Scenario 5.1 {jsd_value}")

p_general_52 = [.7865, .2135]
q_general_52 = [0.0066, 0.9934]
jsd_value = calculate_jsd(p_general_52, q_general_52)
print(f"Scenario 5.2 {jsd_value}")

p_general_53 = [.7191, .2809]
q_general_53 = [0.5, 0.5]
jsd_value = calculate_jsd(p_general_53, q_general_53)
print(f"Scenario 5.3 {jsd_value}")

p_general_54 = [.2247, .7753]
q_general_54 = [0.951, 0.049]
jsd_value = calculate_jsd(p_general_54, q_general_54)
print(f"Scenario 5.4 {jsd_value}")

p_general_55 = [.5393, .4607]
q_general_55 = [0.7, 0.3]
jsd_value = calculate_jsd(p_general_55, q_general_55)
print(f"Scenario 5.5 {jsd_value}")

p_general_56 = [.3596, .6404]
q_general_56 = [0.9, 0.1]
jsd_value = calculate_jsd(p_general_56, q_general_56)
print(f"Scenario 5.6 {jsd_value}")