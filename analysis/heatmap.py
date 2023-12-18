# Here's the Python code to generate the heatmap of Jensen-Shannon Divergence values. 
# You can run this code in your local Python environment.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Adjusting the JSD values array to fit the heatmap requirements
jsd_values_padded = np.array([
    [0.16821889144117086, 0.36364522805632904, 0.12164047518207033, 0.00021734490580526213, 6.876430000197131e-06, np.nan],
    [0.00018617882014667036, 0.25866777622138715, 0.044598505563466014, 0.005439214103510351, 0.42701891386559887, np.nan],
    [0.0017205225243268922, 0.03533816771750341, 0.4189495651347215, 0.0018904640591034328, 0.1833741642563962, np.nan],
    [0.003966057467739953, 0.013193672688724265, 0.039906044689902956, 0.06007196544866409, 0.00020503480089028962, np.nan],
    [0.08113260857809684, 0.3924520734850411, 0.025474698890130397, 0.3134551404401144, 0.013775551434438843, 0.1699267006277785]
])

# Scenario labels
scenarios = ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4", "Scenario 5"]
sub_scenarios = ["1.1", "1.2", "1.3", "1.4", "1.5", "1.6"]

# Create the heatmap
plt.figure(figsize=(12, 6))
ax = sns.heatmap(jsd_values_padded, annot=True, fmt=".2g", cmap="coolwarm", xticklabels=sub_scenarios, yticklabels=scenarios)
plt.title("Heatmap of Jensen-Shannon Divergence Values by Scenario")
plt.xlabel("Sub-Scenario")
plt.ylabel("Main Scenario")

# Display the heatmap
plt.show()
