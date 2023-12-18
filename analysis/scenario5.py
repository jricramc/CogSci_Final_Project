import matplotlib.pyplot as plt
import numpy as np

# Sample data for 6 questions (replace with actual data)
questions = ["Q 5.1", "Q 5.2", "Q 5.3", "Q 5.4", "Q 5.5", "Q 5.6"]
option_a_percentages = [61.8, 78.65, 71.91, 22.47, 53.93, 35.96]  # Percent choosing Option A in each question
option_b_percentages = [38.2, 21.35, 28.09, 77.53, 46.07, 64.04]  # Percent choosing Option B in each question

n_groups = len(questions)
index = np.arange(n_groups)
bar_width = 0.35

fig, ax = plt.subplots()

# Creating bars for each option with different shades of blue
bar1 = ax.bar(index, option_a_percentages, bar_width, color='royalblue', label='Option A')
bar2 = ax.bar(index + bar_width, option_b_percentages, bar_width, color='lightblue', label='Option B')

# Adding labels and title
ax.set_xlabel('Questions')
ax.set_ylabel('Percentages')
ax.set_title('Scenario 5: Soccer Team Skills')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(questions)
ax.legend()

plt.show()
