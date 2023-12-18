import matplotlib.pyplot as plt
import numpy as np

# Sample data for 6 questions (replace with actual data)
questions = ["Q 1.1", "Q 1.2", "Q 1.3", "Q 1.4", "Q 1.5", "Q 1.6"]
option_a_percentages = [47.17, 92.45, 67.3, 98.11, 96.23, 94.97]  # Percent choosing Option A in each question
option_b_percentages = [52.83, 7.55, 32.7, 1.89, 3.77, 5.03]  # Percent choosing Option B in each question

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
ax.set_title('Scenario 1: General Comparisons')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(questions)
ax.legend()

plt.show()
