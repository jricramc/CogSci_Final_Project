import matplotlib.pyplot as plt
import numpy as np

# Sample data for 6 questions (replace with actual data)
questions = ["Q 4.1", "Q 4.2", "Q 4.3", "Q 4.4", "Q 4.5"]
option_a_percentages = [12.22, 85.56, 68.89, 67.78, 31.11]  # Percent choosing Option A in each question
option_b_percentages = [87.78, 14.44, 31.11, 32.22, 68.89]  # Percent choosing Option B in each question

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
ax.set_title('Scenario 4: Stock Trading Skills')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(questions)
ax.legend()

plt.show()
