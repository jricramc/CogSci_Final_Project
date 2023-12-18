import matplotlib.pyplot as plt
import numpy as np

# Sample data for 6 questions (replace with actual data)
questions = ["Q 2.1", "Q 2.2", "Q 2.3", "Q 2.4", "Q 2.5"]
option_a_percentages = [98.11, 58.49, 44.03, 54.72, 13.21]  # Percent choosing Option A in each question
option_b_percentages = [1.89, 41.51, 55.97, 45.28, 86.79]  # Percent choosing Option B in each question

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
ax.set_title('Scenario 2: Cooking Skills')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(questions)
ax.legend()

plt.show()
