import matplotlib.pyplot as plt
import numpy as np

# Sample data for 6 questions (replace with actual data)
questions = ["Q 3.1", "Q 3.2", "Q 3.3", "Q 3.4", "Q 3.5"]
option_a_percentages = [99.32, 85.14, 17.57, 93.24, 6.08]  # Percent choosing Option A in each question
option_b_percentages = [0.68, 14.86, 82.43, 6.76, 93.92]  # Percent choosing Option B in each question

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
ax.set_title('Scenario 3: Table Tennis Skills')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(questions)
ax.legend()

plt.show()
