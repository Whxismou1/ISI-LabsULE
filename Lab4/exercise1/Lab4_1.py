import pandas as pd
import matplotlib.pyplot as plt


# Apartado A
# a. Loading Data: Load the provided CSV file, "student_data.csv", into a pandas
# DataFrame. The CSV file contains columns: Name, Age, and Score.
ruta = './Lab4/exercise1/'
studentsInfo = {'Name': ['Mou', 'Ana', 'Diego', 'Mario', 'Laura'], 'Age': [
    21, 20, 70, 45, 23], 'hours': [100, 22, 39, 30, 3], 'Score': [80, 70, 95, 80, 10]}

df = pd.DataFrame(studentsInfo)

df.to_csv(ruta+'studentsInfo.csv')

# b. Basic Exploration:
# c. Display the first 5 rows of the DataFrame to inspect the data.
print(df.head(5))
print("---------------------------")
# d. Print the data types of each column.
print(df.dtypes)
print("---------------------------")
# e. Check for any missing values in the DataFrame.
print(df.isnull().sum())
print("---------------------------")
# f. Statistical Analysis: Compute and print the mean and median of the test scores.
score_table = df['Score']
print("Mean score: ", score_table.mean())
print("Median score: ", score_table.median())
print("---------------------------")

# g. Filtering Data: Create a new DataFrame containing only the rows where the age is greater than 20.
dfAgeBiggerThan20 = df[df['Age'] > 20]
print(dfAgeBiggerThan20)
print("---------------------------")
# h. Grouping and Aggregation: Group the data by age and calculate the average test
# score for each age group.
age_group = df.groupby('Age')['Score'].mean().reset_index()
print(age_group)
print("---------------------------")
# i. Display the resulting DataFrame showing age and the corresponding average
# test score.
# j. Exporting Data: Save the filtered DataFrame (from step 4) to a new CSV file
# named "filtered_student_data.csv"

dfAgeBiggerThan20.to_csv(ruta+'filtered_student_data.csv')


# df['relScoreAge'] = df['Score'] / df['Age']


plt.figure(figsize=(10, 6))
plt.plot(df['Name'], df['Score'], label='Score', color='r', linewidth=1, marker='o', markersize=10)
plt.bar(df['Name'], df['hours'], width=-0.4, align='edge', label='Hours' )
plt.bar(df['Name'], df['Age'], width=0.4, align='edge' , label='Age' )
plt.xlabel('Nombre')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
