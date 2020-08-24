# import csv

# with open('./data/conceptnet-assertions-5.7.0.csv', mode='r') as csvFile:
#   reader =  csv.reader(csvFile)
#   count = 0
#   for i, row in enumerate(reader):
#     count += 1
#     if i == 57302:
#       print(row)
#       break
#   print(count)
import pandas as pd
csv_data = pd.read_csv('../data/conceptnet-assertions-5.7.0.csv',  header=None, delimiter="\t")
# print(csv_data.info())
pd.set_option('display.max_rows', None)
print(csv_data.loc[(csv_data[2].str.contains('/c/en/excavator') | csv_data[3].str.contains('/c/en/excavator')), 1:3])