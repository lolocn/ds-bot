with open('./data/glove.840B.300d.txt') as f:
  for i, line in enumerate(f):
    if i == 100:
      print(line)
      break