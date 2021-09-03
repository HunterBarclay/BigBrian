import matplotlib.pyplot as plt

file = open("logs\\delta_data.csv", "r")
data = file.read()

lines = data.split('\n')
header = lines[0].split(',')

calculations = []
costs = []

for x in range(len(lines) - 2):
    line = lines[x + 1].split(',')
    a = int(line[0])
    b = float(line[header.index("cost")])
#    print(a)
#    print(b)
    calculations.append(a)
    costs.append(b)

plt.plot(calculations, costs)

plt.xlabel("Calculations")
plt.ylabel("Costs")

plt.show()
