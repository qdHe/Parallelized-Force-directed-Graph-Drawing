import matplotlib.pyplot as plt
fn = "input.txt"
f = open(fn, 'r')
f1 = plt.figure(1)
for line in f.readlines():
    cord = line.split()
    plt.scatter((float)(cord[0]), (float)(cord[1]), color='r')
plt.show()
