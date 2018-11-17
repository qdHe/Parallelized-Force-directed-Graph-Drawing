import matplotlib.pyplot as plt
import sys
fn = sys.argv[1]
edge_fn = sys.argv[2]
f = open(fn, 'r')
f1 = plt.figure(1)
cords = []
for line in f.readlines():
    cord = line.split()
    cords.append([(float)(cord[0]), (float)(cord[1])])
    plt.scatter((float)(cord[0]), (float)(cord[1]), 1,color='r')
f.close()
f = open(edge_fn, 'r')
for line in f.readlines():
    cord = line.split()
    print(cord)
    in1 = (int)(cord[0])
    in2 = (int)(cord[1])
    plt.plot([cords[in1][0],cords[in2][0]], [cords[in1][1],cords[in2][1]], linewidth=1, color='b')
f.close()
plt.show()
