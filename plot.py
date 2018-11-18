import matplotlib.pyplot as plt
import sys
fn = sys.argv[1]
edge_fn = sys.argv[2]
f = open(fn, 'r')
f1 = plt.figure(1)
cords = []
cnt = 0
for line in f.readlines():
    cord = line.split()
    cords.append([(float)(cord[0]), (float)(cord[1])])
    plt.scatter((float)(cord[0]), (float)(cord[1]), 1,color='r')
    #cnt += 1
    #if cnt > 1000:
    #    break
f.close()
f = open(edge_fn, 'r')
for line in f.readlines()[1:]:
    cord = line.split()
    in1 = (int)(cord[0])
    in2 = (int)(cord[1])
    #if in1 >= 1000 or in2 >= 1000:
    #    continue
    plt.plot([cords[in1][0],cords[in2][0]], [cords[in1][1],cords[in2][1]],
				linewidth=1, color='b', alpha=0.1)
f.close()
plt.show()
