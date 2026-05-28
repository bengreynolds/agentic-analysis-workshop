import numpy as np


# This script works, but it is intentionally hard to read.
# Practice goal: ask Codex to improve readability while preserving behavior.

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([4,5,5,6,8,8,9,11,12,14])
z=[]
for i in range(len(x)):
    z.append(y[i]-x[i])

a=sum(z)/len(z)
b=[]
for i in range(len(z)):
    if z[i]>a:
        b.append(z[i])

print("average difference:",a)
print("values above average:",b)

v=0
for i in range(len(y)):
    v=v+y[i]
print("total signal:",v)

if a>2:
    print("large average change")
else:
    print("small average change")
