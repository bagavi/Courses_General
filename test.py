

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

y1 =[ 0.86157867,1.32289644,1.25253654,0.60331238,0.77763153,0.61307573,0.81400574,1.11947213]
y2 = [ 2.87085058,0.56210783,2.79090025,4.08463749,0.08745815,6.29441821
,1.1039089, 2.13491535]
x = [2, 4, 8, 16,32,64,128,256,]

plt.semilogy(x, y1,marker='o', lw=0.5, color="blue",alpha=1)
plt.semilogy(x, y2,marker='o', lw=0.5, color="red",alpha=1)
# plt.plot(Xaxis, Yaxis,marker='o', lw=0.5, color="red",alpha=1)
red_patch =mpatches.Patch(color='blue', label='Indentity')
blue_patch = mpatches.Patch(color='red', label='Correlated')
plt.legend(handles=[red_patch,blue_patch])


plt.title("Stats 311 assignment")
plt.ylabel("Ratio greedy/random", fontsize=16)  
plt.xlabel("K : No of observations", fontsize=16)  
plt.show()


 