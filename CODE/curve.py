def calc_turning_point(x_values, log_y_values):
    # Calculate the first derivative
    dy_dx = np.gradient(log_y_values, x_values)
    # Calculate the second derivative
    d2y_dx2 = np.gradient(dy_dx, x_values)
    
    # The first time the second derivative is greater than zero is the turning point
    turning_point_index = np.where(d2y_dx2[1: -1] > 0.0)[0][0]
    
    # Return the x-value of the turning point
    return x_values[turning_point_index + 1]

import matplotlib.pyplot as plt
import numpy as np

# x and y data points
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
y = "0.0127	0.0274	0.0453	0.0667	0.0895	0.1158	0.1460	0.1839	0.2280	0.2919	0.4140	0.7597	1.9236	3.9159	5.6038"
y = "0.0127	0.0274	0.0453	0.0667	0.0895	0.1158	0.1460	0.1839	0.2280	0.2919	0.4140	0.7597	1.9236	3.9159	5.6038"
y = "0.0375	0.0840	0.1334	0.1961	0.2598	0.3360	0.4239	0.5140	0.6501	0.8248"
y = "0.0406	0.0874	0.1338	0.1970	0.2701	0.3571	0.4569	0.5849	0.7142	0.9689	1.4175	3.1345	5.9669	7.3196	8.043"
y = "0.0285	0.0626	0.1057	0.1489	0.2007	0.2632	0.3405	0.4479	0.6085	0.9092"
y = "0.0291	0.0613	0.0998	0.1456	0.1977	0.2612	0.3365	0.4345	0.5499	0.7701	1.1662	2.2944	4.4524	5.8236	6.7282"
y = "0.0260	0.0482	0.0796	0.1065	0.1477	0.1908	0.2524	0.3241	0.4262	0.5949"
y = "0.0269	0.0599	0.1008	0.1612	0.2480	0.3650	0.5508	0.7262	1.0224	1.5036"
y = "0.0275	0.0592	0.09405	0.1346	0.1754	0.2293	0.2907	0.4100	0.5574	0.8995"
y = "0.01330	0.02996	0.04819	0.06695	0.08918	0.1149	0.1394	0.1682	0.2126	0.2678	0.3350	0.4492	0.8692	2.03085	4.3436"
y="0.01463	0.03266	0.05334	0.07400	0.10800	0.1431	0.2138	0.2882	0.38662	0.5223	0.8053	1.2666	2.7485	4.6639	6.2238"
y = y.split("\t")
y = [float(i) for i in y]

# Plotting the curve
plt.plot(x, np.log(np.array(y)), marker='o', linestyle='-', color='b', label='Data Curve')

# Adding labels and title
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.title('Curve Plot')

# Display the plot
plt.grid(True)
plt.legend()
plt.savefig('curve-yi-chat.png')

print(calc_turning_point(np.array(x), np.log(np.array(y))))
