import numpy as np

def calc_turning_point(x_values, log_y_values):
    # Calculate the first derivative
    dy_dx = np.gradient(log_y_values, x_values)

    # Calculate the second derivative
    d2y_dx2 = np.gradient(dy_dx, x_values)
    
    # The first time the second derivative is greater than zero is the turning point
    turning_point_index = np.argmax(d2y_dx2[1: -1] > 0)
    
    # Return the x-value of the turning point
    return x_values[turning_point_index + 1]

index = "0.1	0.2	0.3	0.4	0.5	0.6	0.7	0.8	0.9	1.0	1.1	1.2	1.3	1.4	1.5	1.7	2.0"
data = "0.0275	0.0592	0.09405	0.1346	0.1754	0.2293	0.2907	0.4100	0.5574	0.8995"
index = index.split("\t")
index = [float(i) for i in index]
data = data.split("\t")
data = [float(i) for i in data]
index = index[:len(data)]
print(index)
print(data)

print(calc_turning_point(index, np.log(data)))