import numpy as np
import json

def calc_turning_point(x_values, log_y_values):
    # Calculate the first derivative
    dy_dx = np.gradient(log_y_values, x_values)

    # Calculate the second derivative
    d2y_dx2 = np.gradient(dy_dx, x_values)

    print("dy_dx: ", dy_dx)
    print("d2y_dx2: ", d2y_dx2)
    
    # The first time the second derivative is greater than zero is the turning point
    mask = d2y_dx2[1:-1] > 0
    if not np.any(mask):
        # handle "no positive second derivative" case
        mask[-1] = True
    turning_point_index = np.where(mask)[0][0]  # first True
    turning_point_index += 1
    return x_values[turning_point_index]

# Read data from JSON file
with open('data.json', 'r') as f:
    json_data = json.load(f)

# Convert JSON keys and values to lists of floats
index = [float(k) for k in json_data.keys()]
data = [float(v) for v in json_data.values()]

# Sort index and data to ensure correct numerical order
sorted_pairs = sorted(zip(index, data))
index, data = zip(*sorted_pairs)

# Print to verify
print("Index:", index)
print("Data:", data)

# Calculate and print the turning point
turning_point = calc_turning_point(index, np.log(data))
print("Turning Point:", turning_point)