## Appendix B Mathematics

### B.1 Circle Packing

Circle Packing ($n=26$)
⬇
1 ...python
2 import numpy as mp
3 from scipy.optimize import minimize
4
5 def run_packing():
6 n = 26

initial_centers = []
initial_radii = []
# Adjusted initial radius and spacing parameters
r_initial = 0.102 # Slightly smaller for better flexibility
buffer = 1e-6 # Small buffer to prevent boundary violations
# Generate staggered grid with 5 rows and varying number of circles per row
for row in range(5): # 5 rows total
# Even rows start at r_initial, odd rows also start with buffer
if row % 2 == 0:
x_start = r_initial + buffer # Even rows start slightly inside
else:
x_start = r_initial + buffer # Odd rows also start with buffer
# Varying number of circles per row to fit better
if row == 0 or row == 2 or row == 4:
num_circles = 5 # Even rows (0, 2, 4) have 5 circles
elif row == 1:
num_circles = 6 # First odd row has 6 circles
else: # row == 3
num_circles = 5 # Second odd row has 5 circles
if num_circles == 0:
continue
# Calculate horizontal spacing for this row
if num_circles == 1:
spacing_row = 0.0
else:
# Ensure horizontal spacing is at least 2*r_initial to prevent overlaps
max_horizontal = 1 - 2 * r_initial
spacing_row = max_horizontal / (num_circles - 1) if max_horizontal > 0 else 0.0
# Place circles in this row
for col in range(num_circles):
x = x_start + col * spacing_row
# Vertical positioning with refined vertical spacing
if row == 0:
y = r_initial + buffer # First row starts with buffer
else:
# Vertical spacing with a refined factor for denser packing
y = r_initial + buffer + row * 1.0 * np.sqrt(3) * r_initial
# Ensure y does not exceed 1 - r_initial
if y + r_initial > 1 + 1e-6:
y = 1 - r_initial - 1e-6 # Clamp to prevent overflow
initial_centers.append([x, y])
# Assign initial radii based on row (middle row gets a slight boost)
if row == 2:
initial_radii.append(r_initial + 0.003) # Increased boost for central row
else:
initial_radii.append(r_initial)
# Flatten the initial variables for optimization
variables_initial = []
for i in range(n):
variables_initial extend(initial_centers[i])
variables_initial.append(initial_radii[i])
# Objective function to maximize sum of radii
def objective(vars):
total = 0.0
for i in range(n):
idx = i * 3
total += vars[idx + 2]
return -total # Minimize negative sum to maximize
# Define constraints
constraints = []
# Constraints for center positions and radii
for i in range(n):

# x_i >= r_i
def constraint1(vars, i=i):
idx = i * 3
return vars[idx] - vars[idx + 2]
constraints.append({"type": 'ineq', 'fun': constraint1})

# y_i + r_i <= 1
def constraint2(vars, i=i):
idx = i * 3
return 1 - (vars[idx] + vars[idx + 2])
constraints.append({"type": 'ineq', 'fun': constraint2})

# y_i >= r_i
def constraint3(vars, i=i):
idx = i * 3
return vars[idx + 1] - vars[idx + 2]
constraints.append({"type": 'ineq', 'fun': constraint3})

# y_i + r_i <= 1
def constraint4(vars, i=i):
idx = i * 3
return 1 - (vars[idx + 1] + vars[idx + 2])
constraints.append({"type": 'ineq', 'fun': constraint4})

# Pairwise distance constraints
for i in range(n):
for j in range(i + 1, n):
def constraint_pair(vars, i=i, j=j):
idx_i = i * 3
idx_j = j * 3
x_i, y_i, r_i = vars[idx_i], vars[idx_i + 1], vars[idx_i + 2]
x_j, y_j, r_j = vars[idx_j], vars[idx_j + 1], vars[idx_j + 2]
dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
return dist - (r_i + r_j)
constraints.append({"type": 'ineq', 'fun': constraint_pair})

# Optimize using Sequential Least Squares Programming with refined parameters
result = minimize(
objective,
variables_initial,
method="SLSQP",
constraints=constraints,
options={
'ftol': 1e-14,
'maxiter': 1000000,
'disp': False,
'eps': 1e-12,
'iprint': 0, # Suppress verbose output
'finite_diff_rel_step': np.sqrt(np.finfo(float).eps)
}
)

# Extract optimized centers and radii
optimized_vars = result.x
centers = []
radii = []
for i in range(n):
idx = i * 3
centers.append([optimized_vars[idx], optimized_vars[idx + 1]])
radii.append(optimized_vars[idx + 2])

sum.radii = sum(radii)
return np.array(centers), np.array(radii), sum.radii
```

### Circle Packing ($n=32$)

⬇
python
import numpy as np
from scipy.optimize import minimize
def run_packing():

n = 32
r_initial = 1.0 / (2.0 + 5.0 + np.sqrt(3)) # Maximum radius for vertical constraint

# Generate hexagonal arrangement for 30 circles
centers = []
for row in range(6): # 6 rows with 5 columns each
y = r_initial * (1 + row + np.sqrt(3))
if row % 2 == 0:
x_start = (1.0 - 9.0 + r_initial) / 2 # Adjusted to use more horizontal space
else:
x_start = (1.0 - 9.0 + r_initial) / 2 + r_initial
for col in range(5): # 5 columns
x = x_start + col * 2 * r_initial
# Ensure the circle is within the square and properly spaced
centers.append([x, y])

# Add two extra circles near the top-right and bottom-right corners with adjusted initial positions
extra.x = 1.0 - r_initial - 0.0005
extra.y.top = 0.5
extra.y_bottom = r_initial + 0.0005
centers.append([extra.x, extra.y.top])
centers.append([extra.x, extra.y_bottom])

# Initial radii for all circles
radii = [r_initial] + n

# Flatten the centers and radii into a single array for optimization
x0 = np.concatenate([np.array(centers).ravel(), np.array(radii)])

# Objective function to maximize: sum of radii
def objective(x):
# Unflatten x into centers and radii
centers_flat = x[:n+2].reshape(n, 2)
radii_flat = x[n+2:]
return -np.sum(radii_flat) # Negative because we minimize

# Constraints: for each circle, x_i - r_i >= -1e-12, x_i + r_i <= 1 + 1e-12, same for y
# and for each pair, distance >= r_i + r_j - 1e-12

# Define constraint functions
def constraint_boundary(x):
centers_flat = x[:n+2].reshape(n, 2)
radii_flat = x[n+2:]
constraints = []
epsilon = 1e-12
for i in range(n):
x_i, y_i = centers_flat[i]
r_i = radii_flat[i]
constraints.append(x_i - r_i + epsilon) # x_i - r_i >= -epsilon
constraints.append(1 + epsilon - x_i - r_i) # x_i + r_i <= 1 + epsilon
constraints.append(y_i - r_i + epsilon) # y_i - r_i >= -epsilon
constraints.append(1 + epsilon - y_i - r_i) # y_i + r_i <= 1 + epsilon
return np.array(constraints)

# Define constraint for non-overlapping
def constraint_overlap(x):
centers_flat = x[:n+2].reshape(n, 2)
radii_flat = x[n+2:]
constraints = []
for i in range(n):
for j in range(i + 1, n):
dx = centers_flat[i, 0] - centers_flat[j, 0]
dy = centers_flat[i, 1] - centers_flat[j, 1]
dist = np.sqrt(dx**2 + dy**2)
constraints.append(dist - radii_flat[i] - radii_flat[j] + 1e-12)
return np.array(constraints)

# Combine all constraints
cons = []
# Boundary constraints
cons.append({‘type’: ‘ineq’, ‘fun’: lambda x: constraint_boundary(x)})
# Overlap constraints
cons.append({‘type’: ‘ineq’, ‘fun’: lambda x: constraint_overlap(x)})

⬇
80 # Perform optimization with adjusted parameters
85 result = minimize(
86 objective,
87 x0,
88 method=’SLSQP’,
89 constraints=cons,
90 tol=1e-10,
91 options={’disp’: False,
92 ’maxiter’: 200000, ’ftol’: 1e-12, ’eps’: 1e-8}
93 )
94 #
95 # Extract the result
96 optimized_x = result.x
97 centers_opt = optimized_x[:n*2].reshape(n, 2)
98 radii_opt = optimized_x[n*2:]
99
100 # Check if optimization was successful
101 if not result.success:
102 print(“Optimization failed”)
103 # Fallback to initial guess
104 centers_opt = np.array(centers)
105 radii_opt = np.array(radii)
106 else:
107 # Validate the packing
108 valid = validate_packing(centers_opt, radii_opt)
109 if not valid:
110 print(“Validation failed”)
111 # Fallback to initial guess
112 centers_opt = np.array(centers)
113 radii_opt = np.array(radii)
114
115 sum_radii = np.sum(radii_opt)
116 return centers_opt, radii_opt, sum_radii
⬇

### B.2 Autocorrelation Inequalities

For autocorrelation inequalities, initial sequences are created by sampling a random value in [0,1] and repeating it between 1,000 and 8,000 times (or loading a state-of-the-art construction when available). For the first inequality, the verifier computes the upper bound $2n\cdot\max(f*f)/(\sum f)^{2}$ where $f*f$ denotes discrete autocorrelation; it validates that inputs are non-empty lists of non-negative floats clamped to [0,1000] with sum $\geq 0.01$, and returns $\infty$ for invalid constructions. For the second inequality, verifier computes the lower bound $C_{2}=\|f*f\|_{2}^{2}/(\|f*f\|_{1}\cdot\|f*f\|_{\infty})$ using piecewise-linear integration for the $L^{2}$ norm (Simpson-like rule with endpoint zeros) over the normalized interval [$-1/2,1/2$]. Each algorithm is run with 1 GB with 2 CPUs each and a timeout of up to 1100 seconds.

### B.3 Erdős’

We initialize TTT-Discover with random constructions of 40-100 samples around 0.5 with random perturbations. We filter out sequences with more than 1000 values in the verifier. Each algorithm is run with 1 GB with 2 CPUs each and a timeout of up to 1100 seconds.
