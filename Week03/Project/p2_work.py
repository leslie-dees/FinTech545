import numpy as np
import sys
import time
sys.path.append('C:/Users/lesli/Documents/Duke/Masters/FinTech545')

import fin_package_3 as f3

n = 500
sigma = 0.9 * np.ones((n, n), dtype=np.float64)
np.fill_diagonal(sigma, 1.0)
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357

f3.is_psd(sigma)

# Initialize the root matrix
root = np.zeros((n, n), dtype=np.float64)

#Proof that near_psd convert a matrix into something usable by the Cholesky PSD function
a = f3.near_psd(sigma)
f3.is_psd(a)

f3.chol_psd(root = root, a = a)
print("Since the Cholesky PDf function runs using the near_psd function, the matrix is now PSD")

# Initialize the root matrix
root = np.zeros((n, n), dtype=np.float64)

b = f3.higham_near_psd(sigma)
f3.is_psd(b)

f3.chol_psd(root = root, a = b)
print("Since the Cholesky PDf function runs using the higham function, the matrix is now PSD")

#Example usage and verification
for n in (50, 100, 200, 500, 750, 1000):
    print("N:", n)
    sigma = 0.9 * np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(sigma, 1.0)
    sigma[0, 1] = 0.7357
    sigma[1, 0] = 0.7357

    # Measure execution time for 'near_psd' function
    start_time = time.time()
    v = f3.near_psd(sigma)
    end_time = time.time()
    near_psd_time = end_time - start_time

    # Print execution times
    print(f"Execution time for 'near_psd' function: {near_psd_time} seconds")
    f3.is_psd(v)

    # Measure execution time for 'higham_near_psd' function
    start_time = time.time()
    b = f3.higham_near_psd(sigma)
    end_time = time.time()
    higham_near_psd_time = end_time - start_time

    print(f"Execution time for 'higham_near_psd' function: {higham_near_psd_time} seconds")
    f3.is_psd(b)

    print(f"Frobenius Norm is: {np.linalg.norm(v-b)}")
