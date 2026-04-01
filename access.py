import numpy as np

# =========================
# 1D ARRAY
# =========================
arr1 = np.array([10, 20, 30, 40, 50])

print("1D Array:", arr1)

# Access elements
print(arr1[0])     # First element
print(arr1[-1])    # Last element
print(arr1[1:4])   # Slicing
print(arr1[::2])   # Step slicing

# =========================
# 2D ARRAY
# =========================
arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("\n2D Array:\n", arr2)

# Access elements
print(arr2[0, 0])   # First row, first col
print(arr2[1, 2])   # Row 2, Col 3
print(arr2[-1, -1]) # Last element

# Access rows
print(arr2[0])      # First row
print(arr2[:, 1])   # Second column

# Slicing
print(arr2[0:2, 1:3])  # Submatrix
print(arr2[:, ::2])    # All rows, step columns

# =========================
# 3D ARRAY
# =========================
arr3 = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])

print("\n3D Array:\n", arr3)

# Access elements
print(arr3[0, 0, 0])   # First block, row, col
print(arr3[1, 1, 2])   # Specific element
print(arr3[-1, -1, -1])# Last element

# Access full sections
print(arr3[0])         # First 2D block
print(arr3[:, 0, :])   # All blocks, first row

# Slicing
print(arr3[:, :, 1])   # All blocks, all rows, 2nd column
print(arr3[0:2, :, 0:2]) # Sub 3D slice

# =========================
# BOOLEAN INDEXING
# =========================
arr = np.array([10, 20, 30, 40, 50])

print("\nElements > 25:", arr[arr > 25])

# =========================
# FANCY INDEXING
# =========================
arr = np.array([10, 20, 30, 40, 50])

print("Select indices [0,2,4]:", arr[[0, 2, 4]])

# =========================
# MULTI-DIMENSION FANCY INDEXING
# =========================
arr2 = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

rows = [0, 1]
cols = [1, 2]

print("Fancy Indexing 2D:", arr2[rows, cols])

# =========================
# CONDITIONAL FILTERING
# =========================
arr = np.array([5, 15, 25, 35, 45])

print("Values > 20:", arr[arr > 20])

# =========================
# MODIFY VALUES USING INDEX
# =========================
arr = np.array([1, 2, 3, 4, 5])
arr[2] = 100
print("Modified Array:", arr)

# =========================
# IMPORTANT SHAPES
# =========================
print("\nShape 1D:", arr1.shape)
print("Shape 2D:", arr2.shape)
print("Shape 3D:", arr3.shape)