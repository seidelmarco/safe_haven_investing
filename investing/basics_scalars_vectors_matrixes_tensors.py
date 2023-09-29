import numpy as np


def matrices():
    """
    The shape of matrixes are depicted rows x columns, so a 3 by 2 matrix contains 3 rows and 2 columns
    :return:
    """
    s = 5
    print(type(s))

    v = np.array([5, 3, 1])
    print(v)
    print(type(v))

    m = np.array([[5, 12, 6], [-3, 0, 14]])
    print(type(m))
    print(m)

    s_array = np.array([5])

    # Data shapes:
    print(f'Shape of m: {m.shape}')
    print(f'Shape of v: {v.shape}')
    try:
        print(f'Shape of s: {s.shape}')
    except AttributeError as e:
        print('You can not print the shape of an int./scalar')

    print(f'Shape of s_array: {s_array.shape}')

    # Creating a column vector:
    print(v)
    print(v.reshape(3, 1))

    # Addition:
    sum_m_s = m+s_array
    print(sum_m_s)


def tensors():
    """
    A scalar has the lowest dimensionality and is always one by one. It can be thought of a vector of length one or
    a one by one matrix.
    Each element of a vector is a scalar. The dimensions of a vector are  m x 1 or 1 x m matrix.
    Matrices are a collection of vectors. The dimension of a matrix are m by n.

    A tensor is the most general concept: scalars, vectors and matrices are TENSORS of ranks zero, one and
    two respectively. Tensors are just a generalization of the scalar, vector, matrix-concept.

    A tensor of rank 3 has the dimensions k x mx n. It's a collection of matrices.
    :return:
    """
    # Creating a tensor:
    m1 = np.array([[5, 12, 6], [-3, 0, 14]])
    print('m1:', m1)

    m2 = np.array([[9, 8, 7], [1, 3, -5]])
    print('m2:', m2)

    t = np.array([m1, m2])  # that's all :-)
    print('Tensor t with shape', t.shape, ':\n', t)

    print("""
        Manually creating a tensor:
        2 x 2 x 3: two matrices, two rows by three columns each
    """)
    t_manual = np.array([[[5, 12, 6], [-3, 0, 14]], [[9, 8, 7], [1, 3, -5]]])
    print(t_manual)


def matrix_operations():
    """
    Only one condition: the two matrices must have the same dimensions
    We just have to add the corresponding entries one with the other.
    Subtraction works the same.
    :return:
    """
    m1 = np.array([[5, 12, 6], [-3, 0, 14]])
    print('m1:', m1)

    m2 = np.array([[9, 8, 7], [1, 3, -5]])
    print('m2:', m2)

    sum_m1_m2 = m1 + m2
    print('Sum:', sum_m1_m2)

    # Adding vectors is the same:
    v1 = np.array([1, 2, 3, 4, 5])
    v2 = np.array([5, 4, 3, 2, 1])
    v_sum = v1 + v2
    print(v_sum)
    v_difference = v1 - v2
    print(v_difference)

    """
    Error handling:
    Value Error: operands could not broadcast together with shapes (2, 3) (2, 2): means
    we are not allowed to calculate matrices with different dimensions
    
    Same for vectors: e. g. row-vectors: operands could not be broadcast together with shapes (5, 1) (3, )
    
    Exception: addition with a scalar - this works
    """

    # Transposing matrices with .T
    A = np.array([[5, 12, 6], [-3, 0, 14]])
    print(A)
    print(A.T)

    # Transposing vectors - only possible after reshaping the vector:
    x = np.array([1, 2, 3])
    print(x)
    print(x.T)

    x_reshaped = x.reshape(1, 3)

    # Lifehack: reshaping also works with double-brackets; you see it after printing
    print(x_reshaped)
    print(x_reshaped.T)

    # Dot-Product with .dot() - Multiplication:
    # Vectors:
    x = np.array([2, 8, -4])
    y = np.array([1, -7, 3])

    print('2*1+8*-7+-4*3 = x * y =', 2*1+8*-7+-4*3)
    print('np.dot of x and y:', np.dot(x, y))

    # Dot-Product of matrices:

    # Condition: we can only multiply an m x n with an n x k matrix
    # for example: 2x3 with a 3x1 matrix
    # if we don't match this condition, we can transpose one matrix

    a = np.array([[5, 12, 6], [-3, 0, 14]])
    b = np.array([[2, 8, 3], [-1, 0, 0]])
    b_shaped_manually = b.reshape(3, 2)
    print(b_shaped_manually)

    # Important: when we have a dot product, we always multiply a row vector times every column vector

    ab_manually_row1col1 = 5*2 + 12*8 + 6*3
    ab_manually_row1col2 = 5*-1 + 12*0 + 6*0
    ab_manually_row2col1 = -3*2 + 0*8 +14*3
    ab_manually_row2col2 = -3*-1 + 0*0 + 14*0

    ab_manually = np.array([[[ab_manually_row1col1, ab_manually_row1col2]], [[ab_manually_row2col1, ab_manually_row2col2]]])
    print('AB-manually:', ab_manually)

    print(a.shape, b.shape)
    b_reshaped = b.T
    print(a.shape, b_reshaped.shape)

    ab = np.dot(a, b_reshaped)
    print(ab)


if __name__ == '__main__':
    matrices()
    tensors()
    matrix_operations()

