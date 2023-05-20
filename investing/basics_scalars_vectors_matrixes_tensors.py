import numpy as np


def matrices():
    """
    The shape of matrixes are depicted rowsxcolumns, so a 3 by 2 matrix contains 3 rows and 2 columns
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
        print('You can not print the shape of an int.')

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


if __name__ == '__main__':
    matrices()
    tensors()
    matrix_operations()

