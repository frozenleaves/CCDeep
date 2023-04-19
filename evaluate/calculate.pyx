
# 计算DICE系数，即DSI
cpdef calDSI(binary_GT, binary_R):
    cdef int row = binary_GT.shape[0], col = binary_GT.shape[1]  # 矩阵的行与列
    cdef int DSI_s = 0, DSI_t = 0
    cdef float DSI
    cdef long i, j
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                DSI_s += 1
            if binary_GT[i][j] == 255:
                DSI_t += 1
            if binary_R[i][j] == 255:
                DSI_t += 1
    DSI = 2 * DSI_s / DSI_t
    # print(DSI)
    return DSI


# 计算VOE系数，即VOE
cpdef calVOE(binary_GT, binary_R):
    cdef int row = binary_GT.shape[0], col = binary_GT.shape[1]  # 矩阵的行与列
    cdef int VOE_s = 0, VOE_t = 0
    cdef float VOE
    cdef long i, j
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                VOE_s += 1
            if binary_R[i][j] == 255:
                VOE_t += 1
    VOE = 2 * (VOE_t - VOE_s) / (VOE_t + VOE_s)
    return VOE


# 计算RVD系数，即RVD
cpdef calRVD(binary_GT, binary_R):
    cdef int row = binary_GT.shape[0], col = binary_GT.shape[1]  # 矩阵的行与列
    cdef int RVD_s = 0, RVD_t = 0
    cdef float RVD
    cdef long i, j
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255:
                RVD_s += 1
            if binary_R[i][j] == 255:
                RVD_t += 1
    RVD = RVD_t / RVD_s - 1
    return RVD


# 计算Prevision系数，即Precison
cpdef calPrecision(binary_GT, binary_R):
    cdef int row = binary_GT.shape[0], col = binary_GT.shape[1]  # 矩阵的行与列
    cdef int P_s = 0, P_t = 0
    cdef float Precision
    cdef long i, j
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                P_s += 1
            if binary_R[i][j] == 255:
                P_t += 1

    Precision = P_s / P_t
    return Precision


cpdef calRecall(binary_GT, binary_R):
    cdef int row = binary_GT.shape[0], col = binary_GT.shape[1]  # 矩阵的行与列
    cdef int R_s = 0, R_t = 0
    cdef float Recall
    cdef long i, j
    for i in range(row):
        for j in range(col):
            if binary_GT[i][j] == 255 and binary_R[i][j] == 255:
                R_s += 1
            if binary_GT[i][j] == 255:
                R_t += 1

    Recall = R_s / R_t
    return Recall

