#ifndef BFM_CONSTANT_H
#define BFM_CONSTANT_H

#define BFM_N_VERTICES     28588
#define BFM_N_FACES        56572
#define BFM_N_ID_PCS       199
#define BFM_N_EXPR_PCS     100

#define DENSE_WEIGHT        1
#define DENSE_WEIGHT_LIGHT  1
#define SPARSE_WEIGHT       5
#define COLOR_REG_WEIGHT    2.5e-5
#define SHAPE_REG_WEIGHT    2.5e-5
#define EXPR_REG_WEIGHT     2.5e-5

#define CHECK_BFM_DATA 0

const unsigned int N_LANDMARKS = 68;

#endif // BFM_CONSTANT_H