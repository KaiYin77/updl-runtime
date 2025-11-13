#ifndef UPDL_NN_UTILS_UDL_H
#define UPDL_NN_UTILS_UDL_H

#include <stdint.h>

#include "updl/updl_operator.h"

#ifdef __cplusplus
extern "C" {
#endif

int64_t updl_udl_bound40(int64_t input);
int16_t updl_udl_bound16(int64_t input);
int64_t updl_udl_right_shift_round64(int64_t input, uint8_t rshift);
int64_t updl_udl_activation64(int64_t input, atype_t activation);

int16_t updl_udl_finalize(int64_t sum,
                          int16_t bias_val,
                          atype_t activation,
                          int32_t eff_multiplier,
                          int16_t eff_shift,
                          int16_t output_zp,
                          int32_t eff_bias_multiplier,
                          int16_t eff_bias_shift);

#ifdef __cplusplus
}
#endif

#endif /* UPDL_NN_UTILS_UDL_H */
