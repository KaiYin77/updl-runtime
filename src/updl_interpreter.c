/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

#include "updl/updl_interpreter.h"

void updl_read_bytes(void *target, uint8_t **fp, size_t bytes)
{
    memcpy(target, *fp, bytes);
    *fp += bytes;
}

void updl_load_data(void *data, uint8_t **fp, dtype_t dtype, uint32_t count, 
                    const char *tag_name, uint8_t tag_field, uint8_t tag_check)
{
    bool flag_tag_correct = true;
    char tag_in[32];

    if (tag_field)
    {

        updl_read_bytes(tag_in, fp, TAG_LENGTH);

        if (tag_check)
        {

            if (strncmp(tag_in, tag_name, TAG_LENGTH) == 0)
                flag_tag_correct = true;
            else
                flag_tag_correct = false;
        }
    }

    if (flag_tag_correct)
    {
        updl_Debug("%s:", tag_name);

        switch (dtype)
        {

        case Dtype_char:
            // target_t = (char *)target;
            updl_read_bytes(data, fp, STRING_FIELD_LENGTH);
            updl_Debug2("%.16s\n", (char *)data);
            break;

        case Dtype_bool:
            for (size_t i = 0; i < count; i++)
            {
                updl_read_bytes((uint16_t *)data + i, fp, sizeof(uint16_t));
                updl_Debug2("%d ", *(uint16_t *)((uint16_t *)data + i));
            }
            updl_Debug2("\n");
            break;

        case Dtype_uint8_t:
            for (size_t i = 0; i < count; i++)
            {
                updl_read_bytes((uint8_t *)data + i, fp, sizeof(uint8_t));
                updl_Debug2("%d ", *(uint8_t *)(data + i));
            }
            updl_Debug2("\n");

            break;

        case Dtype_uint16_t:
            for (size_t i = 0; i < count; i++)
            {
                updl_read_bytes((uint16_t *)data + i, fp, sizeof(uint16_t));
                updl_Debug2("%d ", *(uint16_t *)(((uint16_t *)data) + i));
            }
            updl_Debug2("\n");

            break;

        case Dtype_uint32_t:
            for (size_t i = 0; i < count; i++)
            {
                updl_read_bytes((uint32_t *)data + i, fp, Size_of(dtype));
                updl_Debug2("%d ", *(uint32_t *)(data + i));
            }
            updl_Debug2("\n");

            break;

        case Dtype_int8_t:
            for (size_t i = 0; i < count; i++)
            {
                updl_read_bytes((int8_t *)data + i, fp, sizeof(int8_t));
                updl_Debug2("%d ", *(int8_t *)(data + i));
            }
            updl_Debug2("\n");

            break;

        case Dtype_int16_t:
            for (size_t i = 0; i < count; i++)
            {
                updl_read_bytes((int16_t *)data + i, fp, sizeof(int16_t));
                updl_Debug2("%d ", *(int16_t *)(data + i));
            }
            updl_Debug2("\n");

            break;

        case Dtype_int32_t:
            for (size_t i = 0; i < count; i++)
            {
                updl_read_bytes((int32_t *)data + i, fp, sizeof(int32_t));
                updl_Debug2("%d ", *(int32_t *)(data + i));
            }
            updl_Debug2("\n");

            break;

        case Dtype_float32_t:
            for (size_t i = 0; i < count; i++)
            {
                updl_read_bytes((float *)data + i, fp, sizeof(float));
                updl_Debug2("%.6f ", *(float *)(data + i));
            }
            updl_Debug2("\n");

            break;

        case Dtype_dtype_t:
            *fp += 16; // skip the string 16Bytes block
            updl_read_bytes(data, fp, 4);
            //*(dtype_t *)data = (dtype_t)*((int32_t *)*fp);
            //*fp+=2;
            updl_Debug2("%d\n", *(dtype_t *)data);

            break;

        case Dtype_ltype_t:
            *fp += 16; // skip the string 16Bytes block
            updl_read_bytes(data, fp, 4);
            //*(ltype_t *)data = (ltype_t)*((int32_t *)*fp);
            //*fp+=2;
            updl_Debug2("%d\n", *(ltype_t *)data);

            break;

        case Dtype_ptype_t:
            *fp += 16; // skip the string 16Bytes block
            updl_read_bytes(data, fp, 4);
            //*(ptype_t *)data = (ptype_t)*((int32_t *)*fp);
            //*fp+=2;
            updl_Debug2("%d\n", *(ptype_t *)data);

            break;

        case Dtype_atype_t:
            *fp += 16; // skip the string 16Bytes block
            updl_read_bytes(data, fp, 4);
            //*(atype_t *)data = (atype_t)*((int32_t *)*fp);
            //*fp+=2;
            updl_Debug2("%d\n", *(atype_t *)data);

            break;

        default:
            updl_Warning("Unsupported dtype: %d\n", dtype);
            break;
        }
    }
    else
    {
        updl_Error("Tag incorrect: (%.16s , %.16s)\n", tag_name, tag_in);
    }
}

uint8_t updl_load_weights(weights_t *pWeights, uint8_t **fp)
{
    pWeights->dtype = Dtype_int16_t;
    
    updl_load_data(&(pWeights->weight_shape_d), fp, Dtype_uint16_t, 1, "weight_shape_d", TAG_FIELD,
                  TAG_CHECK);

    updl_load_data((uint16_t *)&(pWeights->weight_shape), fp, Dtype_uint16_t,
                  pWeights->weight_shape_d, "weight_shape", TAG_FIELD, TAG_CHECK);

    uint16_t wsize = 1;
    for (size_t i = 0; i < pWeights->weight_shape_d; i++)
    {
        wsize = wsize * pWeights->weight_shape[i];
    }

    // Skip alignment padding that was added by serializer for hardware compatibility
    // The serializer writes padding bytes to ensure weight data starts at 4-byte aligned address
    uintptr_t current_addr = (uintptr_t)*fp;
    size_t padding_bytes = (4 - (current_addr % 4)) % 4;
    if (padding_bytes > 0) {
        updl_Debug("Skipping %d alignment padding bytes at position %p\n", padding_bytes, (void*)current_addr);
        *fp += padding_bytes;
    }
    
    // Verify alignment after padding skip
    uintptr_t aligned_addr = (uintptr_t)*fp;
    if (aligned_addr % 4 != 0) {
        updl_Warning("Weight data not 4-byte aligned after padding skip: %p\n", (void*)aligned_addr);
    } else {
        updl_Debug("Weight data properly aligned at: %p\n", (void*)aligned_addr);
    }

    // Simplified for int16_t only - NO real allocation, point to XIP address directly
    pWeights->weight = (int16_t *)*fp;
    *fp += wsize * sizeof(int16_t);
    
    return 0;
}

uint8_t updl_load_layer_params(updl_layer_t **layer, uint8_t **fp)
{
    updl_load_data(&((*layer)->name), fp, Dtype_char, 1, "name", TAG_FIELD, TAG_CHECK);

    updl_load_data(&((*layer)->type), fp, Dtype_ltype_t, 1, "type", TAG_FIELD, TAG_CHECK);

    switch (((*layer)->type))
    {
    case Ltype_conv_1d:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 4, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 4, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->filters), fp, Dtype_uint16_t, 1, "filters", TAG_FIELD, TAG_CHECK);

        // Load kernel_size (only one value for Conv1D)
        uint16_t kernel_size_1d;
        updl_load_data(&kernel_size_1d, fp, Dtype_uint16_t, 1, "kernel_size", TAG_FIELD, TAG_CHECK);
        (*layer)->kernel_size[0] = kernel_size_1d;
        (*layer)->kernel_size[1] = 1; // Second dimension is 1 for 1D convolution

        // Load strides (only one value for Conv1D)
        uint16_t strides_1d;
        updl_load_data(&strides_1d, fp, Dtype_uint16_t, 1, "strides", TAG_FIELD, TAG_CHECK);
        (*layer)->strides[0] = strides_1d;
        (*layer)->strides[1] = 1; // Second dimension is 1 for 1D convolution

        updl_load_data(&((*layer)->padding), fp, Dtype_ptype_t, 1, "padding", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        // Load layer quantization parameters (separate weight and activation)
        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        updl_load_weights(&((*layer)->weights), fp);
        updl_load_weights(&((*layer)->bias), fp);

        break;

    case Ltype_conv_2d:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 4, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 4, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->filters), fp, Dtype_uint16_t, 1, "filters", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->kernel_size), fp, Dtype_uint16_t, 2, "kernel_size", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->strides), fp, Dtype_uint16_t, 2, "strides", TAG_FIELD, TAG_CHECK);

        updl_load_data(&((*layer)->padding), fp, Dtype_ptype_t, 1, "padding", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        // Load layer quantization parameters (separate weight and activation)
        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        updl_load_weights(&((*layer)->weights), fp);
        updl_load_weights(&((*layer)->bias), fp);

        break;

    case Ltype_depthwise_conv_2d:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 4, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 4, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        // Note: For depthwise conv, filters field represents depth_multiplier
        // or you can load it separately if stored as depth_multiplier in the model file
        updl_load_data(&((*layer)->depth_multiplier), fp, Dtype_uint16_t, 1, "depth_multiplier",
                      TAG_FIELD, TAG_CHECK);

        updl_load_data(&((*layer)->kernel_size), fp, Dtype_uint16_t, 2, "kernel_size", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->strides), fp, Dtype_uint16_t, 2, "strides", TAG_FIELD, TAG_CHECK);

        updl_load_data(&((*layer)->padding), fp, Dtype_ptype_t, 1, "padding", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        // Load layer quantization parameters (separate weight and activation)
        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        updl_load_weights(&((*layer)->weights), fp);
        updl_load_weights(&((*layer)->bias), fp);

        break;

    case Ltype_max_pooling_2d:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 4, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 4, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->pool_size), fp, Dtype_uint16_t, 2, "pool_size", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->strides), fp, Dtype_uint16_t, 2, "strides", TAG_FIELD, TAG_CHECK);

        updl_load_data(&((*layer)->padding), fp, Dtype_ptype_t, 1, "padding", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        break;

    case Ltype_average_pooling_2d:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 4, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 4, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->pool_size), fp, Dtype_uint16_t, 2, "pool_size", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->strides), fp, Dtype_uint16_t, 2, "strides", TAG_FIELD, TAG_CHECK);

        updl_load_data(&((*layer)->padding), fp, Dtype_ptype_t, 1, "padding", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        break;

    case Ltype_dense:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 2, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 2, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->units), fp, Dtype_uint16_t, 1, "units", TAG_FIELD, TAG_CHECK);

        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        // Load layer quantization parameters (separate weight and activation)
        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        updl_load_weights(&((*layer)->weights), fp);
        updl_load_weights(&((*layer)->bias), fp);

        break;

    case Ltype_flatten:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 4, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 2, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        // Load layer quantization parameters (separate weight and activation)
        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        break;

    case Ltype_lambda:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 2, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 2, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        break;

    case Ltype_add:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 4, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 4, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        // Load layer quantization parameters (no weights for Add layer)
        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        // Add layers can have bias data for the second input tensor
        updl_load_weights(&((*layer)->bias), fp);

        break;

    case Ltype_softmax:
        updl_load_data(&((*layer)->input_shape), fp, Dtype_uint16_t, 4, "input_shape", TAG_FIELD,
                      TAG_CHECK);
        updl_load_data(&((*layer)->output_shape), fp, Dtype_uint16_t, 4, "output_shape", TAG_FIELD,
                      TAG_CHECK);

        updl_load_data(&((*layer)->activation), fp, Dtype_atype_t, 1, "activation", TAG_FIELD,
                      TAG_CHECK);

        // Load layer quantization parameters  
        updl_load_data(&((*layer)->act_scale), fp, Dtype_float32_t, 1, "act_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->act_zp), fp, Dtype_int16_t, 1, "act_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_scale), fp, Dtype_float32_t, 1, "weight_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->weight_zp), fp, Dtype_int16_t, 1, "weight_zp", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_scale), fp, Dtype_float32_t, 1, "bias_scale", TAG_FIELD, TAG_CHECK);
        updl_load_data(&((*layer)->bias_zp), fp, Dtype_int16_t, 1, "bias_zp", TAG_FIELD, TAG_CHECK);

        break;

    default:
        updl_Warning("Unsupported layer type: %d encountered while loading layer params.\n",
                     ((*layer)->type));
        break;
    }

    return 0;
}
