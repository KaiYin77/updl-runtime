/* Copyright 2025 Upbeat, Inc */
/* SPDX-License-Identifier: Apache-2.0 */

// ============================================================================
// INCLUDES
// ============================================================================

#include "updl/updl_debug.h"

// ============================================================================
// DEBUG UTILITY FUNCTIONS
// ============================================================================

const char *updl_layer_type_to_string(ltype_t layer_type) {
    switch (layer_type) {
        case Ltype_conv_1d: return "Conv1D";
        case Ltype_conv_2d: return "Conv2D";
        case Ltype_depthwise_conv_2d: return "DepthwiseConv2D";
        case Ltype_dense: return "Dense";
        case Ltype_max_pooling_2d: return "MaxPooling2D";
        case Ltype_average_pooling_2d: return "AveragePooling2D";
        case Ltype_flatten: return "Flatten";
        case Ltype_softmax: return "Softmax";
        case Ltype_lambda: return "Lambda";
        default: return "Unknown";
    }
}

void updl_print_1d_array(const char *name, const int16_t *data, size_t size, 
                          float scale, int16_t zero_point, size_t elements_per_line) {
    if (!data || size == 0) {
        updl_Info("%s: [empty]\n", name);
        return;
    }
    
    updl_Info("%s[0:%d]=[\n", name, size);
    for (size_t i = 0; i < size; i++) {
        // Dequantize: value = scale * (quantized_value - zero_point)
        float dequant_value = scale * ((float)data[i] - (float)zero_point);
        updl_Info2("%.3f", dequant_value);
        
        if (i < size - 1) updl_Info2(", ");
        if ((i + 1) % elements_per_line == 0) updl_Info2("\n");
    }
    updl_Info2("]\n");
}

void updl_print_2d_array(const char *name, const int16_t *data, 
                         size_t height, size_t width, size_t channels,
                         float scale, int16_t zero_point, size_t max_channels) {
    if (!data || height == 0 || width == 0) {
        updl_Info("%s: [empty]\n", name);
        return;
    }
    
    size_t channels_to_print = (channels > max_channels) ? max_channels : channels;
    
    updl_Info("%s[C=0:%d][H=0:%d][W=0:%d]=[\n", name, channels_to_print, height, width);
    
    for (size_t c = 0; c < channels_to_print; c++) {
        updl_Info2("[ ");
        for (size_t h = 0; h < height; h++) {
            updl_Info2("[ ");
            for (size_t w = 0; w < width; w++) {
                size_t index = c * height * width + h * width + w;
                // Dequantize: value = scale * (quantized_value - zero_point)
                float dequant_value = scale * ((float)data[index] - (float)zero_point);
                updl_Info2("%.3f", dequant_value);
                if (w < width - 1) updl_Info2(", ");
            }
            updl_Info2("],\n");
        }
        if (c < channels_to_print - 1) updl_Info2("], \n");
    }
    updl_Info2("]\n");
}

void updl_print_layer_metadata(size_t layer_index, const updl_layer_t *layer) {
    if (!layer) {
        updl_Info("Layer %d: [invalid layer]\n", layer_index);
        return;
    }
    
    updl_Info("Layer %d %s:\n", layer_index, updl_layer_type_to_string(layer->type));
    updl_Info("- input.shape=(B=%d, H=%d, W=%d, C=%d)\n",  
              layer->input_shape[0], layer->input_shape[1], 
              layer->input_shape[2], layer->input_shape[3]);            
    updl_Info("- output.shape=(B=%d, H=%d, W=%d, C=%d)\n",  
              layer->output_shape[0], layer->output_shape[1], 
              layer->output_shape[2], layer->output_shape[3]);
    updl_Info("- weight_scale=%.8f, weight_zp=%d\n", layer->weight_scale, layer->weight_zp);
    updl_Info("- bias_scale=%.8f, bias_zp=%d\n", layer->bias_scale, layer->bias_zp);
    updl_Info("- act_scale=%.8f, act_zp=%d\n", layer->act_scale, layer->act_zp);
}

void updl_print_eff_quant_param(size_t layer_index, float eff_scale, float input_scale, 
                                   float weight_scale, float output_scale, int32_t eff_mult, 
                                   int16_t eff_shift, float eff_bias_scale, float bias_scale, 
                                   int32_t eff_bias_mult, int16_t eff_bias_shift) {
    updl_Info("[%d] eff_scale=%f = (input_scale=%f x weight_scale=%f) / output_scale=%f\n", 
              layer_index, eff_scale, input_scale, weight_scale, output_scale);
    updl_Info("[%d] (eff_mult=%d, eff_shift=%d) = updl_scale_to_multiplier_shift(%f)\n", 
              layer_index, eff_mult, eff_shift, eff_scale);
    updl_Info("[%d] eff_bias_scale=%f = bias_scale=%f / (input_scale=%f x weight_scale=%f)\n",
              layer_index, eff_bias_scale, bias_scale, input_scale, weight_scale);
    updl_Info("[%d] (eff_bias_mult=%d, eff_bias_shift=%d) = updl_bias_scale_to_multiplier_shift(%f)\n", 
              layer_index, eff_bias_mult, eff_bias_shift, eff_bias_scale);
}