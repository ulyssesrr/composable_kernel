set(DEV_GR_CONV2D_BWD_W_INSTANCES)
if(DTYPES MATCHES "fp32" OR NOT DEFINED DTYPES)
    list(APPEND DEV_GR_CONV2D_BWD_W_INSTANCES device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f32_instance.cpp)
endif()
if(DTYPES MATCHES "fp16" OR NOT DEFINED DTYPES)
    list(APPEND DEV_GR_CONV2D_BWD_W_INSTANCES device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_f16_instance.cpp)
endif()
if(DTYPES MATCHES "bf16" OR NOT DEFINED DTYPES)
    list(APPEND DEV_GR_CONV2D_BWD_W_INSTANCES device_grouped_conv2d_bwd_weight_xdl_gnhwc_gkyxc_gnhwk_bf16_instance.cpp)
endif()

add_instance_library(device_grouped_conv2d_bwd_weight_instance ${DEV_GR_CONV2D_BWD_W_INSTANCES})

