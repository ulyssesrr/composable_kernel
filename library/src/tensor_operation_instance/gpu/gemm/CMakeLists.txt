add_instance_library(device_gemm_instance
   # device_gemm_xdl_f64_f64_f64_mk_kn_mn_instance.cpp
   # device_gemm_xdl_f64_f64_f64_mk_nk_mn_instance.cpp
   # device_gemm_xdl_f64_f64_f64_km_kn_mn_instance.cpp
   # device_gemm_xdl_f64_f64_f64_km_nk_mn_instance.cpp
   # device_gemm_xdl_f32_f32_f32_mk_kn_mn_instance.cpp
   # device_gemm_xdl_f32_f32_f32_mk_nk_mn_instance.cpp
   # device_gemm_xdl_f32_f32_f32_km_kn_mn_instance.cpp
   # device_gemm_xdl_f32_f32_f32_km_nk_mn_instance.cpp
   # device_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp
   device_gemm_xdl_f16_f16_f16_mk_nk_mn_instance.cpp
   # device_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp
   # device_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_i8_i8_i8_mk_kn_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_i8_i8_i8_mk_nk_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_i8_i8_i8_km_kn_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_i8_i8_i8_km_nk_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp
   device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_f32_f32_f32_mk_nk_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instance.cpp
   # device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instance.cpp
   device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instance.cpp
   # device_gemm_dl_f32_f32_f32_mk_kn_mn_instance.cpp
   # device_gemm_dl_f32_f32_f32_mk_nk_mn_instance.cpp
   # device_gemm_dl_f32_f32_f32_km_kn_mn_instance.cpp
   # device_gemm_dl_f32_f32_f32_km_nk_mn_instance.cpp
   # device_gemm_dl_f16_f16_f16_mk_kn_mn_instance.cpp
   # device_gemm_dl_f16_f16_f16_mk_nk_mn_instance.cpp
   # device_gemm_dl_f16_f16_f16_km_kn_mn_instance.cpp
   # device_gemm_dl_f16_f16_f16_km_nk_mn_instance.cpp
   # device_gemm_dl_i8_i8_i8_mk_kn_mn_instance.cpp
   # device_gemm_dl_i8_i8_i8_mk_nk_mn_instance.cpp
   # device_gemm_dl_i8_i8_i8_km_kn_mn_instance.cpp
   # device_gemm_dl_i8_i8_i8_km_nk_mn_instance.cpp
)

set(ENABLE_IGLP_OPT OFF)
if(ENABLE_IGLP_OPT)
set_source_files_properties(device_gemm_xdl_f16_f16_f16_mk_nk_mn_instance.cpp PROPERTIES
  COMPILE_OPTIONS ";--save-temps;-Wno-gnu-line-marker;-mllvm;-amdgpu-enable-max-ilp-scheduling-strategy;"
  COMPILE_DEFINITIONS ";USE_IGLP_OPT;USE_WAVES_PER_EU;")
set_source_files_properties(device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp PROPERTIES
  COMPILE_OPTIONS ";--save-temps;-Wno-gnu-line-marker;-mllvm;-amdgpu-enable-max-ilp-scheduling-strategy;"
  COMPILE_DEFINITIONS ";USE_IGLP_OPT;USE_WAVES_PER_EU;")
set_source_files_properties(device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instance.cpp PROPERTIES
  COMPILE_OPTIONS ";--save-temps;-Wno-gnu-line-marker;-mllvm;-amdgpu-enable-max-ilp-scheduling-strategy;"
  COMPILE_DEFINITIONS ";USE_IGLP_OPT;USE_WAVES_PER_EU;")
endif(ENABLE_IGLP_OPT)
