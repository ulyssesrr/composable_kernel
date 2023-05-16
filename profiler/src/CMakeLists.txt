# ckProfiler
set(PROFILER_SOURCES
    profiler.cpp
    profile_gemm.cpp
)

set(PROFILER_EXECUTABLE ckProfiler)

add_executable(${PROFILER_EXECUTABLE} ${PROFILER_SOURCES})
target_compile_options(${PROFILER_EXECUTABLE} PRIVATE -Wno-global-constructors)

target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE utility)
target_link_libraries(${PROFILER_EXECUTABLE} PRIVATE device_gemm_instance)
rocm_install(TARGETS ${PROFILER_EXECUTABLE} COMPONENT profiler)
