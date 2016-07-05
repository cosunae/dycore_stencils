
cuda_add_executable( vertical_advection vertical_advection.cu vertical_advection_benchmark.cu ../Options.cpp  ${COMMON_INC} ${GTEST_FILES})
target_link_libraries(vertical_advection ${exe_LIBS} )

cuda_add_executable( vertical_advection_split vertical_advection_split.cu vertical_advection_benchmark.cu ../Options.cpp  ${COMMON_INC} ${GTEST_FILES})
target_link_libraries(vertical_advection_split ${exe_LIBS} )

cuda_add_executable( vertical_advection_cache vertical_advection.cu vertical_advection_benchmark.cu ../Options.cpp  ${COMMON_INC} ${GTEST_FILES} OPTIONS "-DUSE_CACHE_")
target_link_libraries(vertical_advection_cache ${exe_LIBS} )

cuda_add_executable( vertical_advection_split_cache vertical_advection_split.cu vertical_advection_benchmark.cu ../Options.cpp  ${COMMON_INC} ${GTEST_FILES} OPTIONS "-DUSE_CACHE_")
target_link_libraries(vertical_advection_split_cache ${exe_LIBS} )
