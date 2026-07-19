alaya_cc_target(artifact_disk_flat_segment_generator SRCS artifact_disk_flat_segment_generator.cpp)
alaya_cc_target(artifact_memory_qg_generator SRCS artifact_memory_qg_generator.cpp)
if(ALAYA_ENABLE_LASER)
  alaya_cc_target(
    artifact_collection_qg_laser_generator
    LASER
    SRCS artifact_collection_qg_laser_generator.cpp
  )
endif()
