# Compile-only public surface baseline. Keeping these as separate translation units proves that each entry header
# supplies its own transitive prerequisites.
alaya_cc_target(
  abstraction_public_header_golden
  SRCS ../build_matrix/public_graph_header.cpp ../build_matrix/public_space_header.cpp
       ../build_matrix/public_disk_header.cpp ../build_matrix/public_diskann_header.cpp
)

alaya_cc_target(artifact_diskann_generator SRCS artifact_diskann_generator.cpp)
alaya_cc_target(artifact_disk_flat_segment_generator SRCS artifact_disk_flat_segment_generator.cpp)
alaya_cc_target(artifact_disk_vamana_segment_generator SRCS artifact_disk_vamana_segment_generator.cpp)
alaya_cc_target(artifact_memory_qg_generator SRCS artifact_memory_qg_generator.cpp)
alaya_cc_target(artifact_memory_vamana_generator SRCS artifact_memory_vamana_generator.cpp)
alaya_cc_target(artifact_legacy_v1_generator SRCS artifact_legacy_v1_generator.cpp)
