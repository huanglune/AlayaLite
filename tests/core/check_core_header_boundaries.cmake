# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

if(NOT DEFINED ALAYA_SOURCE_DIR)
  message(FATAL_ERROR "ALAYA_SOURCE_DIR is required")
endif()

file(GLOB core_headers "${ALAYA_SOURCE_DIR}/include/core/*.hpp")
foreach(header IN LISTS core_headers)
  file(STRINGS "${header}" include_lines REGEX "^#[ \t]*include")
  foreach(line IN LISTS include_lines)
    if(line MATCHES "[<\"](index/|space/|storage/|storage_io/|metadata/|recovery/|sdk/|python/|utils/metric_type\\.hpp|utils/quantization_type\\.hpp)")
      message(FATAL_ERROR "core reverse dependency in ${header}: ${line}")
    endif()
  endforeach()
endforeach()
