# Compile-only closure check: these core entry points must remain usable without LASER definitions, headers, or its
# libaio backend dependency.
alaya_cc_target(core_header_closure SRCS core_header_closure.cpp)
