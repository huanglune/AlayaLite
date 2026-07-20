// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::validate_options(const CollectionOptions &options,
                                                core::OperationStage stage) -> core::Status {
  if (options.root.empty() || options.dim == 0 || options.max_logical_id_bytes == 0 ||
      core::scalar_type_size(options.scalar_type) == 0 || options.build_threads == 0 ||
      options.max_neighbors == 0 || options.ef_construction == 0) {
    return error(core::StatusCode::invalid_argument,
                 stage,
                 core::StatusDetail::malformed_struct,
                 "canonical Collection schema/root/build parameters are invalid");
  }
  if (!checked_active_laser_capacity(options.auto_seal_rows).has_value()) {
    return error(core::StatusCode::invalid_argument,
                 stage,
                 core::StatusDetail::arithmetic_overflow,
                 "canonical Collection auto_seal_rows exceeds the active LASER capacity limit");
  }
  const auto metric_valid = options.metric == core::Metric::l2 ||
                            options.metric == core::Metric::inner_product ||
                            options.metric == core::Metric::cosine;
  const auto quantization_valid = options.quantization == CollectionQuantization::none ||
                                  options.quantization == CollectionQuantization::sq8 ||
                                  options.quantization == CollectionQuantization::sq4 ||
                                  options.quantization == CollectionQuantization::rabitq;
  if (!metric_valid || !quantization_valid) {
    return error(core::StatusCode::invalid_argument,
                 stage,
                 core::StatusDetail::malformed_struct,
                 "canonical Collection metric/quantization schema is invalid");
  }
  const auto algorithm_valid = options.target_algorithm == core::algorithm::flat ||
                               options.target_algorithm == core::algorithm::qg ||
                               options.target_algorithm == core::algorithm::laser;
  if (!algorithm_valid) {
    return error(core::StatusCode::not_supported,
                 stage,
                 core::StatusDetail::operation_slot_absent,
                 "canonical Collection target algorithm is unsupported");
  }
  // rabitq quantization is the RaBitQ-graph family's native (and only)
  // format -- both qg (in-memory) and laser (on-disk) always quantize this
  // way internally, so this cross-check covers both instead of qg alone.
  if (options.quantization == CollectionQuantization::rabitq &&
      options.target_algorithm != core::algorithm::qg &&
      options.target_algorithm != core::algorithm::laser) {
    return error(core::StatusCode::invalid_argument,
                 stage,
                 core::StatusDetail::malformed_struct,
                 "canonical Collection requires explicit index_type=qg or laser for rabitq");
  }
  if ((options.target_algorithm == core::algorithm::qg ||
       options.target_algorithm == core::algorithm::laser) &&
      options.quantization != CollectionQuantization::rabitq) {
    return error(core::StatusCode::invalid_argument,
                 stage,
                 core::StatusDetail::malformed_struct,
                 "canonical Collection qg/laser requires quantization=rabitq");
  }
  if ((options.quantization == CollectionQuantization::sq8 ||
       options.quantization == CollectionQuantization::sq4) &&
      options.scalar_type != core::ScalarType::float32) {
    return error(core::StatusCode::not_supported,
                 stage,
                 core::StatusDetail::unsupported_scalar_type,
                 "canonical Collection quantization requires float32 vectors");
  }
  // 2B active engine (ruling 7 / B-08). Default flat is always valid. The on-disk
  // mutable LASER active engine constrains the schema hard: L2 + float32 + rabitq +
  // dim in [33,2048] (non-power-of-two dims use the same FHT padding as sealed
  // LASER) + FastScan-legal max_neighbors in {32,64} (32-slot block packing,
  // QGScanner cap 64) + target_algorithm=laser (v1 single engine).
  if (options.active_engine != core::algorithm::flat &&
      options.active_engine != core::algorithm::laser) {
    return error(core::StatusCode::not_supported,
                 stage,
                 core::StatusDetail::operation_slot_absent,
                 "canonical Collection active engine is unsupported");
  }
#if !ALAYA_COLLECTION_HAS_ACTIVE_LASER
  // Capability admission must precede every filesystem mutation in create():
  // rejecting only inside create_active_laser_segment() would leave a persisted
  // schema/control layout that later blocks a flat retry with already_exists.
  if (options.active_engine == core::algorithm::laser) {
    return error(core::StatusCode::not_supported,
                 stage,
                 core::StatusDetail::operation_slot_absent,
                 "active LASER engine requires a Linux build with ALAYA_ENABLE_LASER");
  }
#endif
  if (options.active_engine == core::algorithm::laser) {
    const bool dimension_supported =
        ::alaya::disk::laser_importer_detail::dimension_supported_v1(options.dim);
    if (options.metric != core::Metric::l2 || options.scalar_type != core::ScalarType::float32 ||
        options.quantization != CollectionQuantization::rabitq || !dimension_supported ||
        (options.max_neighbors != 32 && options.max_neighbors != 64) ||
        options.target_algorithm != core::algorithm::laser) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::malformed_struct,
                   "active LASER engine requires metric=l2, float32, quantization=rabitq, "
                   "dim in [33,2048] (non-power-of-two dims use FHT padding), max_neighbors "
                   "in {32,64}, target_algorithm=laser");
    }
  }
  return core::Status::success();
}

auto Collection::active_laser_dir(const std::filesystem::path &root,
                                  std::uint64_t segment_id,
                                  std::uint64_t generation) -> std::filesystem::path {
  return root / ".alaya_internal" / "active_laser" /
         (internal::collection::detail::collection_segment_name(segment_id) + "_g" +
          std::to_string(generation));
}

void Collection::sweep_orphan_active_laser_dirs(
    const std::filesystem::path &root,
    const internal::collection::CollectionControlState &control_state) {
  std::error_code error;
  const auto active_root = root / ".alaya_internal" / "active_laser";
  if (!std::filesystem::is_directory(active_root, error)) {
    return;
  }
  std::set<std::filesystem::path> keep;
  keep.insert(
      active_laser_dir(root, control_state.active_segment_id, control_state.active_generation)
          .filename());
  if (control_state.phase == internal::collection::CollectionControlPhase::successor_active ||
      control_state.phase == internal::collection::CollectionControlPhase::building ||
      control_state.phase == internal::collection::CollectionControlPhase::manifest_published) {
    for (const auto &source : control_state.sources) {
      keep.insert(active_laser_dir(root, source.segment_id, source.generation).filename());
    }
  }
  std::vector<std::filesystem::path> orphans;
  for (const auto &entry : std::filesystem::directory_iterator(active_root, error)) {
    if (entry.is_directory(error) && !keep.contains(entry.path().filename())) {
      orphans.push_back(entry.path());
    }
  }
  for (const auto &orphan : orphans) {
    std::filesystem::remove_all(orphan, error);
  }
  if (!orphans.empty()) {
    try {
      platform::sync_directory_or_throw(active_root);
    } catch (...) {  // NOLINT(bugprone-empty-catch): durability of the unlink is
      // best-effort; a crash before the parent fsync just re-sweeps on next open.
    }
  }
}

[[nodiscard]] auto Collection::checked_active_laser_capacity(std::uint64_t auto_seal_rows)
    -> std::optional<std::size_t> {
  if (auto_seal_rows == 0) {
    return 4096;
  }
  if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
    if (auto_seal_rows > std::numeric_limits<std::size_t>::max()) {
      return std::nullopt;
    }
  }
  const auto threshold = static_cast<std::size_t>(auto_seal_rows);
  std::size_t doubled{};
  std::size_t capacity{};
  if (alaya_mul_overflow(threshold, std::size_t{2}, &doubled) ||
      alaya_add_overflow(doubled, std::size_t{4096}, &capacity) || capacity <= threshold) {
    return std::nullopt;
  }
  return capacity;
}

[[nodiscard]] auto Collection::create_active_laser_segment(const CollectionOptions &options,
                                                           std::uint64_t segment_id,
                                                           std::uint64_t generation)
    -> core::Status {
#if ALAYA_COLLECTION_HAS_ACTIVE_LASER
  try {
    ::alaya::disk::MutableLaserSegment::
        create_empty(active_laser_dir(options.root, segment_id, generation),
                     internal::collection::detail::collection_segment_name(segment_id),
                     options.dim,
                     options.dim,
                     options.max_neighbors,
                     options.metric);
    return core::Status::success();
  } catch (...) {
    return core::status_from_exception(core::OperationStage::build);
  }
#else
  (void)options;
  (void)segment_id;
  (void)generation;
  return error(core::StatusCode::not_supported,
               core::OperationStage::build,
               core::StatusDetail::operation_slot_absent,
               "active LASER engine requires a Linux build with ALAYA_ENABLE_LASER");
#endif
}

[[nodiscard]] auto Collection::make_active_registration(const CollectionOptions &options,
                                                        std::uint64_t segment_id,
                                                        std::uint64_t generation)
    -> core::Result<internal::collection::SegmentRegistration> {
  const internal::collection::CollectionSchema schema{options.dim,
                                                      options.metric,
                                                      options.scalar_type,
                                                      options.max_logical_id_bytes};
  if (options.active_engine != core::algorithm::laser) {
    return internal::collection::detail::make_canonical_flat_registration(schema,
                                                                          segment_id,
                                                                          generation);
  }
#if ALAYA_COLLECTION_HAS_ACTIVE_LASER
  const auto dir = active_laser_dir(options.root, segment_id, generation);
  if (!std::filesystem::exists(dir / "manifest.txt")) {
    return error(core::StatusCode::corruption,
                 core::OperationStage::open,
                 core::StatusDetail::malformed_struct,
                 "active LASER segment directory is missing: " + dir.string());
  }
  try {
    laser::UpdateParams params;
    params.max_points = active_laser_capacity(options);
    params.enable_pid_reuse = true;
    auto segment =
        std::make_shared<::alaya::disk::MutableLaserSegment>(dir,
                                                             params,
                                                             laser::ResidencyMode::kPagedPool,
                                                             /*allow_empty=*/true);
    return internal::collection::detail::make_active_laser_registration(std::move(segment),
                                                                        schema,
                                                                        segment_id,
                                                                        generation);
  } catch (...) {
    return core::status_from_exception(core::OperationStage::open);
  }
#else
  return error(core::StatusCode::not_supported,
               core::OperationStage::open,
               core::StatusDetail::operation_slot_absent,
               "active LASER engine requires a Linux build with ALAYA_ENABLE_LASER");
#endif
}
}  // namespace alaya
