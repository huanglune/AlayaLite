/**
 * @file qg_query.hpp
 * @brief Query preprocessing for RaBitQ-based ANN search.
 *
 * Zero-alloc refactored: QGQuery is now a stack-only value type.
 * - Removed owned std::vector<uint8_t> lut_ member
 * - All scratch buffers (rotated_query, byte_query, lut) come from LaserSearchContext
 * - sizeof(QGQuery) <= 32 bytes
 */

#pragma once

#include <cstdint>

#include "index/laser/laser_common.hpp"
#include "index/laser/laser_search_context.hpp"
#include "index/laser/quantization/scalar_quantize.hpp"
#include "index/laser/qg_scanner.hpp"
#include "index/laser/transform/fht_rotator.hpp"

namespace symqg {

/**
 * @brief Stack-only query object for RaBitQ distance computation.
 *
 * Contains only scalar fields and a const float* pointer. No heap memory.
 * All buffer arguments come from LaserSearchContext.
 */
class QGQuery {
   private:
    const float* query_data_ = nullptr;
    size_t padded_dim_ = 0;
    float width_ = 0;
    float lower_val_ = 0;
    float upper_val_ = 0;
    int32_t sumq_ = 0;
    float sqr_qr_ = 0;

   public:
    explicit QGQuery(const float* q, size_t padded_dim)
        : query_data_(q), padded_dim_(padded_dim) {}

    /**
     * @brief Prepare query using external buffers from SearchContext.
     * Zero heap allocations: rotated_query, byte_query, and lut come from ctx.
     */
    void query_prepare(
        const FHTRotator& rotator,
        const QGScanner& scanner,
        LaserSearchContext& ctx
    ) {
        float* rd_query = ctx.rotated_query();
        rotator.rotate(query_data_, rd_query);

        uint8_t* byte_query = ctx.byte_query();
        scalar::data_range(rd_query, padded_dim_, lower_val_, upper_val_);
        width_ = (upper_val_ - lower_val_) / ((1 << QG_BQUERY) - 1);
        scalar::quantize(
            byte_query, rd_query, padded_dim_, lower_val_, width_, sumq_
        );

        scanner.pack_lut(byte_query, ctx.lut());
    }

    [[nodiscard]] auto width() const -> const float& { return width_; }
    [[nodiscard]] auto lower_val() const -> const float& { return lower_val_; }
    [[nodiscard]] auto sumq() const -> const int32_t& { return sumq_; }
    [[nodiscard]] auto query_data() const -> const float* { return query_data_; }

    auto set_sqr_qr(float sqr_qr) -> void { sqr_qr_ = sqr_qr; }
    [[nodiscard]] auto sqr_qr() const -> float { return sqr_qr_; }
};

}  // namespace symqg
