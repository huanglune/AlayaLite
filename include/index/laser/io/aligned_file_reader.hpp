/**
 * @file aligned_file_reader.hpp
 * @brief Linux AIO (Asynchronous I/O) wrapper for high-performance disk access.
 *
 * This module provides direct disk I/O capabilities using Linux kernel AIO,
 * which is essential for achieving predictable, low-latency disk reads in
 * disk-based vector search systems.
 *
 * Key Design Decisions:
 * - Uses O_DIRECT to bypass the OS page cache
 * - Uses Linux AIO (libaio) for asynchronous, non-blocking I/O
 * - Requires 512-byte alignment for all buffers, offsets, and lengths
 */

#pragma once

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

#include <vector>
#include <atomic>

#include <malloc.h>
#include <mutex>
#include <thread>
#include <iostream>
// #include "tsl/robin_map.h"
#include <map>
#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>

constexpr size_t kDefaultAioEventsPerThread = 32;

#ifndef ROUND_UP
#define ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y)
#endif  // !ROUND_UP

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)

constexpr size_t kMaxIoDepth = 128;


using IOContext = io_context_t;


/**
 * @brief Represents a single aligned read request for O_DIRECT I/O.
 *
 * All fields must be 512-byte aligned to satisfy O_DIRECT requirements.
 * This alignment constraint comes from the Linux kernel's direct I/O path,
 * which bypasses the page cache and requires sector-aligned access.
 */
struct AlignedRead {
    uint64_t offset;  // File offset to read from (must be 512-aligned)  // NOLINT(readability-identifier-naming)
    uint64_t len;     // Number of bytes to read (must be 512-aligned)  // NOLINT(readability-identifier-naming)
    uint64_t id;      // User-defined ID for tracking this request              // NOLINT(readability-identifier-naming)
    void*    buf;     // Destination buffer (must be 512-aligned)            // NOLINT(readability-identifier-naming)

    AlignedRead() : offset(0), len(0), id(0), buf(nullptr) {
    }

    AlignedRead(uint64_t offset, uint64_t len, uint64_t id, void* buf)
        : offset(offset), len(len), id(id), buf(buf) {
        assert(IS_512_ALIGNED(offset));
        assert(IS_512_ALIGNED(len));
        assert(IS_512_ALIGNED(buf));
    }
};

class AlignedFileReader {
protected:
    std::map<std::thread::id, IOContext> ctx_map_;
    std::mutex                           ctx_mut_;

public:
    // returns the thread-specific context
    // returns (io_context_t)(-1) if thread is not registered
    virtual auto get_ctx() -> IOContext& = 0;

    virtual ~AlignedFileReader() = default;

    // register thread-id for a context
    virtual auto register_thread() -> void = 0;
    virtual auto register_thread(size_t max_events) -> void = 0;
    // de-register thread-id for a context
    virtual auto deregister_thread() -> void = 0;
    virtual auto deregister_all_threads() -> void = 0;

    // Open & close ops
    // Blocking calls
    virtual auto open(const std::string& fname) -> void = 0;
    virtual auto close() -> void = 0;

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    virtual auto read(std::vector<AlignedRead>& read_reqs, IOContext& ctx, bool async) -> void = 0;
    auto read(std::vector<AlignedRead>& read_reqs, IOContext& ctx) -> void {
        read(read_reqs, ctx, false);
    }
    virtual auto submit_reqs(std::vector<AlignedRead>& read_reqs, IOContext& ctx) -> int = 0;
    virtual auto get_events(IOContext& ctx, int n_ops) -> void = 0;
};

class LinuxAlignedFileReader : public AlignedFileReader {
private:
    int          file_desc_;
    io_context_t bad_ctx_ = nullptr;

public:
    LinuxAlignedFileReader();
    ~LinuxAlignedFileReader() override;

    auto get_ctx() -> IOContext& override;

    // register thread-id for a context
    auto register_thread() -> void override;
    auto register_thread(size_t max_events) -> void override;

    // de-register thread-id for a context
    auto deregister_thread() -> void override;
    auto deregister_all_threads() -> void override;

    // Open & close ops
    // Blocking calls
    auto open(const std::string &fname) -> void override;
    auto close() -> void override;

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    auto read(std::vector<AlignedRead> &read_reqs, IOContext &ctx,
            bool async) -> void override;

    auto submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx) -> int override;
    auto submit_reqs(AlignedRead* read_reqs, size_t count, IOContext &ctx) -> int;
    auto submit_reqs(AlignedRead* read_reqs, size_t count, IOContext &ctx,
                    iocb* cb_buf, iocb** cbs_buf) -> int;
    auto get_events(IOContext &ctx, int n_ops) -> void override;
};


using io_event_t = struct io_event;
using iocb_t = struct iocb;

inline auto execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs,
                size_t max_events = kDefaultAioEventsPerThread, uint64_t n_retries = 0) -> void {
    #ifdef DEBUG
        for (auto &req : read_reqs) {
            assert(IS_ALIGNED(req.len, 512));
            assert(IS_ALIGNED(req.offset, 512));
            assert(IS_ALIGNED(req.buf, 512));
        }
    #endif

    // break-up requests into chunks of size max_events each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), max_events) / max_events;
    for (uint64_t iter = 0; iter < n_iters; iter++) {
        uint64_t n_ops =
            std::min(static_cast<uint64_t>(read_reqs.size()) - (iter * max_events),
                        static_cast<uint64_t>(max_events));
        std::vector<iocb_t *>    cbs(n_ops, nullptr);
        std::vector<io_event_t>  evts(n_ops);
        std::vector<struct iocb> cb(n_ops);
        for (uint64_t j = 0; j < n_ops; j++) {
            io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * max_events].buf,
                        read_reqs[j + iter * max_events].len,
                        static_cast<off_t>(read_reqs[j + iter * max_events].offset));
        }

        // initialize `cbs` using `cb` array
        for (uint64_t i = 0; i < n_ops; i++) {
            cbs[i] = cb.data() + i;
        }

        uint64_t n_tries = 0;
        while (n_tries <= n_retries) {
            // issue reads
            int64_t ret = io_submit(ctx, static_cast<int64_t>(n_ops), cbs.data());
            // if requests didn't get accepted
            if (ret != static_cast<int64_t>(n_ops)) {
                std::cerr << "io_submit() failed; returned " << ret
                            << ", expected=" << n_ops << ", ernno=" << errno << "="
                            << ::strerror(static_cast<int>(-ret)) << ", try #" << n_tries + 1;
                std::cout << "ctx: " << ctx << "\n";
                exit(-1);
            } else {
                // wait on io_getevents
                ret = io_getevents(ctx, static_cast<int64_t>(n_ops), static_cast<int64_t>(n_ops), evts.data(),
                                    nullptr);
                // if requests didn't complete
                if (ret != static_cast<int64_t>(n_ops)) {
                    std::cerr << "io_getevents() failed; returned " << ret
                            << ", expected=" << n_ops << ", ernno=" << errno << "="
                            << ::strerror(static_cast<int>(-ret)) << ", try #" << n_tries + 1;
                    exit(-1);
                } else {
                    break;
                }
            }
        }
    }
}


inline LinuxAlignedFileReader::LinuxAlignedFileReader() {
    this->file_desc_ = -1;
}

inline LinuxAlignedFileReader::~LinuxAlignedFileReader() {
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc_, F_GETFD);
    if (ret == -1) {
        if (errno != EBADF) {
            std::cerr << "close() not called" << '\n';
            // close file desc
            ret = ::close(this->file_desc_);
            // error checks
            if (ret == -1) {
                std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                        << ":" << ::strerror(errno) << '\n';
            }
        }
    }
}

inline auto LinuxAlignedFileReader::get_ctx() -> IOContext& {
    std::unique_lock<std::mutex> lk(ctx_mut_);
    // perform checks only in DEBUG mode
    if (ctx_map_.find(std::this_thread::get_id()) == ctx_map_.end()) {
        std::cerr << "bad thread access; returning -1 as io_context_t" << '\n';
        return this->bad_ctx_;
    }
    return ctx_map_[std::this_thread::get_id()];

}

inline auto LinuxAlignedFileReader::register_thread() -> void {
    register_thread(kDefaultAioEventsPerThread);
}

inline auto LinuxAlignedFileReader::register_thread(size_t max_events) -> void {
    auto                         my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut_);
    if (ctx_map_.find(my_id) != ctx_map_.end()) {
        std::cerr << "multiple calls to register_thread from the same thread"
                << '\n';
        return;
    }
    io_context_t ctx = nullptr;
    int          ret = io_setup(static_cast<int>(max_events), &ctx);
    if (ret != 0) {
        lk.unlock();
        if (errno == EAGAIN) {
            throw std::runtime_error(
                "io_setup() failed with EAGAIN: AIO context limit reached. "
                "Check: cat /proc/sys/fs/aio-nr /proc/sys/fs/aio-max-nr. "
                "Fix: sudo sysctl -w fs.aio-max-nr=1048576");
        }
        if (errno == ENOMEM) {
            throw std::runtime_error(
                "io_setup() failed: insufficient kernel memory for AIO context");
        }
        throw std::runtime_error(
            std::string("io_setup() failed; returned ") + std::to_string(ret) +
            ", errno=" + std::to_string(errno) + ":" + ::strerror(errno));
    }
    ctx_map_[my_id] = ctx;
    lk.unlock();
}

inline auto LinuxAlignedFileReader::deregister_thread() -> void {
    auto                         my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut_);
    assert(ctx_map_.find(my_id) != ctx_map_.end());

    lk.unlock();
    io_context_t ctx = this->get_ctx();
    io_destroy(ctx);
    lk.lock();
    ctx_map_.erase(my_id);
    lk.unlock();
}

inline auto LinuxAlignedFileReader::deregister_all_threads() -> void {
    std::unique_lock<std::mutex> lk(ctx_mut_);
    for (auto & x : ctx_map_) {
        io_context_t ctx = x.second;
        io_destroy(ctx);
    }
    ctx_map_.clear();
}

inline auto LinuxAlignedFileReader::open(const std::string &fname) -> void {
    // O_DIRECT: Bypass the OS page cache and read directly from disk.
    //   Why? For large-scale vector search, the working set far exceeds RAM.
    //   Without O_DIRECT, the OS would try to cache data, causing:
    //   1. Unpredictable memory usage (cache competes with our application memory)
    //   2. Double-buffering overhead (data copied: disk -> page cache -> user buffer)
    //   3. Cache thrashing (random access patterns pollute the page cache)
    //   With O_DIRECT, we have full control over memory and avoid these issues.
    //
    // O_RDONLY: Open file for reading only (index files are read-only during search).
    int flags = O_DIRECT | O_RDONLY;
    this->file_desc_ = ::open(fname.c_str(), flags);
    assert(this->file_desc_ != -1);
}

inline auto LinuxAlignedFileReader::close() -> void {
    ::fcntl(this->file_desc_, F_GETFD);
    ::close(this->file_desc_);
}

inline auto LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs,
                                IOContext &ctx, bool async) -> void {
    if (async) {
        std::cout << "Async currently not supported in linux." << '\n';
    }
    assert(this->file_desc_ != -1);
    execute_io(ctx, this->file_desc_, read_reqs);
}


/**
 * @brief Submit read requests asynchronously (non-blocking).
 *
 * Submits I/O requests to the kernel AIO queue and returns immediately
 * without waiting for completion. Use io_getevents() to check for completion.
 * This enables overlapping disk I/O with CPU computation for better throughput.
 *
 * @param read_reqs Vector of aligned read requests to submit
 * @param ctx       Thread-local AIO context
 * @return Number of requests successfully submitted
 */
inline auto LinuxAlignedFileReader::submit_reqs(std::vector<AlignedRead> &read_reqs,
                                        IOContext &ctx) -> int {
    assert(this->file_desc_ != -1);

    size_t n_ops = read_reqs.size();
    std::vector<struct iocb *> cbs(n_ops, nullptr);
    std::vector<io_event>  evts(n_ops);
    std::vector<struct iocb> cb(n_ops);
    for (size_t j = 0; j < n_ops; j++) {
        io_prep_pread(cb.data() + j, this->file_desc_, read_reqs[j].buf,
                    read_reqs[j].len,
                    static_cast<off_t>(read_reqs[j].offset));
    }
    for (size_t i = 0; i < n_ops; i++) {
        cbs[i] = cb.data() + i;
        cbs[i]->data = reinterpret_cast<void*>(static_cast<uintptr_t>(read_reqs[i].id));  // NOLINT(performance-no-int-to-ptr)
    }

    int ret = io_submit(ctx, static_cast<int64_t>(n_ops), cbs.data());
    if (ret != static_cast<int>(n_ops)) {
        std::cerr << "io_submit() failed; returned " << ret
                << ", expected=" << n_ops << ", ernno=" << errno << "="
                << ::strerror(-ret) << '\n';
        std::cout << "ctx: " << ctx << "\n";
        exit(-1);
    }
    return static_cast<int>(n_ops);
}

inline auto LinuxAlignedFileReader::submit_reqs(AlignedRead* read_reqs, size_t count,
                                        IOContext &ctx) -> int {
    assert(this->file_desc_ != -1);

    std::vector<struct iocb *> cbs(count, nullptr);
    std::vector<struct iocb> cb(count);
    for (size_t j = 0; j < count; j++) {
        io_prep_pread(cb.data() + j, this->file_desc_, read_reqs[j].buf,
                    read_reqs[j].len,
                    static_cast<off_t>(read_reqs[j].offset));
    }
    for (size_t i = 0; i < count; i++) {
        cbs[i] = cb.data() + i;
        cbs[i]->data = reinterpret_cast<void*>(static_cast<uintptr_t>(read_reqs[i].id));  // NOLINT(performance-no-int-to-ptr)
    }

    int ret = io_submit(ctx, static_cast<int64_t>(count), cbs.data());
    if (ret != static_cast<int>(count)) {
        std::cerr << "io_submit() failed; returned " << ret
                << ", expected=" << count << ", errno=" << errno << "="
                << ::strerror(-ret) << '\n';
        std::cout << "ctx: " << ctx << "\n";
        exit(-1);
    }
    return static_cast<int>(count);
}

inline auto LinuxAlignedFileReader::submit_reqs(AlignedRead* read_reqs, size_t count,
                                        IOContext &ctx,
                                        iocb* cb_buf, iocb** cbs_buf) -> int {
    assert(this->file_desc_ != -1);

    for (size_t j = 0; j < count; j++) {
        io_prep_pread(&cb_buf[j], this->file_desc_, read_reqs[j].buf,
                    read_reqs[j].len,
                    static_cast<off_t>(read_reqs[j].offset));
        cb_buf[j].data = reinterpret_cast<void*>(static_cast<uintptr_t>(read_reqs[j].id));  // NOLINT(performance-no-int-to-ptr)
        cbs_buf[j] = &cb_buf[j];
    }

    int ret = io_submit(ctx, static_cast<int64_t>(count), cbs_buf);
    if (ret != static_cast<int>(count)) {
        std::cerr << "io_submit() failed; returned " << ret
                << ", expected=" << count << ", errno=" << errno << "="
                << ::strerror(-ret) << '\n';
        std::cout << "ctx: " << ctx << "\n";
        exit(-1);
    }
    return static_cast<int>(count);
}

inline auto LinuxAlignedFileReader::get_events(IOContext& ctx, int n_ops) -> void {
    std::vector<io_event> evts(n_ops);
    auto ret = io_getevents(ctx, static_cast<int64_t>(n_ops), static_cast<int64_t>(n_ops),
                            evts.data(), nullptr);
    if (ret != static_cast<int64_t>(n_ops)) {
        std::cerr << "io_getevents() failed; returned " << ret << '\n';
        exit(-1);
    }
}
