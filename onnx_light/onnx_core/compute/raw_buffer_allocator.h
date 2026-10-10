// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "onnx_core/runtime/memory/simple_tensor.h"

#include <cstddef>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

/**
 * Abstract base class for RawBuffer allocators.
 *
 * Provides a virtual interface for allocating and freeing
 * :cpp:struct:`RawBuffer` instances, enabling custom memory management
 * strategies without changing call sites.
 *
 * All concrete allocators must implement :cpp:func:`Allocate`,
 * :cpp:func:`Free`, :cpp:func:`TotalAllocatedSize`,
 * :cpp:func:`PeakAllocatedSize`, and :cpp:func:`ResetPeak`.
 */
class RawBufferAllocator {
public:
  virtual ~RawBufferAllocator() = default;

  /**
   * Allocates a :cpp:struct:`RawBuffer` of at least ``n_bytes`` bytes.
   *
   * @param n_bytes  Number of bytes to allocate.
   * @returns        Pointer to the allocated :cpp:struct:`RawBuffer`, owned
   *                 by the allocator. The caller must release it with
   *                 :cpp:func:`Free`.
   * @throws std::bad_alloc (or a derived exception) if no free slot is
   *         available.
   */
  virtual RawBuffer *Allocate(size_t n_bytes) = 0;

  /**
   * Releases a previously allocated :cpp:struct:`RawBuffer`.
   *
   * @param buf  Pointer previously returned by :cpp:func:`Allocate`. Passing a
   *             pointer that was not returned by this allocator, or that has
   *             already been freed, is undefined behaviour.
   */
  virtual void Free(RawBuffer *buf) = 0;

  /**
   * Returns the total number of bytes across all currently allocated buffers.
   */
  virtual size_t TotalAllocatedSize() const = 0;

  /**
   * Returns the memory peak, i.e. the maximum value ever reached by
   * :cpp:func:`TotalAllocatedSize` since construction or since the last call to
   * :cpp:func:`ResetPeak`.
   */
  virtual size_t PeakAllocatedSize() const = 0;

  /**
   * Resets the memory peak to the current value of
   * :cpp:func:`TotalAllocatedSize`.
   */
  virtual void ResetPeak() = 0;
};

/**
 * Fixed-capacity pool allocator for :cpp:struct:`RawBuffer` instances.
 *
 * Manages a pre-allocated array of ``capacity`` :cpp:struct:`RawBuffer`
 * slots. A free-slot stack gives O(1) :cpp:func:`Allocate`, a pointer-to-index
 * map gives O(1) :cpp:func:`Free`, and running counters give O(1)
 * :cpp:func:`TotalAllocatedSize`, :cpp:func:`PeakAllocatedSize`, and
 * :cpp:func:`allocated_count`.
 *
 * @note This class is not thread-safe.
 */
class SimpleRawBufferAllocator : public RawBufferAllocator {
public:
  /**
   * Constructs an allocator with the given slot capacity.
   *
   * All ``capacity`` slot addresses are stable for the lifetime of this object
   * because ``buffers_`` is never resized after construction.
   *
   * @param capacity  Maximum number of :cpp:struct:`RawBuffer` instances that
   *                  can be alive at the same time.
   */
  explicit SimpleRawBufferAllocator(size_t capacity);

  /**
   * Pops a free slot from the stack, resizes it to ``n_bytes``, records the
   * pointer in the index map, and returns the slot pointer — O(1).
   *
   * @throws std::bad_alloc if all slots are already in use.
   */
  RawBuffer *Allocate(size_t n_bytes) override;

  /**
   * Looks up the slot index in the pointer map (O(1)), clears the slot,
   * removes it from the map, and pushes the index back onto the free stack.
   *
   * @throws std::invalid_argument if ``buf`` does not belong to this
   *         allocator.
   */
  void Free(RawBuffer *buf) override;

  /// Returns the sum of the sizes of all currently allocated buffers — O(1).
  size_t TotalAllocatedSize() const override;

  /**
   * Returns the memory peak — the maximum value ever reached by
   * :cpp:func:`TotalAllocatedSize` since construction or the last
   * :cpp:func:`ResetPeak` — O(1).
   */
  size_t PeakAllocatedSize() const override;

  /// Resets the memory peak to the current :cpp:func:`TotalAllocatedSize` — O(1).
  void ResetPeak() override;

  /// Returns the total number of slots managed by this allocator.
  size_t capacity() const noexcept;

  /// Returns the number of slots currently in use — O(1).
  size_t allocated_count() const noexcept;

private:
  std::vector<RawBuffer> buffers_;
  /// Stack of free slot indices; popped by Allocate, pushed back by Free.
  std::vector<size_t> free_slots_;
  /// Maps each live slot's address to its index for O(1) Free lookup.
  std::unordered_map<RawBuffer *, size_t> index_map_;
  size_t total_allocated_size_ = 0;
  size_t peak_allocated_size_ = 0;
  size_t allocated_count_ = 0;
};

/**
 * Fixed-slot arena that retains and reuses the storage of freed buffers.
 *
 * Free buffers are indexed by their retained capacity. Allocation selects the
 * smallest sufficient capacity, or grows the largest free buffer when every
 * retained buffer is too small. Slot addresses remain stable for the lifetime
 * of the arena.
 *
 * A retention cap bounds the total capacity kept on the free lists. When freeing
 * a buffer pushes the retained capacity above the cap, the arena releases the
 * storage of the least-recently-freed buffers until the retained capacity fits
 * again. Live buffers are never evicted. The cap defaults to unbounded.
 *
 * @note This class is not thread-safe.
 */
class ExecutionArena : public RawBufferAllocator {
public:
  /**
   * Constructs an arena with at most ``capacity`` simultaneously live buffers.
   *
   * @param capacity       Number of stable :cpp:struct:`RawBuffer` slots.
   * @param retention_cap  Maximum total capacity kept on the retained free
   *                       lists. Defaults to unbounded, disabling eviction.
   */
  explicit ExecutionArena(size_t capacity,
                          size_t retention_cap = std::numeric_limits<size_t>::max());

  /**
   * Allocates ``n_bytes`` from the smallest sufficient retained buffer.
   *
   * An unused slot is preferred when no retained buffer is large enough. Once
   * all slots have acquired storage, the largest undersized free buffer grows
   * to satisfy the request.
   *
   * @throws std::bad_alloc if all slots are currently live.
   */
  RawBuffer *Allocate(size_t n_bytes) override;

  /**
   * Returns a live buffer to its capacity bucket without releasing its storage.
   *
   * @throws std::invalid_argument if ``buf`` is not live in this arena.
   */
  void Free(RawBuffer *buf) override;

  /// Returns the logical bytes held by live buffers.
  size_t TotalAllocatedSize() const override;

  /// Returns the peak logical live-byte count.
  size_t PeakAllocatedSize() const override;

  /// Resets the live-byte peak to the current live-byte count.
  void ResetPeak() override;

  /// Returns the maximum number of simultaneously live buffers.
  size_t capacity() const noexcept;

  /// Returns the number of currently live buffers.
  size_t allocated_count() const noexcept;

  /// Returns the total capacity of free retained buffers.
  size_t RetainedSize() const noexcept;

  /// Returns the number of free retained buffers.
  size_t retained_count() const noexcept;

  /// Returns the maximum total capacity kept on the retained free lists.
  size_t retention_cap() const noexcept;

  /**
   * Sets the retention cap and evicts least-recently-freed buffers if needed.
   *
   * Lowering the cap below the current :cpp:func:`RetainedSize` immediately
   * releases the storage of the least-recently-freed buffers until the retained
   * capacity fits. Live buffers are never evicted.
   */
  void SetRetentionCap(size_t retention_cap) noexcept;

  /**
   * Releases the storage retained by every free buffer.
   *
   * Live buffers are left untouched. Each trimmed slot returns to the unused
   * pool and acquires fresh storage on a later allocation, so trimming only
   * gives back the capacity currently held on the retained free lists.
   *
   * @returns The number of retained bytes released.
   */
  size_t Trim() noexcept;

private:
  /// Records slot ``i`` as the most-recently-freed retained buffer.
  void TrackFreeSlot(size_t i);
  /// Removes slot ``i`` from the least-recently-freed tracking.
  void UntrackFreeSlot(size_t i);
  /// Evicts least-recently-freed buffers until the retention cap is satisfied.
  size_t EnforceRetentionCap() noexcept;

  std::vector<RawBuffer> buffers_;
  std::vector<size_t> unused_slots_;
  std::map<size_t, std::vector<size_t>> free_buckets_;
  std::unordered_map<RawBuffer *, size_t> slot_indices_;
  std::vector<bool> live_slots_;
  /// Free slots in least-recently-freed order; front is the eviction candidate.
  std::list<size_t> lru_order_;
  std::unordered_map<size_t, std::list<size_t>::iterator> lru_iter_;
  size_t retention_cap_;
  size_t total_allocated_size_ = 0;
  size_t peak_allocated_size_ = 0;
  size_t allocated_count_ = 0;
  size_t retained_size_ = 0;
  size_t retained_count_ = 0;
};

class IOArena;

/**
 * Reference-counted lease that pins one I/O allocation for an external owner.
 *
 * A lease is the ownership token handed to a cross-boundary consumer such as a
 * NumPy capsule. While the lease is alive its buffer stays live and pinned: the
 * owning :cpp:class:`IOArena` never reuses that storage. The lease also holds a
 * shared reference to the arena, so the arena outlives every exported buffer and
 * a capsule is never left with a dangling arena pointer.
 *
 * The lease is move-only. Destruction or :cpp:func:`Reset` returns the buffer to
 * its arena's retained free list exactly once; an empty or moved-from lease is a
 * no-op.
 */
class IOLease {
public:
  IOLease() noexcept = default;
  ~IOLease();

  IOLease(const IOLease &) = delete;
  IOLease &operator=(const IOLease &) = delete;
  IOLease(IOLease &&other) noexcept;
  IOLease &operator=(IOLease &&other) noexcept;

  /// Returns whether this lease pins an allocation.
  explicit operator bool() const noexcept { return buffer_ != nullptr; }

  /// Returns the leased buffer, or ``nullptr`` for an empty lease.
  RawBuffer *buffer() const noexcept { return buffer_; }

  /// Returns the logical byte size captured when the lease was created.
  size_t logical_size() const noexcept { return logical_size_; }

  /// Returns the buffer to its arena and makes this lease empty.
  void Reset() noexcept;

private:
  friend class IOArena;
  IOLease(std::shared_ptr<IOArena> arena, RawBuffer *buffer, size_t logical_size) noexcept;

  std::shared_ptr<IOArena> arena_;
  RawBuffer *buffer_ = nullptr;
  size_t logical_size_ = 0;
};

/**
 * Fixed-slot arena for I/O buffers with capacity-preserving reuse and leasing.
 *
 * :cpp:class:`IOArena` retains and reuses freed storage exactly like
 * :cpp:class:`ExecutionArena`, but it additionally supports exporting a live
 * allocation to an external owner through an :cpp:class:`IOLease`. An exported
 * buffer stays live and pinned until its lease is released; only then does the
 * arena reclaim its storage for reuse. Leases keep the arena alive, so an
 * exported buffer never outlives the arena that owns its storage.
 *
 * A retention cap bounds the total capacity kept on the free lists exactly like
 * :cpp:class:`ExecutionArena`; live and leased buffers are never evicted. The
 * cap defaults to unbounded.
 *
 * Because leases share ownership of the arena, an ``IOArena`` must be created on
 * the heap through :cpp:func:`Create` and held by ``std::shared_ptr``.
 *
 * @note This class is not thread-safe.
 */
class IOArena : public RawBufferAllocator, public std::enable_shared_from_this<IOArena> {
public:
  /**
   * Creates a shared arena with at most ``capacity`` simultaneously live buffers.
   *
   * @param capacity       Number of stable :cpp:struct:`RawBuffer` slots.
   * @param retention_cap  Maximum total capacity kept on the retained free
   *                       lists. Defaults to unbounded, disabling eviction.
   */
  static std::shared_ptr<IOArena> Create(size_t capacity,
                                         size_t retention_cap = std::numeric_limits<size_t>::max());

  /**
   * Allocates ``n_bytes`` from the smallest sufficient retained buffer.
   *
   * An unused slot is preferred when no retained buffer is large enough. Once
   * all slots have acquired storage, the largest undersized free buffer grows to
   * satisfy the request.
   *
   * @throws std::bad_alloc if all slots are currently live.
   */
  RawBuffer *Allocate(size_t n_bytes) override;

  /**
   * Returns a live, unleased buffer to its capacity bucket without releasing its
   * storage.
   *
   * @throws std::invalid_argument if ``buf`` is not live in this arena or has
   *         been exported through a lease.
   */
  void Free(RawBuffer *buf) override;

  /**
   * Exports a live buffer to an external owner, returning a pinning lease.
   *
   * The buffer stays live and counted while the lease exists; the arena does not
   * reuse it until the lease is released.
   *
   * @throws std::invalid_argument if ``buf`` is not live in this arena or has
   *         already been exported.
   */
  IOLease Export(RawBuffer *buf);

  /**
   * Exports a live buffer as a self-owning :cpp:class:`AllocationHandle`.
   *
   * The returned handle owns the buffer's :cpp:class:`IOLease`, so it keeps this
   * arena alive on its own and returns the buffer exactly once when destroyed.
   * It may therefore outlive the :cpp:class:`RuntimeContext` that produced it,
   * which is what lets an exported graph output be transferred into a NumPy
   * capsule without keeping the mutable context alive as the data owner.
   *
   * @throws std::invalid_argument if ``buf`` is not live in this arena or has
   *         already been exported.
   */
  AllocationHandle ExportHandle(RawBuffer *buf);

  /**
   * Exports a live buffer already owned by an :cpp:class:`AllocationHandle`.
   *
   * ``handle`` must own a plain allocator-backed buffer from this arena. Its
   * ownership is transferred into the returned self-owning handle without moving
   * or copying the payload; ``handle`` is left empty. This is the ergonomic
   * entry point for exporting a graph output that a :cpp:class:`Tensor` released
   * (see :cpp:func:`Tensor::ReleaseAllocation`) into a NumPy capsule.
   *
   * @throws std::invalid_argument if ``handle`` is empty or already lease-backed,
   *         or if its buffer is not live in this arena or has already been
   *         exported.
   */
  AllocationHandle ExportHandle(AllocationHandle &&handle);

  /// Returns the logical bytes held by live buffers (allocated and leased).
  size_t TotalAllocatedSize() const override;

  /// Returns the peak logical live-byte count.
  size_t PeakAllocatedSize() const override;

  /// Resets the live-byte peak to the current live-byte count.
  void ResetPeak() override;

  /// Returns the maximum number of simultaneously live buffers.
  size_t capacity() const noexcept;

  /// Returns the number of live buffers owned directly by the arena (not leased).
  size_t allocated_count() const noexcept;

  /// Returns the number of buffers currently exported through a lease.
  size_t leased_count() const noexcept;

  /// Returns the total capacity of free retained buffers.
  size_t RetainedSize() const noexcept;

  /// Returns the number of free retained buffers.
  size_t retained_count() const noexcept;

  /// Returns the maximum total capacity kept on the retained free lists.
  size_t retention_cap() const noexcept;

  /**
   * Sets the retention cap and evicts least-recently-freed buffers if needed.
   *
   * Lowering the cap below the current :cpp:func:`RetainedSize` immediately
   * releases the storage of the least-recently-freed buffers until the retained
   * capacity fits. Live and leased buffers are never evicted.
   */
  void SetRetentionCap(size_t retention_cap) noexcept;

  /**
   * Releases the storage retained by every free buffer.
   *
   * Live and leased buffers are left untouched: only buffers already returned to
   * the retained free lists give back their storage. Each trimmed slot returns
   * to the unused pool and acquires fresh storage on a later allocation.
   *
   * @returns The number of retained bytes released.
   */
  size_t Trim() noexcept;

private:
  friend class IOLease;

  explicit IOArena(size_t capacity, size_t retention_cap);

  /// Returns a previously leased buffer to the retained free list — called by
  /// :cpp:class:`IOLease`.
  void ReturnLease(RawBuffer *buf) noexcept;

  /// Records slot ``i`` as the most-recently-freed retained buffer.
  void TrackFreeSlot(size_t i);
  /// Removes slot ``i`` from the least-recently-freed tracking.
  void UntrackFreeSlot(size_t i);
  /// Evicts least-recently-freed buffers until the retention cap is satisfied.
  size_t EnforceRetentionCap() noexcept;

  std::vector<RawBuffer> buffers_;
  std::vector<size_t> unused_slots_;
  std::map<size_t, std::vector<size_t>> free_buckets_;
  std::unordered_map<RawBuffer *, size_t> slot_indices_;
  std::vector<bool> live_slots_;
  std::vector<bool> leased_slots_;
  /// Free slots in least-recently-freed order; front is the eviction candidate.
  std::list<size_t> lru_order_;
  std::unordered_map<size_t, std::list<size_t>::iterator> lru_iter_;
  size_t retention_cap_;
  size_t total_allocated_size_ = 0;
  size_t peak_allocated_size_ = 0;
  size_t allocated_count_ = 0;
  size_t leased_count_ = 0;
  size_t retained_size_ = 0;
  size_t retained_count_ = 0;
};

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime
