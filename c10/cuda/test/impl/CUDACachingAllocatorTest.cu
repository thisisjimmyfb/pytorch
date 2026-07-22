#include <gtest/gtest.h>

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/PeerToPeerAccess.h>
#include <c10/util/Exception.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// This bare c10_cuda test binary doesn't link ATen/CUDAHooks.cpp (part of
// the much larger libtorch_cuda target), so c10::cuda::hasPrimaryContext is
// otherwise left as an unregistered dummy that unconditionally throws (see
// c10/cuda/CUDAFunctions.cpp). Other c10_cuda tests don't hit this because
// they never touch CUDAGuard-based code paths; getStreamFromPool() and
// CUDACachingAllocator::allocate() (via getCurrentCUDAStream) do. Register
// a minimal real implementation here, mirroring what CUDAHooks.cpp does for
// the full libtorch build.
namespace c10::cuda::_internal {
void setHasPrimaryContext(bool (*func)(DeviceIndex));
}

namespace {

bool always_has_primary_context(c10::DeviceIndex /*device_index*/) {
  return true;
}

struct PrimaryContextRegistrar {
  PrimaryContextRegistrar() {
    c10::cuda::_internal::setHasPrimaryContext(always_has_primary_context);
  }
  ~PrimaryContextRegistrar() {
    c10::cuda::_internal::setHasPrimaryContext(nullptr);
  }
} g_primary_context_registrar;

} // namespace

namespace {

// Repeatedly reads and writes a large scratch buffer, to generate sustained
// memory-bus traffic on the GPU (rather than pure ALU/clock-based spinning)
// while the host issues expandable-segment map/unmap calls on the same
// stream without synchronizing. Real workloads that trigger allocator
// growth (e.g. weight loading) are doing memory traffic, not idle spinning,
// so this is a closer analog for whatever contention the driver race
// depends on.
__global__ void memory_churn_kernel(float* data, size_t n, int iters) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (int it = 0; it < iters; ++it) {
    for (size_t i = idx; i < n; i += stride) {
      data[i] = data[i] * 1.0000001f + 1.0f;
    }
  }
}

constexpr int kChurnGridDim = 256;
constexpr int kChurnBlockDim = 256;

void launch_memory_churn(
    float* data,
    size_t n,
    int iters,
    cudaStream_t stream) {
  memory_churn_kernel<<<kChurnGridDim, kChurnBlockDim, 0, stream>>>(
      data, n, iters);
}

} // namespace

// Allocates from the caching allocator's expandable-segment pool on two
// separate threads/streams at once for a short while, forcing concurrent
// mapAndSetAccess/cuMemMap_/cuMemSetAccess_ growth. Not expected to fail on
// bare-metal Linux -- meant to be run on the affected WSL2 hardware.
TEST(CUDACachingAllocatorTest, ConcurrentSegmentExpand) {
  if (c10::cuda::device_count() == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  // Normally done by CUDAHooks.cpp, which this bare test binary doesn't link.
  c10::cuda::detail::init_p2p_access_cache(c10::cuda::device_count());
  c10::cuda::CUDACachingAllocator::init(1);
  c10::cuda::CUDACachingAllocator::setAllocatorSettings(
      "expandable_segments:True,max_split_size_mb:20");
  c10::cuda::CUDACachingAllocator::emptyCache();
  auto* allocator = c10::cuda::CUDACachingAllocator::get();

  constexpr int kNumRounds = 3;
  constexpr int kAllocsPerRound = 60;

  // Base sizes are tuned to push ~85-95% memory utilization by the last
  // round relative to kReferenceFreeMB. Scale them by currently-free device
  // memory so the test drives the allocator to the same relative memory
  // pressure regardless of how much VRAM is actually available.
  constexpr size_t kReferenceFreeMB = 16ULL * 1024;
  constexpr size_t kBaseSizesMB[] = {8, 16, 32, 64, 128};
  constexpr int kNumSizes = sizeof(kBaseSizesMB) / sizeof(kBaseSizesMB[0]);
  size_t device_free = 0;
  size_t device_total = 0;
  C10_CUDA_CHECK(cudaMemGetInfo(&device_free, &device_total));
  size_t device_free_mb = device_free / (1024 * 1024);
  size_t kSizesMB[kNumSizes];
  for (int i = 0; i < kNumSizes; ++i) {
    kSizesMB[i] = std::max<size_t>(
        1, kBaseSizesMB[i] * device_free_mb / kReferenceFreeMB);
  }

  // Multiple background threads (rather than just one) increase contention
  // on cuMemSetAccess_/cuMemMap_ during concurrent segment expansion, which
  // raises the reproduction rate of the driver-level races this test targets.
  constexpr int kNumBgThreads = 9;

  std::atomic<bool> stop_bg{false};
  std::atomic<bool> bg_failed{false};
  std::mutex bg_err_mutex;
  std::string bg_first_error;
  std::vector<std::thread> bg_threads;
  bg_threads.reserve(kNumBgThreads);
  for (int t = 0; t < kNumBgThreads; ++t) {
    bg_threads.emplace_back([&]() {
      try {
        c10::cuda::CUDAStream bg_stream = c10::cuda::getStreamFromPool();
        c10::cuda::CUDAStreamGuard bg_guard(bg_stream);
        std::deque<c10::DataPtr> ring;
        int i = 0;
        while (!stop_bg) {
          size_t bytes = kSizesMB[i % kNumSizes] * 1024ULL * 1024ULL;
          ring.push_back(allocator->allocate(bytes));
          if (ring.size() > 25) {
            ring.pop_front();
          }
          ++i;
        }
        C10_CUDA_CHECK(cudaStreamSynchronize(bg_stream.stream()));
      } catch (const c10::OutOfMemoryError&) {
      } catch (const std::exception& e) {
        bg_failed = true;
        std::lock_guard<std::mutex> lock(bg_err_mutex);
        if (bg_first_error.empty()) {
          bg_first_error = e.what();
        }
      }
    });
  }

  bool fg_failed = false;
  std::string fg_first_error;
  c10::cuda::CUDAStream fg_stream = c10::cuda::getStreamFromPool();
  c10::cuda::CUDAStreamGuard fg_guard(fg_stream);
  try {
    for (int round = 0; round < kNumRounds; ++round) {
      std::vector<c10::DataPtr> allocations;
      allocations.reserve(kAllocsPerRound);
      for (int i = 0; i < kAllocsPerRound; ++i) {
        size_t bytes = kSizesMB[i % kNumSizes] * 1024ULL * 1024ULL;
        allocations.push_back(allocator->allocate(bytes));
      }
      launch_memory_churn(
          static_cast<float*>(allocations.front().get()),
          kSizesMB[0] * 1024ULL * 1024ULL / sizeof(float),
          /*iters=*/5,
          fg_stream.stream());
      C10_CUDA_CHECK(cudaStreamSynchronize(fg_stream.stream()));
      allocations.clear();
    }
  } catch (const c10::OutOfMemoryError&) {
  } catch (const std::exception& e) {
    fg_failed = true;
    fg_first_error = e.what();
  }

  stop_bg = true;
  for (auto& t : bg_threads) {
    t.join();
  }

  if (fg_failed) {
    FAIL() << "Foreground allocation thread failed: " << fg_first_error;
  }
  if (bg_failed) {
    FAIL() << "Background free/realloc thread failed: " << bg_first_error;
  }
}
