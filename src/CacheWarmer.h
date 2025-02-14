#ifndef LLVM_TDG_CACHE_WARMER_H
#define LLVM_TDG_CACHE_WARMER_H

#include "DynamicInstruction.h"

#include "TDGInstruction.pb.h"

#include "llvm/IR/DataLayout.h"

#include <string>
#include <unordered_map>

constexpr size_t DEFAULT_CACHE_LINE_SIZE = 64;
constexpr size_t DEFAULT_CACHE_SIZE = 16 * (1 << 20);

class CacheWarmer {
public:
  CacheWarmer(const std::string &_ExtraFolder, const std::string &_FileName,
              const size_t _CacheLineSize = DEFAULT_CACHE_LINE_SIZE,
              const size_t _CacheSize = DEFAULT_CACHE_SIZE);

  CacheWarmer(const CacheWarmer &Other) = delete;
  CacheWarmer &operator=(const CacheWarmer &Other) = delete;

  CacheWarmer(CacheWarmer &&Other) = delete;
  CacheWarmer &operator=(CacheWarmer &&Other) = delete;

  void addAccess(DynamicInstruction *DynInst, llvm::DataLayout *DataLayout);
  void dumpToFile() const;

private:
  const std::string FileName;
  const std::string SnapshotFileName;
  const size_t CacheLineSize;
  const size_t CacheSize;

  /**
   * Stores basically the memory request history.
   */
  LLVM::TDG::CacheWarmUp CacheWarmUpProto;

  std::unordered_map<uint64_t, uint8_t> InitialMemorySnapshot;

  // void addAccess(uint64_t Addr);

  LLVM::TDG::MemorySnapshot generateSnapshot() const;
};

#endif