#ifndef PROGRAMMABLE_STREAM_PREFETCH_ISA_H
#define PROGRAMMABLE_STREAM_PREFETCH_ISA_H

#include <stdint.h>

/****************************************************************
 * Editor: K16DIABLO (Sungjun Jung)
 * These are used as C API for programmer.
 * Wrapper function of assembly
 * 2023-10-09: Not available yet, Use this as example code (Hardcode stream_id)
 ****************************************************************/

void stream_config (uint32_t stream_id, void* idx_base_addr, uint64_t idx_granularity, void* val_base_addr, uint64_t val_granularity) {
  asm volatile (
      "stream.cfg.idx.base  %[stream_id], %[idx_base_addr] \t\n"    // Configure stream (base address of index)
      "stream.cfg.idx.gran  %[stream_id], %[idx_granularity] \t\n"  // Configure stream (access granularity of index)
      "stream.cfg.val.base  %[stream_id], %[val_base_addr] \t\n"    // Configure stream (base address of value)
      "stream.cfg.val.gran  %[stream_id], %[val_granularity] \t\n"  // Configure stream (access granularity of value)
      :
      :[stream_id]"r"(stream_id), [idx_base_addr]"r"(idx_base_addr), [idx_granularity]"r"(idx_granularity),
      [val_base_addr]"r"(val_base_addr), [val_granularity]"r"(val_granularity)
      );
  return;
}

void stream_input (uint32_t stream_id, int64_t offset_begin, int64_t offset_end) {
  asm volatile (
      "stream.input.offset.begin  %[stream_id], %[offset_begin] \t\n"  // Input stream (offset_begin)
      "stream.input.offset.end    %[stream_id], %[offset_end] \t\n"    // Input stream (offset_end)
      :
      :[stream_id]"r"(stream_id), [offset_begin]"r"(offset_begin), [offset_end]"r"(offset_end)
      );
  return;
}

void stream_terminate (uint32_t stream_id) {
  asm volatile (
      "stream.terminate %[stream_id] \t\n"  // Terminate stream
      :
      :[stream_id]"r"(stream_id)
      );
  return;
}

#endif
