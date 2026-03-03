// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_LOCAL_PIPELINE_LAYOUT_H_
#define IREE_HAL_LOCAL_LOCAL_PIPELINE_LAYOUT_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_local_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT 128

typedef struct iree_hal_local_descriptor_set_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_descriptor_set_layout_flags_t flags;
  iree_host_size_t binding_count;
  iree_hal_descriptor_set_layout_binding_t bindings[];
} iree_hal_local_descriptor_set_layout_t;

iree_status_t iree_hal_local_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

iree_hal_local_descriptor_set_layout_t*
iree_hal_local_descriptor_set_layout_cast(
    iree_hal_descriptor_set_layout_t* base_value);

//===----------------------------------------------------------------------===//
// iree_hal_local_pipeline_layout_t
//===----------------------------------------------------------------------===//

#define IREE_HAL_LOCAL_MAX_DESCRIPTOR_SET_COUNT 2
#define IREE_HAL_LOCAL_MAX_PUSH_CONSTANT_COUNT 64

// Number of 64-bit words needed to represent the binding mask.
// To increase max bindings, just increase IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT.
#define IREE_HAL_LOCAL_BINDING_MASK_WORDS \
  ((IREE_HAL_LOCAL_MAX_DESCRIPTOR_BINDING_COUNT + 63) / 64)

typedef struct {
  uint64_t words[IREE_HAL_LOCAL_BINDING_MASK_WORDS];
} iree_hal_local_binding_mask_t;

#define IREE_HAL_LOCAL_BINDING_MASK_BITS \
  (IREE_HAL_LOCAL_BINDING_MASK_WORDS * 64)

// Initialize a binding mask to zero.
static inline void iree_hal_local_binding_mask_clear(
    iree_hal_local_binding_mask_t* mask) {
  for (int i = 0; i < IREE_HAL_LOCAL_BINDING_MASK_WORDS; ++i) {
    mask->words[i] = 0;
  }
}

// Set a bit in the binding mask.
static inline void iree_hal_local_binding_mask_set(
    iree_hal_local_binding_mask_t* mask, int bit) {
  int word_idx = bit / 64;
  int bit_idx = bit % 64;
  mask->words[word_idx] |= (1ull << bit_idx);
}

// Test if a bit is set in the binding mask.
static inline bool iree_hal_local_binding_mask_test(
    const iree_hal_local_binding_mask_t* mask, int bit) {
  int word_idx = bit / 64;
  int bit_idx = bit % 64;
  return (mask->words[word_idx] & (1ull << bit_idx)) != 0;
}

// Count the number of set bits in the binding mask.
static inline int iree_hal_local_binding_mask_count_ones(
    const iree_hal_local_binding_mask_t* mask) {
  int count = 0;
  for (int i = 0; i < IREE_HAL_LOCAL_BINDING_MASK_WORDS; ++i) {
    count += __builtin_popcountll(mask->words[i]);
  }
  return count;
}

// Find the index of the next set bit starting from |start| (inclusive).
// Returns -1 if no more bits are set.
static inline int iree_hal_local_binding_mask_next_set_bit(
    const iree_hal_local_binding_mask_t* mask, int start) {
  int word_idx = start / 64;
  int bit_idx = start % 64;
  for (int i = word_idx; i < IREE_HAL_LOCAL_BINDING_MASK_WORDS; ++i) {
    uint64_t word = mask->words[i];
    // Mask off bits before our starting position in the first word.
    if (i == word_idx) {
      word &= ~((1ull << bit_idx) - 1);
    }
    if (word) {
      return i * 64 + __builtin_ctzll(word);
    }
  }
  return -1;
}

typedef struct iree_hal_local_pipeline_layout_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t push_constants;
  iree_hal_local_binding_mask_t used_bindings;
  iree_hal_local_binding_mask_t read_only_bindings;
  iree_host_size_t set_layout_count;
  iree_hal_descriptor_set_layout_t* set_layouts[];
} iree_hal_local_pipeline_layout_t;

iree_status_t iree_hal_local_pipeline_layout_create(
    iree_host_size_t push_constants, iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

iree_hal_local_pipeline_layout_t* iree_hal_local_pipeline_layout_cast(
    iree_hal_pipeline_layout_t* base_value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_LOCAL_PIPELINE_LAYOUT_H_
