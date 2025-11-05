#ifndef OCTREE_H
#define OCTREE_H

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <stdint.h>

// Bit field layout for 32-bit octree descriptor:
// Bits 31-17: Child pointer (15 bits)
// Bit  16:    Far bit (1 bit)
// Bits 15-8:  Valid mask (8 bits)
// Bits 7-0:   Leaf mask (8 bits)

#define CHILD_POINTER_SHIFT 17
#define CHILD_POINTER_MASK  0x7FFF  // 15 bits
#define FAR_BIT_SHIFT       16
#define FAR_BIT_MASK        0x1     // 1 bit
#define VALID_MASK_SHIFT    8
#define VALID_MASK_MASK     0xFF    // 8 bits
#define LEAF_MASK_SHIFT     0
#define LEAF_MASK_MASK      0xFF    // 8 bits

typedef struct {
    uint32_t d;
} octree_t;

// Extract child pointer (15 bits, bits 31-17)
__host__ __device__ inline uint32_t octree_get_child_pointer(uint32_t descriptor) {
    return (descriptor >> CHILD_POINTER_SHIFT) & CHILD_POINTER_MASK;
}

// Extract far bit (1 bit, bit 16)
__host__ __device__ inline uint32_t octree_get_far_bit(uint32_t descriptor) {
    return (descriptor >> FAR_BIT_SHIFT) & FAR_BIT_MASK;
}

// Check if far bit is set
__host__ __device__ inline bool octree_is_far_pointer(uint32_t descriptor) {
    return ((descriptor >> FAR_BIT_SHIFT) & FAR_BIT_MASK) != 0;
}

// Extract valid mask (8 bits, bits 15-8)
__host__ __device__ inline uint8_t octree_get_valid_mask(uint32_t descriptor) {
    return (descriptor >> VALID_MASK_SHIFT) & VALID_MASK_MASK;
}

// Extract leaf mask (8 bits, bits 7-0)
__host__ __device__ inline uint8_t octree_get_leaf_mask(uint32_t descriptor) {
    return (descriptor >> LEAF_MASK_SHIFT) & LEAF_MASK_MASK;
}

// Check if a specific child is valid (child index 0-7)
__host__ __device__ inline bool octree_is_child_valid(uint32_t descriptor, uint8_t child_index) {
    uint8_t valid_mask = octree_get_valid_mask(descriptor);
    return (valid_mask & (1 << child_index)) != 0;
}

// Check if a specific child is a leaf (child index 0-7)
__host__ __device__ inline bool octree_is_child_leaf(uint32_t descriptor, uint8_t child_index) {
    uint8_t leaf_mask = octree_get_leaf_mask(descriptor);
    return (leaf_mask & (1 << child_index)) != 0;
}

// Create a descriptor from components
__host__ __device__ inline uint32_t octree_create_descriptor(uint32_t child_pointer, 
                                                            bool far_bit, 
                                                            uint8_t valid_mask, 
                                                            uint8_t leaf_mask) {
    return ((child_pointer & CHILD_POINTER_MASK) << CHILD_POINTER_SHIFT) |
           ((far_bit ? 1 : 0) << FAR_BIT_SHIFT) |
           ((valid_mask & VALID_MASK_MASK) << VALID_MASK_SHIFT) |
           ((leaf_mask & LEAF_MASK_MASK) << LEAF_MASK_SHIFT);
}

// Calculate actual child address considering far pointers
__host__ __device__ inline octree_t* octree_get_child_address(octree_t* base_address, 
                                                             octree_t* current_descriptor,
                                                             uint32_t descriptor_value) {
    uint32_t child_pointer = octree_get_child_pointer(descriptor_value);
    
    if (octree_is_far_pointer(descriptor_value)) {
        // Far pointer case: child_pointer is an indirect reference
        // The actual 32-bit far pointer is stored nearby in the array
        octree_t* far_pointer_location = current_descriptor + child_pointer;
        uint32_t far_offset = far_pointer_location->d;
        return base_address + far_offset;
    } else {
        // Direct pointer case: child_pointer is the direct offset
        return base_address + child_pointer;
    }
}

// Count number of valid children
__host__ __device__ inline uint8_t octree_count_valid_children(uint32_t descriptor) {
    uint8_t valid_mask = octree_get_valid_mask(descriptor);
    uint8_t count = 0;
    for (int i = 0; i < 8; i++) {
        if (valid_mask & (1 << i)) count++;
    }
    return count;
}

// Count number of leaf children
__host__ __device__ inline uint8_t octree_count_leaf_children(uint32_t descriptor) {
    uint8_t leaf_mask = octree_get_leaf_mask(descriptor);
    uint8_t count = 0;
    for (int i = 0; i < 8; i++) {
        if (leaf_mask & (1 << i)) count++;
    }
    return count;
}

#endif