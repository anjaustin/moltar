#ifndef _LINUX_ION_H
#define _LINUX_ION_H

#include <stdint.h>

#define ION_IOC_MAGIC 0x49

struct ion_allocation_data {
    size_t len;
    size_t align;
    unsigned int heap_id_mask;
    unsigned int flags;
    int handle;
};

#define ION_IOC_ALLOC _IOWR(ION_IOC_MAGIC, 0, struct ion_allocation_data)

#endif
