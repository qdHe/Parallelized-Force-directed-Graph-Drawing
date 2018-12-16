#ifndef __MIC_H__
#define __MIC_H__

#ifdef RUN_MIC
#include <offload.h>
#endif

// Helper for memory management on phi.
/* Allocating memory in Xeon Phi */
#define ALLOC alloc_if(1) free_if(0)
/* After we use the memory, free the space */
#define FREE alloc_if(0) free_if(1)
/* Reuse the memory (must be allocated previously)*/
#define REUSE alloc_if(0) free_if(0)
/* Allocate memory in the beginning and free in the end */
#define INOUT alloc_if(1) free_if(1)
#endif /* __MIC_H__ */
