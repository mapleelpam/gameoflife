
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ACL specific includes
#include "CL/opencl.h"

#define EPSILON (1e-4f)

static const size_t vectorSize = 32768; //must be evenly disible by workSize
static const size_t workSize = 128;

// ACL runtime configuration
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_kernel kernel;
static cl_program program;
static cl_int status;

static cl_mem kernelX, kernelY, kernelZ;

// input and output vectors
static void *X, *Y, *Z;


// Need to align host data to 32 bytes to be able to use DMA
// LINUX/WINDOWS macros are defined in Makefiles.
#define ACL_ALIGNMENT 32

#ifdef LINUX
#include <stdlib.h>
void* acl_aligned_malloc (size_t size) {
  void *result = NULL;
  posix_memalign (&result, ACL_ALIGNMENT, size);
  return result;
}
void acl_aligned_free (void *ptr) {
  free (ptr);
}
#else // WINDOWS
void* acl_aligned_malloc (size_t size) {
  return _aligned_malloc (size, ACL_ALIGNMENT);
}
void acl_aligned_free (void *ptr) {
  _aligned_free (ptr);
}
#endif // LINUX


static void initializeVector(float* vector, int size) {
  for (int i = 0; i < size; ++i) {
    vector[i] = rand() / (float)RAND_MAX;
  }
}

static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d\n", status);
}

// free the resources allocated during initialization
static void freeResources() {
  if(kernel) 
    clReleaseKernel(kernel);  
  if(program) 
    clReleaseProgram(program);
  if(queue) 
    clReleaseCommandQueue(queue);
  if(context) 
    clReleaseContext(context);
  if(kernelX) 
    clReleaseMemObject(kernelX);
  if(kernelY) 
    clReleaseMemObject(kernelY);
  if(kernelZ) 
    clReleaseMemObject(kernelZ);
  if(X) 
    acl_aligned_free(X);
  if(Y) 
    acl_aligned_free(Y);
  if(Z) 
    acl_aligned_free(Z);
}



int main() {

  cl_uint num_platforms;
  cl_uint num_devices;

  // allocate and initialize the input vectors
  X = (void *)acl_aligned_malloc(sizeof(cl_float) * vectorSize);
  Y = (void *)acl_aligned_malloc(sizeof(cl_float) * vectorSize);
  Z = (void *)acl_aligned_malloc(sizeof(cl_float) * vectorSize);

  initializeVector((float*)X, vectorSize);
  initializeVector((float*)Y, vectorSize);

  // get the platform ID
  status = clGetPlatformIDs(1, &platform, &num_platforms);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetPlatformIDs.", status);
    freeResources();
    return 1;
  }
  if(num_platforms != 1) {
    printf("Found %d platforms!\n", num_platforms);
    freeResources();
    return 1;
  }

  // get the device ID
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
  if(status != CL_SUCCESS) {
    dump_error("Failed clGetDeviceIDs.", status);
    freeResources();
    return 1;
  }
  if(num_devices != 1) {
    printf("Found %d devices!\n", num_devices);
    freeResources();
    return 1;
  }

  // create a context
  context = clCreateContext(0, 1, &device, NULL, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateContext.", status);
    freeResources();
    return 1;
  }

  // create a command queue
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateCommandQueue.", status);
    freeResources();
    return 1;
  }

  // create the input buffer
  kernelX = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * vectorSize, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  // create the input buffer
  kernelY = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * vectorSize, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  // create the output buffer
  kernelZ = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * vectorSize, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }

  // create the kernel
  const char *kernel_name = "vectorAdd";
  cl_int kernel_status;

  // create the program using binary already compiled offline using aoc (i.e. the .aocx file)
  FILE* fp = fopen("life.aocx", "rb");
  if (fp == NULL) {
    printf("Failed to open vectorAdd.aocx file (fopen).\n");
	return -1;
  }
  fseek(fp, 0, SEEK_END);
  size_t binary_length = ftell(fp);
  unsigned char*binary = (unsigned char*) malloc(sizeof(unsigned char) * binary_length);
  assert(binary && "Malloc failed");
  rewind(fp);
  if (fread((void*)binary, binary_length, 1, fp) == 0) {
    printf("Failed to read from moving_average.aocx file (fread).\n");
	return -1;
  }
  fclose(fp);
  program = clCreateProgramWithBinary(context, 1, &device, &binary_length, (const unsigned char**)&binary, &kernel_status, &status);
  if(status != CL_SUCCESS || kernel_status != CL_SUCCESS) {
    dump_error("Failed clCreateProgramWithBinary.", status);
    freeResources();
    return 1;
  }

  // build the program
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  if(status != CL_SUCCESS) {
    dump_error("Failed clBuildProgram.", status);
    freeResources();
    return 1;
  }

  // create the kernel
  kernel = clCreateKernel(program, kernel_name, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateKernel.", status);
    freeResources();
    return 1;
  }

  // set the arguments
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&kernelX);
  if(status != CL_SUCCESS) {
    dump_error("Failed set arg 0.", status);
    return 1;
  }
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&kernelY);
  if(status != CL_SUCCESS) {
    dump_error("Failed Set arg 1.", status);
      freeResources();
    return 1;
  }
  status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&kernelZ);
  if(status != CL_SUCCESS) {
    dump_error("Failed Set arg 2.", status);
    freeResources();
    return 1;
  }

  status = clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&vectorSize);
  if(status != CL_SUCCESS) {
    dump_error("Failed Set arg 3.", status);
    freeResources();
    return 1;
  }

  printf("Kernel initialization is complete.\n");

  status = clEnqueueWriteBuffer(queue, kernelX, CL_FALSE, 0, sizeof(cl_float) * vectorSize, X, 0, NULL, NULL);
  if(status != CL_SUCCESS) {
    dump_error("Failed to enqueue buffer kernelX.", status);
    freeResources();
    return 1;
  }

  status = clEnqueueWriteBuffer(queue, kernelY, CL_FALSE, 0, sizeof(cl_float) * vectorSize, Y, 0, NULL, NULL);
  if(status != CL_SUCCESS) {
    dump_error("Failed to enqueue buffer kernelY.", status);
    freeResources();
    return 1;
  }

  printf("Launching the kernel...\n");

  // launch kernel
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &vectorSize, &workSize, 0, NULL, NULL);
  if (status != CL_SUCCESS) {
    dump_error("Failed to launch kernel.", status);
    freeResources();
    return 1;
  }

  printf("Kernel execution is complete.\n");

  // read the output
  status = clEnqueueReadBuffer(queue, kernelZ, CL_TRUE, 0, sizeof(cl_float) * vectorSize, Z, 0, NULL, NULL);
  if(status != CL_SUCCESS) {
    dump_error("Failed to enqueue buffer kernelZ.", status);
    freeResources();
    return 1;
  }

  // verify the output
  for(int i = 0; i < vectorSize; i++) {
    if(fabs(((float*)X)[i] + ((float*)Y)[i] - ((float*)Z)[i]) > EPSILON) {
      printf("Verification failed X[%d](%f) + Y[%d](%f) != Z[%d](%f)",
	     i, ((float*)X)[i], i, ((float*)Y)[i], i, ((float*)Z)[i]);
      return 1;
    }
  }
  
  printf("Verification succeeded.\n");

  // free the resources allocated
  freeResources();

  return 0;
}

