// This version of GoL host uses shared memory between FPGA and CPU for 
// all buffers. Works on CV SoC only.
//
// clCreateBuffer() allocates shared memory.
// clEnqueueMapBuffer() maps allocated cl_mem object to a pointer the host can use.
//
// Note the absense of clEnqueueRead/WriteBuffer calls.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ACL specific includes
#include "CL/opencl.h"

extern "C" 
{
	#include "gfx.h"
}

#define EPSILON (1e-4f)

static const size_t vectorDimSize = 512; //must be evenly disible by workSize
static const size_t vectorSize = vectorDimSize*vectorDimSize; //must be evenly disible by workSize
static const size_t workSize = 32;

// ACL runtime configuration
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_kernel kernel;
static cl_program program;
static cl_int status;

static cl_mem kernelX, kernelY;

// input and output vectors
static void *X, *Y;


void swap( void** X, void** Y)
{
	void* tmp = *X;
	*X = *Y;
	*Y = tmp;
}

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


static void initializeVector(int* vector, int size) {
  for (int i = 0; i < size; ++i) {
    //vector[i] = rand() / (float)RAND_MAX;
    vector[i] = (rand() % 4) ? 0 : 1;
  }
}

static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d\n", status);
}

// free the resources allocated during initialization
static void freeResources() {

  if(kernelX) {
    clEnqueueUnmapMemObject(queue, kernelX, X, 0, NULL, NULL);
    clReleaseMemObject(kernelX);
  }
  if(kernelY) {
    clEnqueueUnmapMemObject(queue, kernelY, Y, 0, NULL, NULL);
    clReleaseMemObject(kernelY);
  }

  if(kernel) 
    clReleaseKernel(kernel);  
  if(program) 
    clReleaseProgram(program);
  if(queue) 
    clReleaseCommandQueue(queue);
  if(context) 
    clReleaseContext(context);
}


void draw( const int *in, int sizeOfDim )
{
	int posx, posy;
	for( posx = 0 ; posx < sizeOfDim ; posx ++ ) {
		for( posy = 0 ; posy < sizeOfDim ; posy ++ ) {
			int index = posy*sizeOfDim+posx;
			if( in[index] ) {
				gfx_color(200,200,100);
			} else {
				gfx_color(0,0,0);
			}	
			gfx_point(posx,posy);
		}
	}
}
void next(const int *in, 
		int *out, 
		int sizeOfDim 
			)
{
    // get index of the work item
	int posx, posy;

	for( posx = 0 ; posx < sizeOfDim ; posx ++ ) {
		for( posy = 0 ; posy < sizeOfDim ; posy ++ ) {
			// *begin* num of neighbor
			int num = 0;	
			if( posy > 0 ) {
				if( in[ (posy-1)*sizeOfDim+posx ])	num ++;
				if( posx > 0 && in[ (posy-1)*sizeOfDim+(posx-1) ])	num ++;
				if( posx+1 < sizeOfDim && in[ (posy-1)*sizeOfDim+(posx+1) ])	num ++;
			}
			if( posy+1 < sizeOfDim ) {
				if( in[ (posy+1)*sizeOfDim+posx ])	num ++;
				if( posx > 0 && in[ (posy+1)*sizeOfDim+(posx-1) ])	num ++;
				if( posx+1 < sizeOfDim && in[ (posy+1)*sizeOfDim+(posx+1) ])	num ++;
			}
			if( posx > 0 && in[ (posy)*sizeOfDim+(posx-1) ])	num ++;
			if( posx+1 < sizeOfDim && in[ (posy)*sizeOfDim+(posx+1) ])	num ++;
			// *end* num of neighbor
			int index = posy*sizeOfDim+posx;
    			out[index] = ( num == 3 ) || ( num == 2 && in[index] ); 
		}
	}

}


int main() {

  cl_uint num_platforms;
  cl_uint num_devices;

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
  kernelX = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int) * vectorSize, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }
  X = clEnqueueMapBuffer(queue, kernelX, CL_TRUE, CL_MAP_WRITE|CL_MAP_READ, 0, sizeof(cl_int) * vectorSize, 0, NULL, NULL, NULL);

  // create the input buffer
  kernelY = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(cl_int) * vectorSize, NULL, &status);
  if(status != CL_SUCCESS) {
    dump_error("Failed clCreateBuffer.", status);
    freeResources();
    return 1;
  }
  Y = clEnqueueMapBuffer(queue, kernelY, CL_TRUE, CL_MAP_WRITE|CL_MAP_READ, 0, sizeof(cl_int) * vectorSize, 0, NULL, NULL, NULL);

  // initialize the input vectors
  initializeVector((int*)X, vectorSize);
 // initializeVector((float*)Y, vectorSize);


  // create the kernel
  const char *kernel_name = "next";
  cl_int kernel_status;

  // create the program using binary already compiled offline using aoc (i.e. the .aocx file)
  FILE* fp = fopen("life.aocx", "rb");
  if (fp == NULL) {
    printf("Failed to open life.aocx file (fopen).\n");
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

  status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&vectorDimSize);
  if(status != CL_SUCCESS) {
    dump_error("Failed Set arg 2.", status);
    freeResources();
    return 1;
  }


  printf("Kernel initialization is complete.\n");

  // No need to do clEnqueueWriteBuffer for kernelX and kernelY!
  
  gfx_open(vectorDimSize,vectorDimSize,"FPGA");
  printf("Launching the kernel...\n");


  for(int idx = 0 ; idx < 1000 ; idx ++ ) {

		  if(idx%2 == 0 ) {
				  draw((int*)X,vectorDimSize);
				  //	next((int*)X,(int*)Y,vectorDimSize);
				  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&kernelX);
				  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&kernelY);
		  } else {
				  draw((int*)Y,vectorDimSize);
				  //	next((int*)Y,(int*)X,vectorDimSize);
				  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&kernelY);
				  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&kernelX);
		  }

		  // launch kernel
		  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &vectorSize, &workSize, 0, NULL, NULL);
		  if (status != CL_SUCCESS) {
				  dump_error("Failed to launch kernel.", status);
				  freeResources();
				  return 1;
		  }

		  printf("Kernel execution is complete. %d\n",idx);
  }
  
  printf("Verification succeeded.\n");

  // free the resources allocated
  freeResources();

  return 0;
}

