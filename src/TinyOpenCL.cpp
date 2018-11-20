//============================================================================
// Name        : TinyOpenCL.cpp
// Author      : Darius Malysiak
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <sys/types.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <CL/cl.h>

/*
 * Converts the contents of a file into a string
 */
char* readFileToCString(std::string filename)
{
	size_t size;
	char*  str;

	std::fstream f(filename.c_str(), (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);

		str = new char[size+1];
		if(!str)
		{
			f.close();
			return NULL;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';

		return str;
	}
	else
	{
		printf("Error: file not found\n");
	}

	return NULL;
}

int createKernelFromSourceFile(std::string filename_, std::string kernel_name_, cl_device_id device, cl_context context, cl_kernel* kernel, std::string cl_options)
{
	char* filename = (char*)filename_.c_str();
	char* kernel_name = (char*)kernel_name_.c_str();

	cl_int error;

	//load the cl source file and create a cl program
	const char*  cl_source = readFileToCString(filename);
	const size_t source_length = strlen(cl_source);

	cl_program cl_program = clCreateProgramWithSource(context,
													  1,
													  &cl_source,
													  &source_length,
													  &error);
	if(error != CL_SUCCESS)
	{
	  printf("Error: Creating cl program object (clCreateProgramWithSource)\n");
	  return 1;
	}

	/* create a cl program executable for all the devices specified */
	error = clBuildProgram(cl_program, 1, &device,
			cl_options.c_str(), NULL, NULL);

	//save the build log
	char* build_log;
	size_t log_size;
	// First call to know the proper size
	clGetProgramBuildInfo(cl_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	build_log = new char[log_size+1];
	// Second call to get the log
	clGetProgramBuildInfo(cl_program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
	build_log[log_size] = '\0';

	std::string m_build_log;
	m_build_log = m_build_log + std::string("~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CL compiler messages:\n") + std::string(build_log);

	delete[] build_log;

	//get build info in case of error
	if(error != CL_SUCCESS)
	{
		printf("Error: Building cl program from object (clBuildProgram) code:%d\n\n",error);

		printf("%s",m_build_log.c_str());

		delete[] cl_source;
		return -1;
	}


	/* get a kernel object handle for a kernel with the given name */
	*kernel = clCreateKernel(cl_program, kernel_name, &error);
	if(error != CL_SUCCESS)
	{
		printf("Error: Creating Kernel from program. (clCreateKernel)\n");
		delete[] cl_source;
		return 1;
	}

	delete[] cl_source;

	return 0;
}

int main()
{
	unsigned int element_count = 10;

	cl_int err;
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context ctx = 0;
	cl_command_queue queue = 0;
	cl_mem bufA;
	unsigned int* A = new unsigned int[element_count];

	/* Setup OpenCL environment. */
	unsigned int platcount = 0;
	err = clGetPlatformIDs(0, NULL, &platcount);
	if (err != CL_SUCCESS) {
		printf( "clGetPlatformIDs() failed with %d\n", err );
		return 1;
	}
	printf("found %u platforms\n",platcount);

	err = clGetPlatformIDs(1, &platform, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetPlatformIDs() failed with %d\n", err );
		return 1;
	}

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if (err != CL_SUCCESS) {
		printf( "clGetDeviceIDs() failed with %d\n", err );
		return 1;
	}

	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		printf( "clCreateContext() failed with %d\n", err );
		return 1;
	}

	queue = clCreateCommandQueue(ctx, device, 0, &err);
	if (err != CL_SUCCESS) {
		printf( "clCreateCommandQueue() failed with %d\n", err );
		clReleaseContext(ctx);
		return 1;
	}

	/* Prepare OpenCL memory objects and place matrices inside them. */
	bufA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, element_count * sizeof(unsigned int),
						  NULL, &err);

	if(err != CL_SUCCESS)
	{
		printf("ERROR: clCreateBuffer failed (error: %d)\n",err);
	}

	//transfer data to device
	err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0,
			element_count * sizeof(unsigned int), A, 0, NULL, NULL);

	if(err != CL_SUCCESS)
	{
		printf("ERROR: clEnqueueWriteBuffer failed (error: %d)\n",err);
	}

	//compile the kernel
	cl_kernel kernel;
	createKernelFromSourceFile("../src/tiny.cl", "tiny", device, ctx, &kernel, "");

	//set kernel params
	err =  clSetKernelArg(kernel,0,sizeof(cl_mem), &bufA);

	if(err != CL_SUCCESS)
	{
		printf("ERROR: could not set argument of kernel (error: %d)\n",err);
	}

	//launch the kernel
	size_t global_item_size = element_count;
	size_t local_item_size = 1;
	err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf( "clEnqueueNDRangeKernel() failed with %d\n", err );
		clReleaseContext(ctx);
		return 1;
	}

	/* Wait for calculations to be finished. */
	//err = clWaitForEvents(1, &event);

	/* Fetch results of calculations from GPU memory. */
	err = clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0,
							  element_count * sizeof(unsigned int),
							  A, 0, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf( "clEnqueueReadBuffer() failed with %d\n", err );
		clReleaseContext(ctx);
		return 1;
	}

	//print the result (should be 0,1,2,...,element_count-1)
	for(unsigned int i=0;i<element_count;++i)
	{
		printf("%u \n",A[i]);
	}

	/* Release OpenCL memory objects. */
	clReleaseMemObject(bufA);

	/* Release OpenCL working objects. */
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);

	delete[] A;

	return 0;
}
