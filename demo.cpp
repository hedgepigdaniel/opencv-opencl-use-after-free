#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <thread>
#include <vector>
#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>


using namespace cv;
using namespace std;

void create_opencl_context(
    cl_platform_id *platform_id,
    cl_device_id *device_id,
    cl_context *context)
{
    cl_int cl_error;
    cl_uint num_platforms;
    cl_uint num_devices;

    cl_error = clGetPlatformIDs(1, platform_id, &num_platforms);
    if (cl_error != CL_SUCCESS || num_platforms <= 0) {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        throw;
    }

    cl_error = clGetDeviceIDs(
        *platform_id,
        CL_DEVICE_TYPE_DEFAULT,
        1,
        device_id,
        &num_devices);
    if (cl_error != CL_SUCCESS || num_devices <= 0) {
        std::cerr << "Failed to find any OpenCL devices." << std::endl;
        throw;
    }

    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) *platform_id,
        0
    };
    *context = clCreateContext(properties, 1, device_id, NULL, NULL, &cl_error);
    if (cl_error != CL_SUCCESS) {
        std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
        throw;
    }
}

vector<char> get_opencl_platform_name(cl_platform_id platform) {
    int err;
    vector<char> platform_name;
    size_t param_value_size = 0;

    err = clGetPlatformInfo(
        platform, CL_PLATFORM_NAME, 0, NULL, &param_value_size);

    if (err != CL_SUCCESS) {
        std::cerr << "clGetPlatformInfo failed to get platform name size\n";
        throw;
    }
    if (param_value_size == 0) {
        std::cerr << "clGetPlatformInfo returned 0 size for name\n";
        throw;
    }
    platform_name.resize(param_value_size);
    err = clGetPlatformInfo(
        platform,
        CL_PLATFORM_NAME,
        param_value_size,
        platform_name.data(),
        NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "clGetPlatformInfo failed\n";
        throw;
    }

    return platform_name;
}

void set_thread_global_opencl_context(
    cl_platform_id platform,
    cl_device_id device,
    cl_context context)
{
    vector<char> platform_name = get_opencl_platform_name(platform);

    std::cerr << "Initialising OpenCV OpenCL context with platform \""
              << string(platform_name.begin(), platform_name.end()) << "\"\n";

    /**
     * I don't expect that this takes ownership of the passed context/device,
     * but it seems that it does!
     */
    ocl::OpenCLExecutionContext opencv_context =
        ocl::OpenCLExecutionContext::create(
            platform_name.data(), platform, context, device);

    opencv_context.bind();
}

void opencv_thread_entry(cl_platform_id platform, cl_device_id device, cl_context context)
{
    set_thread_global_opencl_context(platform, device, context);
}

int main(int argc, char** argv )
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;

    // Create an OpenCL context
    create_opencl_context(&platform, &device, &context);

    // Spawn a thread, and pass the OpenCL context for it to use.
    thread opencv_thread(opencv_thread_entry, platform, device, context);
    opencv_thread.join();

    /**
     * I expect that this context and device is still valid, because we
     * maintained a reference to them in this thread, and did not (willingly)
     * relinquish ownership of them in the other thread.
     * 
     * But instead, there are use-after-free errors when clReleaseContext
     * is called!
     */
    clReleaseContext(context);
    clReleaseDevice(device);
}
