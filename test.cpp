#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <onnxruntime_cxx_api.h>

// resnet, vgg
#define INPUT_LEN (1 * 3 * 224 * 224)
#define INPUT_SHAPE                                                                                \
    { 1, 3, 224, 224 }
#define INPUT_SHAPE_LEN 4

float *read_input(const char *input_path) {
    static float input_buf[INPUT_LEN];

    std::ifstream f(input_path, std::ios::in | std::ios::binary);
    f.read(reinterpret_cast<char *>(input_buf), INPUT_LEN * sizeof(float));
    return input_buf;
}

void post_process(const std::vector<Ort::Value> &output) {
    if (output.size() != 1) {
        std::cerr << "Run Fail! Output size = " << output.size() << std::endl;
    }
    if (!output[0].IsTensor()) {
        std::cerr << "Run Fail Output is not tensor" << std::endl;
    }
    auto out_buf = reinterpret_cast<const float *>(output[0].GetTensorRawData());

    // no softmax, just find max
    float max_value = std::numeric_limits<float>::min();
    int max_index = -1;
    for (int i = 0; i < 1000; ++i) {
        if (out_buf[i] > max_value) {
            max_value = out_buf[i];
            max_index = i;
        }
    }
    std::cout << "Max is " << max_value << " with index " << max_index << std::endl;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage " << argv[0] << " Model_Path Input_Path [CPU|CUDA|TensorRT]"
                  << std::endl;
        return -1;
    }

    const ORTCHAR_T *model_path = argv[1];
    float *input_data = read_input(argv[2]);
    int64_t input_shape[] = INPUT_SHAPE;

    auto const &api = Ort::GetApi();
    auto session_options = Ort::SessionOptions();

    if (strcmp(argv[3], "CPU") == 0) {
        // do nothing
    } else if (strcmp(argv[3], "CUDA") == 0) {
        OrtCUDAProviderOptions o;
        session_options.AppendExecutionProvider_CUDA(o);
    } else if (strcmp(argv[3], "TensorRT") == 0) {
        OrtTensorRTProviderOptions o;
        session_options.AppendExecutionProvider_TensorRT(o);
    } else {
        std::cerr << "Unsupported Option `" << argv[3] << "`" << std::endl;
        return -1;
    }

    auto session = Ort::Session(Ort::Env(), model_path, session_options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
        std::cerr << "Unsupported Model" << std::endl;
    }
    auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto allocator = Ort::Allocator(session, cpu_mem_info);
    auto input =
        Ort::Value::CreateTensor(cpu_mem_info, input_data, INPUT_LEN, input_shape, INPUT_SHAPE_LEN);
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char *const input_names[] = {&*input_name};
    const char *const output_names[] = {&*output_name};

    auto start = std::chrono::system_clock::now();
    auto output = session.Run(Ort::RunOptions(), input_names, &input, 1, output_names, 1);
    auto end = std::chrono::system_clock::now();
    auto e = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time = " << e / 1000 << "." << e % 1000 << " ms" << std::endl;

    post_process(output);
    return 0;
}
