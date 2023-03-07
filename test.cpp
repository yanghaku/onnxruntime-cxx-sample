#include <algorithm>
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

#define OUTPUT_LEN 1000
#define OUTPUT_SHAPE                                                                               \
    { 1, OUTPUT_LEN }
#define OUTPUT_SHAPE_LEN 2

// read the binary input from file
std::array<float, INPUT_LEN> read_input(const char *input_path) {
    auto input_buf = std::array<float, INPUT_LEN>();
    std::ifstream f(input_path, std::ios::in | std::ios::binary);
    f.read(reinterpret_cast<char *>(input_buf.begin()), INPUT_LEN * sizeof(float));
    return input_buf;
}

void post_process(const std::array<float, OUTPUT_LEN> &output) {
    // no softmax, just find max
    float max_value = std::numeric_limits<float>::min();
    int max_index = -1;
    for (int i = 0; i < 1000; ++i) {
        if (output[i] > max_value) {
            max_value = output[i];
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

    // init api and create Session
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

    auto env = Ort::Env();
    // env.UpdateEnvWithCustomLogLevel(ORT_LOGGING_LEVEL_VERBOSE);
    auto session = Ort::Session(env, model_path, session_options);
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
        std::cerr << "Unsupported Model" << std::endl;
    }

    // prepare input and output
    std::array<float, INPUT_LEN> input_data = read_input(argv[2]);
    std::array<float, OUTPUT_LEN> output_data;
    int64_t input_shape[] = INPUT_SHAPE;
    int64_t output_shape[] = OUTPUT_SHAPE;
    // input and output name
    auto cpu_mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto allocator = Ort::Allocator(session, cpu_mem_info);
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    const char *const input_names[] = {&*input_name};
    const char *const output_names[] = {&*output_name};
    // create tensor for input and output
    auto input = Ort::Value::CreateTensor(cpu_mem_info, input_data.begin(), INPUT_LEN, input_shape,
                                          INPUT_SHAPE_LEN);
    auto output = Ort::Value::CreateTensor(cpu_mem_info, output_data.begin(), OUTPUT_LEN,
                                           output_shape, OUTPUT_SHAPE_LEN);

    // run inference and record time.
    const int test_run_num = 20;
    static_assert(test_run_num > 2, "");
    auto run_time = std::vector<int64_t>();

    for (auto i = 0; i < test_run_num; ++i) {
        auto start = std::chrono::system_clock::now();
        session.Run(Ort::RunOptions(), input_names, &input, 1, output_names, &output, 1);
        auto end = std::chrono::system_clock::now();
        auto e = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        run_time.push_back(e);
    }

    std::cout << "Run inference " << test_run_num << " times." << std::endl;
    std::sort(run_time.begin(), run_time.end());
    std::cout << "Min time = " << run_time[0] / 1000 << "." << run_time[0] % 1000 << " ms"
              << std::endl;
    std::cout << "Max time = " << run_time.back() / 1000 << "." << run_time.back() % 1000 << " ms"
              << std::endl;
    auto sum = 0;
    for (auto t : run_time) {
        sum += t;
    }
    auto avg = double(sum - run_time[0] - run_time.back()) / (run_time.size() - 2);
    std::cout << "Average time = " << avg / 1000.0 << " ms" << std::endl;

    post_process(output_data);
    return 0;
}
