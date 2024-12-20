#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

using std::vector;

class FigureProcessor {
private:
  unsigned char* figure;
  unsigned char* result;
  float LUT[256];
  const size_t size;

public:
  FigureProcessor(size_t size, size_t seed = 0) : size(size) {
    
    // !!! Please do not modify the following code !!!
    // 如果你需要修改内存的数据结构，请不要修改初始化的顺序和逻辑
    // 助教可能会通过指定某个初始化seed 的方式来验证你的代码
    // 如果你修改了初始化的顺序，可能会导致你的代码无法通过测试
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // 两个数组的初始化在这里，可以改动，但请注意 gen 的顺序是从上到下从左到右即可。

    figure=(unsigned char*)malloc(size * size * sizeof(unsigned char));
    result=(unsigned char*)malloc(size * size * sizeof(unsigned char));

    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        figure[i * size + j]=static_cast<unsigned char>(distribution(gen));
      }
    }

    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        result[i * size + j]=0;
      }
    }

    LUT[0] = 0;
    for (size_t i = 1; i < 256; ++i) {
      float normalized = i / 255.0f;
      LUT[i] = static_cast<unsigned char>(
          255.0f * std::pow(normalized, 0.5f) + 0.5f); 
    }
  }

  ~FigureProcessor() {
    free(figure);
    free(result);
  }

  // Gaussian filter
  // [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16
  //FIXME: Feel free to optimize this function
  //Hint: You can use SIMD instructions to optimize this function
  void gaussianFilter() {
    // 处理内部区域
    for (size_t i = 1; i < size - 1; ++i) {
      for (size_t j = 1; j < size - 1; ++j) {
        result[i * size + j] =
            (figure[(i - 1) * size + j - 1] + 2 * figure[(i - 1)* size + j] +
             figure[(i - 1) * size + j + 1] + 2 * figure[i * size + j - 1] + 4 * figure[i * size + j] +
             2 * figure[i * size + j + 1] + figure[(i + 1) * size + j - 1] +
             2 * figure[(i + 1) * size + j] + figure[(i + 1) * size + j + 1]) /
            16;
      }
    }

    for (size_t i = 1; i < size - 1; ++i) {
      result[i * size] =
          (figure[(i - 1) * size] + 2 * figure[(i - 1) * size] + figure[(i - 1) * size + 1] +
           2 * figure[i * size] + 4 * figure[i * size] + 2 * figure[i * size + 1] +
           figure[(i + 1) * size] + 2 * figure[(i + 1) * size] + figure[(i + 1) * size + 1]) /
          16;

      result[i * size + size - 1] =
          (figure[(i - 1) * size + size - 2] + 2 * figure[(i - 1) * size + size - 1] +
           figure[(i - 1) * size + size - 1] + 2 * figure[i * size + size - 2] +
           4 * figure[i * size + size - 1] + 2 * figure[i * size + size - 1] +
           figure[(i + 1) * size + size - 2] + 2 * figure[(i + 1) * size + size - 1] +
           figure[(i + 1) * size + size - 1]) /
          16;
    }

    for (size_t j = 1; j < size - 1; ++j) {
      result[j] =
          (figure[j - 1] + 2 * figure[j] + figure[j + 1] +
           2 * figure[j - 1] + 4 * figure[j] + 2 * figure[j + 1] +
           figure[size + j - 1] + 2 * figure[size + j] + figure[size + j + 1]) /
          16;

      result[(size - 1) * size + j] =
          (figure[(size - 2) * size + j - 1] + 2 * figure[(size - 2) * size + j] +
           figure[(size - 2) * size + j + 1] + 2 * figure[(size - 1) * size + j - 1] +
           4 * figure[(size - 1) * size + j] + 2 * figure[(size - 1) * size + j + 1] +
           figure[(size - 1) * size + j - 1] + 2 * figure[(size - 1) * size + j] +
           figure[(size - 1) * size + j + 1]) /
          16;
    }

    // 处理四个角点
    // 左上角
    result[0] = (4 * figure[0] + 2 * figure[1] + 2 * figure[size] +
                    figure[size + 1]) /
                   9; 

    // 右上角
    result[size - 1] = (4 * figure[size - 1] + 2 * figure[size - 2] +
                           2 * figure[2 * size - 1] + figure[2 * size - 2]) /
                          9;

    // 左下角
    result[(size - 1) * size] = (4 * figure[(size - 1) * size] + 2 * figure[(size - 1) * size + 1] +
                           2 * figure[(size - 2) * size] + figure[(size - 2) * size + 1]) /
                          9;

    // 右下角
    result[size * size - 1] =
        (4 * figure[size * size - 1] + 2 * figure[size * size - 2] +
         2 * figure[(size - 2) * size + size - 1] + figure[(size - 2) * size + size - 2]) /
        9;
  }

  // Power law transformation
  // FIXME: Feel free to optimize this function
  // Hint: LUT to optimize this function?
  void powerLawTransformation() {
    constexpr float gamma = 0.5f;
    
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        result[i * size + j] = LUT[figure[i * size + j]];
      }
    }
  }

  // Run benchmark
  unsigned int calcChecksum() {
    unsigned int sum = 0;
    constexpr size_t mod = 1000000007;
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        sum += result[i * size + j];
        sum %= mod;
      }
    }
    return sum;
  }
  void runBenchmark() {
    auto start = std::chrono::high_resolution_clock::now();
    gaussianFilter();
    auto middle = std::chrono::high_resolution_clock::now();

    unsigned int sum = calcChecksum();

    auto middle2 = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    auto end = std::chrono::high_resolution_clock::now();

    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";

    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(middle - start) +
        std::chrono::duration_cast<std::chrono::milliseconds>(end - middle2);
    std::cout << "Benchmark time: " << milliseconds.count() << " ms\n";
  }
};

// Main function
// !!! Please do not modify the main function !!!
int main(int argc, const char **argv) {
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}
