#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <omp.h>

class FigureProcessor {
private:
  uint16_t* figure;
  uint16_t* result;
  float LUT[256];
  const size_t size;
  const size_t outsize;

public:
  FigureProcessor(size_t size, size_t seed = 0) : size(size), outsize(size+2){
    
    // !!! Please do not modify the following code !!!
    // 如果你需要修改内存的数据结构，请不要修改初始化的顺序和逻辑
    // 助教可能会通过指定某个初始化seed 的方式来验证你的代码
    // 如果你修改了初始化的顺序，可能会导致你的代码无法通过测试
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // 两个数组的初始化在这里，可以改动，但请注意 gen 的顺序是从上到下从左到右即可。

    omp_set_num_threads(4);  // 设置使用 4 个线程

    // 对齐到32字节
    figure=(uint16_t*)_aligned_malloc(outsize * outsize * sizeof(uint16_t), 32);
    result=(uint16_t*)_aligned_malloc(size * size * sizeof(uint16_t), 32);

    //内部赋值
    for (size_t i = 1; i < size + 1; ++i) {
      for (size_t j = 1; j < size + 1; ++j) {
        figure[i * outsize + j]=static_cast<uint16_t>(distribution(gen));
      }
    }

    //补边，周围补一圈与最近点相同的点
    for (size_t i = 0; i < outsize; ++i) {
      figure[i * outsize] = figure[i * outsize + 1];
    }
    for (size_t i = 0; i < outsize; ++i) {
      figure[i * outsize + outsize - 1] = figure[i * outsize + outsize - 2];
    }
    for (size_t j = 0; j < outsize; ++j) {
      figure[j] = figure[outsize + j];
    }
    for (size_t j = 0; j < outsize; ++j) {
      figure[(outsize - 1) * outsize + j] = figure[(outsize - 2) * outsize + j];
    }

    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        result[i * size + j] = 0;
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
    _aligned_free(figure);
    _aligned_free(result);
  }

  // Gaussian filter
  // [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16
  //FIXME: Feel free to optimize this function
  //Hint: You can use SIMD instructions to optimize this function
  void gaussianFilter() {
    // 每个像素占16位，因此AVX_SIZE=256/16=16
    size_t AVX_SIZE = 16;
    #pragma omp parallel for collapse(2)
    for (size_t i = 1; i< size + 1; ++i) {
      for (size_t j = 1; j < size + 1; j += AVX_SIZE) {
        // 载入数据
        __m256i figure_top_left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[(i - 1) * outsize + j - 1]));
        __m256i figure_top_mid = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[(i - 1) * outsize + j]));
        __m256i figure_top_right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[(i - 1) * outsize + j + 1]));

        __m256i figure_mid_left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[i * outsize + j - 1]));
        __m256i figure_mid_mid = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[i * outsize + j]));
        __m256i figure_mid_right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[i * outsize + j + 1]));

        __m256i figure_bottom_left = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[(i + 1) * outsize + j - 1]));
        __m256i figure_bottom_mid = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[(i + 1) * outsize + j]));
        __m256i figure_bottom_right = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[(i + 1) * outsize + j + 1]));

        // 将系数与数据相乘，进行加权求和
        __m256i sum = figure_top_left;
        sum = _mm256_add_epi16(sum, _mm256_slli_epi16(figure_top_mid, 1));
        sum = _mm256_add_epi16(sum, figure_top_right);

        sum = _mm256_add_epi16(sum, _mm256_slli_epi16(figure_mid_left, 1));
        sum = _mm256_add_epi16(sum, _mm256_slli_epi16(figure_mid_mid, 2));
        sum = _mm256_add_epi16(sum, _mm256_slli_epi16(figure_mid_right, 1));

        sum = _mm256_add_epi16(sum, figure_bottom_left);
        sum = _mm256_add_epi16(sum, _mm256_slli_epi16(figure_bottom_mid, 1));
        sum = _mm256_add_epi16(sum, figure_bottom_right);

        // 除以 16，即右移4位
        sum = _mm256_srli_epi16(sum, 4);

        // 将结果存回 `result` 数组
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[((i - 1) << 14) + j - 1]), sum);
      }
    }
    // 处理四个角点
    // 左上角
    result[0] = ((figure[outsize + 1] << 2) + (figure[outsize + 2] << 1) + (figure[(outsize << 1) + 1] << 1) +
                    figure[(outsize << 1) + 2]) /
                   9; 

    // 右上角
    result[size - 1] = ((figure[(outsize << 1) - 2] << 2) + (figure[(outsize << 1) - 3] << 1) +
                          (figure[3 * outsize - 2] << 1) + figure[3 * outsize - 3]) /
                          9;

    // 左下角
    result[(size - 1) << 14] = ((figure[(outsize - 2) * outsize + 1] << 2) + (figure[(outsize - 2) * outsize + 2] << 1) +
                           (figure[(outsize - 3) * outsize + 1] << 1) + figure[(outsize - 3) * outsize + 2]) /
                          9;

    // 右下角
    result[(size << 14) - 1] =((figure[(outsize - 2) * outsize + outsize - 2] << 2) + (figure[(outsize - 2) * outsize + outsize - 3] << 1) +
                            (figure[(outsize - 3) * outsize + outsize - 2] << 1) + figure[(outsize - 3) * outsize + outsize - 3]) /
                            9;
  }

  // Power law transformation
  // FIXME: Feel free to optimize this function
  // Hint: LUT to optimize this function?
  void powerLawTransformation() {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        result[(i << 14) + j] = LUT[figure[(i + 1) * outsize + j + 1]];
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
