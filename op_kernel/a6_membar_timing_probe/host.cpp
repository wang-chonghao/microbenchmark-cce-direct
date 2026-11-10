#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "acl/acl.h"
#include "runtime/kernel.h"
#include "runtime/rt.h"

namespace {
void save(const char *path, const void *data, size_t bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char *>(data), bytes);
    if (!file) std::fprintf(stderr, "Cannot write %s\n", path);
}
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: host kernel.o kernel_name\n");
        return 64;
    }
    std::ifstream file(argv[1], std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(file)), {});
    if (code.empty()) return 65;
    std::vector<float> input(64);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    // One extra 256B block is a guard against an oversized output DMA.
    std::vector<unsigned char> output(2048 + 256, 0xa5);
    std::vector<unsigned char> golden = output;
    for (size_t block = 0; block < 8; ++block) {
        std::memcpy(golden.data() + block * 256, input.data(), 256);
    }
    void *src = nullptr;
    void *dst = nullptr;
    void *handle = nullptr;
    void *function = nullptr;
    rtStream_t stream = nullptr;
    rtDevBinary_t binary {};
    void *args[2] {};
    bool initialized = false;
    bool device = false;
    int rc = 0;
#define CHECK(call) do { auto status = (call); if (status != 0) { \
    std::fprintf(stderr, "%s failed: %d\n", #call, int(status)); rc = 1; goto cleanup; } } while (0)
    CHECK(aclInit(nullptr));
    initialized = true;
    CHECK(rtSetDevice(0));
    device = true;
    CHECK(rtStreamCreate(&stream, 0));
    CHECK(rtMalloc(&src, 256, RT_MEMORY_HBM, 0));
    CHECK(rtMalloc(&dst, output.size(), RT_MEMORY_HBM, 0));
    CHECK(rtMemcpy(src, 256, input.data(), 256, RT_MEMCPY_HOST_TO_DEVICE));
    CHECK(rtMemcpy(dst, output.size(), output.data(), output.size(), RT_MEMCPY_HOST_TO_DEVICE));
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    binary.length = code.size();
    binary.data = code.data();
    CHECK(rtDevBinaryRegister(&binary, &handle));
    CHECK(rtFunctionRegister(handle, argv[2], reinterpret_cast<const char_t *>(argv[2]), argv[2], 0));
    CHECK(rtGetFunctionByName(reinterpret_cast<const char_t *>(argv[2]), &function));
    args[0] = src;
    args[1] = dst;
    CHECK(rtKernelLaunch(function, 1, args, sizeof(args), nullptr, stream));
    CHECK(rtStreamSynchronize(stream));
    CHECK(rtMemcpy(output.data(), output.size(), dst, output.size(), RT_MEMCPY_DEVICE_TO_HOST));
    save("input.bin", input.data(), 256);
    save("golden.bin", golden.data(), golden.size());
    save("output.bin", output.data(), output.size());
    if (output != golden) {
        size_t bad = 0;
        while (bad < output.size() && output[bad] == golden[bad]) ++bad;
        std::fprintf(stderr, "[CHECK] FAIL byte=%zu actual=%02x expected=%02x\n",
                     bad, unsigned(output[bad]), unsigned(golden[bad]));
        rc = 2;
    } else {
        std::printf("[CHECK] PASS: all 8 blocks bitwise equal, 256B output guard intact\n");
    }
cleanup:
    if (dst) rtFree(dst);
    if (src) rtFree(src);
    if (stream) rtStreamDestroy(stream);
    if (device) rtDeviceReset(0);
    if (initialized) aclFinalize();
    return rc;
}
