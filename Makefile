# 定义 CUDA 编译器
NVCC = nvcc

# 定义目标文件
TARGETS = lib/deepgpr.so

# 默认目标
all: $(TARGETS)

# 规则：编译 lib/ 目录中的 .cu 文件
lib/%.so: lib/%.cu
	$(NVCC) -shared -Xcompiler -fPIC -o $@ $< -D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm -f $(TARGETS)
