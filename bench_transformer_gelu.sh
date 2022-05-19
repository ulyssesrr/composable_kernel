# python3 generate_transformer_sizes.py | sort -u
# bs = [1, 64]
# sl = [384]
# hs = [768, 1024]
# nh = [12, 16]
set -x
# ./ckProfiler batched_gemm 1 1 0 1 0 10 384 384 24 -1 -1 -1 1024
# ./ckProfiler batched_gemm 1 1 0 1 0 10 384 384 24 -1 -1 -1 16
# ./ckProfiler batched_gemm 1 1 0 1 0 10 384 384 32 -1 -1 -1 12
# ./ckProfiler batched_gemm 1 1 0 1 0 10 384 384 32 -1 -1 -1 768
# ./ckProfiler batched_gemm 1 3 0 1 0 10 24 384 384 -1 -1 -1 1024
# ./ckProfiler batched_gemm 1 3 0 1 0 10 24 384 384 -1 -1 -1 16
# ./ckProfiler batched_gemm 1 3 0 1 0 10 32 384 384 -1 -1 -1 12
# ./ckProfiler batched_gemm 1 3 0 1 0 10 32 384 384 -1 -1 -1 768
./ckProfiler gemm 1 3 0 1 0 10 1024 24576 1024 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 1024 24576 4096 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 1024 384 1024 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 1024 384 4096 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 2304 24576 1 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 2304 24576 768 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 2304 384 1 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 2304 384 768 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 3072 24576 1 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 3072 24576 1024 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 3072 24576 768 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 3072 384 1 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 3072 384 1024 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 3072 384 768 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 4096 24576 1024 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 4096 384 1024 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 768 24576 3072 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 768 24576 768 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 768 384 3072 -1 -1 -1
./ckProfiler gemm 1 3 0 1 0 10 768 384 768 -1 -1 -1