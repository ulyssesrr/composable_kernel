import argparse, itertools, operator, ast, re

"""
bs [1, 2, 4, 8, 16, 32, 64]
sl [16, 32, 64, 128, 256, 384, 512]
hs [768 (bert-base), 1024 (bert-large)]
nh [12(bert-base), 16(bert-large)]
./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m hs -n bs*sl -k hs …
./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m hs*4 -n bs*sl -k hs
./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m hs -n bs*sl -k hs*4
./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m hs*3 -n bs*sl -k 1
./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m hs*3 -n bs*sl -k hs
./rocblas-bench -f gemm_strided_batched_ex --transposeA T --transposeB N -m sl -n sl -k sl/nh …… --batch_count bs*nh
./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB N -m sl/nh -n sl -k sl …… --batch_count bs*nh
"""

bs = [1, 64]
sl = [384]
hs = [768, 1024]
nh = [12, 16]

cmdlist = [
"-f gemm_ex --transposeA N --transposeB N -m hs -n bs*sl -k hs",
"-f gemm_ex --transposeA N --transposeB N -m hs*4 -n bs*sl -k hs",
"-f gemm_ex --transposeA N --transposeB N -m hs -n bs*sl -k hs*4",
"-f gemm_ex --transposeA N --transposeB N -m hs*3 -n bs*sl -k 1",
"-f gemm_ex --transposeA N --transposeB N -m hs*3 -n bs*sl -k hs",
"-f gemm_strided_batched_ex --transposeA T --transposeB N -m sl -n sl -k sl/nh --batch_count bs*nh",
"-f gemm_strided_batched_ex --transposeA N --transposeB N -m sl/nh -n sl -k sl --batch_count bs*nh"
]

def multiply(lhs, rhs):
    return [operator.mul(lhs_, rhs_) for lhs_, rhs_ in zip(lhs, rhs)]

def divide(lhs, rhs):
    return [operator.floordiv(lhs_, rhs_) for lhs_, rhs_ in zip(lhs, rhs)]

operators = {ast.Mult: multiply, ast.Div: divide}

def eval_(node, accessor):
    if isinstance(node, ast.Num):
        return itertools.repeat(node.n)
    elif isinstance(node, ast.Name):
        return accessor(node.id)
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](eval_(node.left, accessor), eval_(node.right, accessor))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](eval_(node.operand, accessor))
    else:
        raise TypeError(node)

# from https://stackoverflow.com/a/9558001/692438
def eval_expr(expr, accessor = lambda key : globals()[key]):
    return eval_(ast.parse(expr, mode='eval').body, accessor)

def unpack_invoke(f, params):
    ret = []
    for p in params:
        ret.append(f(p))
    return ret

parser = argparse.ArgumentParser()
parser.add_argument("-f")
parser.add_argument("-m")
parser.add_argument("-n")
parser.add_argument("-k")
parser.add_argument("--transposeA")
parser.add_argument("--transposeB")
parser.add_argument("--batch_count", default=None)

typemap = {'fp32': 0, 'fp16': 1, 'bf16': 2, 'int8': 3}
verification = 0
initialization = 1
printval = 0
iters = 10
layout = {'TT': 0, 'TN': 1, 'NT': 2, 'NN': 3}

for cmd in cmdlist:
    args = parser.parse_args(cmd.split(' '))
    tokens = re.split(r'[^a-zA-Z]', ','.join(filter(None, [args.m, args.n, args.k, args.batch_count])))
    tokens = list(set(filter(None, tokens)))
    # print('tokens =', tokens)
    # print('eval = ', list(unpack_invoke(eval_expr, tokens)))
    # print('product = ', list(itertools.product(*list(unpack_invoke(eval_expr, tokens)))))
    expanded = list(itertools.product(*list(unpack_invoke(eval_expr, tokens))))
    eval_expr_ = lambda arg : eval_expr(arg, lambda key : [ i[tokens.index(key)] for i in expanded ])
    if args.batch_count is None:
        [print(f'./ckProfiler gemm {typemap["fp16"]} {layout[args.transposeA + args.transposeB]} {verification} {initialization} {printval} {iters} {m} {n} {k} -1 -1 -1') for m, n, k in \
            zip(eval_expr_(args.m), eval_expr_(args.n), eval_expr_(args.k))]
    else:
        [print(f'./ckProfiler batched_gemm {typemap["fp16"]} {layout[args.transposeA + args.transposeB]} {verification} {initialization} {printval} {iters} {m} {n} {k} -1 -1 -1 {b}') for m, n, k, b in \
            zip(eval_expr_(args.m), eval_expr_(args.n), eval_expr_(args.k), eval_expr_(args.batch_count))]
