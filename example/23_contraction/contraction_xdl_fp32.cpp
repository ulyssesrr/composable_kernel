#include <iostream>
#include <fstream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "device_contraction_xdl_cshuffle.hpp"
#include "element_wise_operation.hpp"
#include "reference_gemm.hpp"
#include "gemm_specialization.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using ADataType   = float;
using BDataType   = float;
using CDataType   = float;
using AccDataType = float;

static constexpr ck::index_t NumDimM = 2;
static constexpr ck::index_t NumDimN = 2;
static constexpr ck::index_t NumDimK = 2;

using AElementOp = ck::tensor_operation::element_wise::PassThrough;
using BElementOp = ck::tensor_operation::element_wise::PassThrough;
using CElementOp = ck::tensor_operation::element_wise::PassThrough;

//static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::MNKPadding;


// (M0 * M1 ...) % MPerBlock == 0
// (N0 * N1 ...) % NPerBlock == 0
// (K0 * K1 ...) % KPerBlock == 0
//
//
//
// clang-format off
// Fast changing dimension in A/B/C are K/N/N dimensions
using ContractionInstanceKNN = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   4,   1,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              1,         0,           1,           1,              S<1, 16, 1, 16>,               4>;

// Fast changing dimension in A/B/C are K/K/N dimensions
using ContractionInstanceKKN = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   4,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              4,              4,         1,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,              S<1, 16, 1, 16>,               4>;

// Fast changing dimension in A/B/C are M/N/N dimensions
using ContractionInstanceMNN = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   1,   1,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              1,         0,     S<8, 32, 1>,     S<0, 2, 1>,     S<0, 2, 1>,             1,              4,              1,         0,           1,           1,              S<1, 16, 1, 16>,               4>;

// Fast changing dimension in A/B/C are M/K/N dimensions
using ContractionInstanceMKN = ck::tensor_operation::device::
        //############################| NumDimM| NumDimN| NumDimK| AData| BData| CData| AccData| CShuffle|           A|           B|           C|           GEMM| NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer|
        //############################|        |        |        |  Type|  Type|  Type|    Type| DataType| Elementwise| Elementwise| Elementwise| Spacialization| Prefetch|  Size| Block| Block| Block|    |    |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|
        //############################|        |        |        |      |      |      |        |         |   Operation|   Operation|   Operation|               |    Stage|      |      |      |      |    |    |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|
        //############################|        |        |        |      |      |      |        |         |            |            |            |               |         |      |      |      |      |    |    |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |
        DeviceContraction_Xdl_CShuffle< NumDimM, NumDimN, NumDimK,   F32,   F32,   F32,     F32,      F32, PassThrough, PassThrough, PassThrough,    GemmDefault,        1,   256,   256,   128,    16,   1,   4,   32,   32,    4,    2,     S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              1,         0,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,             2,              4,              4,         1,           1,           1,               S<1, 16, 1, 16>,              4>;
// clang-format on

using ContractionInstance = ContractionInstanceKKN;

template <typename T, typename Range>
void LogRangeToFile(std::ofstream& fs, Range&& range, std::string delim)
{
    bool first = true;
    for(auto&& v : range)
    {
        if(first)
            first = false;
        else
            fs << delim;
        fs << static_cast<T>(v);
    }
    return;
}


// hardcoded for NumDimM == NumDimN == NumDimK == 2
template <ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::enable_if_t<NumDimM == 2 && NumDimN == 2 && NumDimK == 2, bool> = false>
struct ReferenceContraction_M2_N2_K2 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_ms_ks,
                 const Tensor<BDataType>& b_ks_ns,
                 Tensor<CDataType>& c_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_ms_ks_{a_ms_ks},
              b_ks_ns_{b_ks_ns},
              c_ms_ns_{c_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ADataType>& a_ms_ks_;
        const Tensor<BDataType>& b_ks_ns_;
        Tensor<CDataType>& c_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceContraction_M2_N2_K2::Argument;

        float Run(const Argument& arg)
        {
            auto f_ms_ns = [&](auto m0, auto m1, auto n0, auto n1) {
                const int K0 = arg.a_ms_ks_.mDesc.GetLengths()[2];
                const int K1 = arg.a_ms_ks_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(int k0 = 0; k0 < K0; ++k0)
                {
                    for(int k1 = 0; k1 < K1; ++k1)
                    {
                        AccDataType v_a;
                        AccDataType v_b;

                        arg.a_element_op_(
                            v_a, static_cast<const AccDataType>(arg.a_ms_ks_(m0, m1, k0, k1)));
                        arg.b_element_op_(
                            v_b, static_cast<const AccDataType>(arg.b_ks_ns_(k0, k1, n0, n1)));

                        v_acc += v_a * v_b;
                    }
                }

                AccDataType v_c;

                arg.c_element_op_(v_c, v_acc);

                arg.c_ms_ns_(m0, m1, n0, n1) = v_c;
            };

            make_ParallelTensorFunctor(f_ms_ns,
                                       arg.c_ms_ns_.mDesc.GetLengths()[0],
                                       arg.c_ms_ns_.mDesc.GetLengths()[1],
                                       arg.c_ms_ns_.mDesc.GetLengths()[2],
                                       arg.c_ms_ns_.mDesc.GetLengths()[3])(
                std::thread::hardware_concurrency());

            return 0;
        }

        float Run(const ck::tensor_operation::device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const ck::tensor_operation::device::BaseArgument*) override
    {
        return true;
    }

    static auto MakeArgument(const Tensor<ADataType>& a_ms_ks,
                             const Tensor<BDataType>& b_ks_ns,
                             Tensor<CDataType>& c_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{a_ms_ks, b_ks_ns, c_ms_ns, a_element_op, b_element_op, c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<ck::tensor_operation::device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceContraction_M2_N2_K2"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

using ReferenceOpInstance = ReferenceContraction_M2_N2_K2<NumDimM,
                                                          NumDimN,
                                                          NumDimK,
                                                          ADataType,
                                                          BDataType,
                                                          CDataType,
                                                          AccDataType,
                                                          AElementOp,
                                                          BElementOp,
                                                          CElementOp>;

int main(int argc, char* argv[])
{
    bool do_verification = true;
    int init_method      = 3;
    bool time_kernel     = false;

    if(argc == 4)
    {
        do_verification = std::stoi(argv[1]);
        init_method     = std::stoi(argv[2]);
        time_kernel     = std::stoi(argv[3]);
    }
    else
    {
        printf("arg1: verification (0=no, 1=yes)\n");
        printf("arg2: initialization (0=no init, 1=integer value, 2=decimal value, 3=cutensor_style_init)\n");
        printf("arg3: time kernel (0=no, 1=yes)\n");
        exit(0);
    }

    std::ofstream tensorA;
    std::ofstream tensorB;
    std::ofstream tensorC;
    std::ofstream tensorC_d;
    std::cout << "RAND_MAX value is " << RAND_MAX << std::endl;

    
    // Physical layout
    // A[m0, k0, m1, k1]   : leng [5, 6, 3, 4], stride  [108, 20, 16, 1]
    // B[k0, n0, k1, n1]
    // C[m0, m1, n0, n1]


    // logic layout
    // A[m0, m1, k0, k1]   : leng [5, 3, 6, 4], stride [108, 16, 20, 1]  K is fast changing
    // C[k0, k1, n0, n1]
#if 1
    // fast changing dimension: K/K/N
    // a[m0, m1, k0, k1]
    std::vector<ck::index_t> a_ms_ks_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> a_ms_ks_strides{524288, 4096, 128, 1};
    // b[k0, k1, n0, n1]
    std::vector<ck::index_t> b_ks_ns_lengths{32, 64, 32, 64};
    std::vector<ck::index_t> b_ks_ns_strides{128, 1, 524288, 4096};
    // c[m0, m1, n0, n1]
    std::vector<ck::index_t> c_ms_ns_lengths{30, 128, 32, 64};
    std::vector<ck::index_t> c_ms_ns_strides{524288, 4096, 128, 1};
#elif 0
    // fast changing dimension: K/N/N
    // a[m0, m1, k0, k1]
    std::vector<ck::index_t> a_ms_ks_lengths{5,6,3,4};
    std::vector<ck::index_t> a_ms_ks_strides{108,20,16,1};
    // b[k0, k1, n0, n1]
    std::vector<ck::index_t> b_ks_ns_lengths{3,4,3,4};
    std::vector<ck::index_t> b_ks_ns_strides{48,12,4,1};
    // c[m0, m1, n0, n1]
    std::vector<ck::index_t> c_ms_ns_lengths{5,6,3,4};
    std::vector<ck::index_t> c_ms_ns_strides{108,20,16,1};
#elif 1
    // fast changing dimension: K/K/N
    // a[m0, m1, k0, k1]
    std::vector<ck::index_t> a_ms_ks_lengths{5,6,3,4};
    std::vector<ck::index_t> a_ms_ks_strides{108,20,16,1};
    // b[k0, k1, n0, n1]
    std::vector<ck::index_t> b_ks_ns_lengths{3,4,3,4};
    std::vector<ck::index_t> b_ks_ns_strides{16,1,108,20};
    // c[m0, m1, n0, n1]
    std::vector<ck::index_t> c_ms_ns_lengths{5,6,3,4};
    std::vector<ck::index_t> c_ms_ns_strides{108,20,16,1};
#elif 0
    // fast changing dimension: M/N/N
    // a[m0, m1, k0, k1]
    std::vector<ck::index_t> a_ms_ks_lengths{5,6,3,4};
    std::vector<ck::index_t> a_ms_ks_strides{6,1,72,24};
    // b[k0, k1, n0, n1]
    std::vector<ck::index_t> b_ks_ns_lengths{3,4,3,4};
    std::vector<ck::index_t> b_ks_ns_strides{48,12,4,1};
    // c[m0, m1, n0, n1]
    std::vector<ck::index_t> c_ms_ns_lengths{5,6,3,4};
    std::vector<ck::index_t> c_ms_ns_strides{108,20,16,1};
#elif 1
    // fast changing dimension: M/K/N
    // a[m0, m1, k0, k1]
    std::vector<ck::index_t> a_ms_ks_lengths{5,6,3,4};
    std::vector<ck::index_t> a_ms_ks_strides{6,1,72,24};
    // b[k0, k1, n0, n1]
    std::vector<ck::index_t> b_ks_ns_lengths{3,4,3,4};
    std::vector<ck::index_t> b_ks_ns_strides{16,1,108,20};
    // c[m0, m1, n0, n1]
    std::vector<ck::index_t> c_ms_ns_lengths{5,6,3,4};
    std::vector<ck::index_t> c_ms_ns_strides{108,20,16,1};
#endif

    Tensor<ADataType> a_ms_ks(
        std::vector<std::size_t>(a_ms_ks_lengths.begin(), a_ms_ks_lengths.end()),
        std::vector<std::size_t>(a_ms_ks_strides.begin(), a_ms_ks_strides.end()));
    Tensor<BDataType> b_ks_ns(
        std::vector<std::size_t>(b_ks_ns_lengths.begin(), b_ks_ns_lengths.end()),
        std::vector<std::size_t>(b_ks_ns_strides.begin(), b_ks_ns_strides.end()));
    Tensor<CDataType> c_ms_ns_host_result(
        std::vector<std::size_t>(c_ms_ns_lengths.begin(), c_ms_ns_lengths.end()),
        std::vector<std::size_t>(c_ms_ns_strides.begin(), c_ms_ns_strides.end()));
    Tensor<CDataType> c_ms_ns_device_result(
        std::vector<std::size_t>(c_ms_ns_lengths.begin(), c_ms_ns_lengths.end()),
        std::vector<std::size_t>(c_ms_ns_strides.begin(), c_ms_ns_strides.end()));

    std::cout << "a_ms_ks: " << a_ms_ks.mDesc << std::endl;
    std::cout << "b_ks_ns: " << b_ks_ns.mDesc << std::endl;
    std::cout << "c_ms_ns: " << c_ms_ns_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_ks_ns.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    case 2:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_ks_ns.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    case 3:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_cuTensor<ADataType>{});
        b_ks_ns.GenerateTensorValue(GeneratorTensor_cuTensor<BDataType>{});
        break;
    default:
        a_ms_ks.GenerateTensorValue(GeneratorTensor_Sequential<0>{});
        b_ks_ns.GenerateTensorValue(GeneratorTensor_Sequential<1>{});
    }

    DeviceMem a_ms_ks_device_buf(sizeof(ADataType) * a_ms_ks.mDesc.GetElementSpace());
    DeviceMem b_ks_ns_device_buf(sizeof(BDataType) * b_ks_ns.mDesc.GetElementSpace());
    DeviceMem c_ms_ns_device_buf(sizeof(CDataType) * c_ms_ns_device_result.mDesc.GetElementSpace());

    std::cout << "Tensor A element space: " << a_ms_ks.mDesc.GetElementSpace() << std::endl;
    std::cout << "Tensor B element space: " << b_ks_ns.mDesc.GetElementSpace() << std::endl;
    std::cout << "Tensor C element space: " <<  c_ms_ns_device_result.mDesc.GetElementSpace() << std::endl;

    a_ms_ks_device_buf.ToDevice(a_ms_ks.mData.data());
    b_ks_ns_device_buf.ToDevice(b_ks_ns.mData.data());

    // set zero
    c_ms_ns_device_buf.SetZero();

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    // device operation
    auto op       = ContractionInstance{};
    auto invoker  = op.MakeInvoker();
    auto argument = op.MakeArgument(static_cast<ADataType*>(a_ms_ks_device_buf.GetDeviceBuffer()),
                                    static_cast<BDataType*>(b_ks_ns_device_buf.GetDeviceBuffer()),
                                    static_cast<CDataType*>(c_ms_ns_device_buf.GetDeviceBuffer()),
                                    a_ms_ks_lengths,
                                    std::vector<ck::index_t>(a_ms_ks.mDesc.mStrides.begin(), a_ms_ks.mDesc.mStrides.end()),
                                    b_ks_ns_lengths,
				    std::vector<ck::index_t>(b_ks_ns.mDesc.mStrides.begin(), b_ks_ns.mDesc.mStrides.end()),
                                    c_ms_ns_lengths,
                                    std::vector<ck::index_t>(c_ms_ns_host_result.mDesc.mStrides.begin(), c_ms_ns_host_result.mDesc.mStrides.end()),
                                    a_element_op,
                                    b_element_op,
                                    c_element_op);

    if(!op.IsSupportedArgument(argument))
    {
        std::cout << op.GetTypeString() << " does not support this problem" << std::endl;

        return 0;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    ck::index_t M = std::accumulate(c_ms_ns_lengths.begin(),
                                    c_ms_ns_lengths.begin() + NumDimM,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    ck::index_t N = std::accumulate(c_ms_ns_lengths.begin() + NumDimM,
                                    c_ms_ns_lengths.begin() + NumDimM + NumDimN,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    ck::index_t K = std::accumulate(a_ms_ks_lengths.begin() + NumDimM,
                                    a_ms_ks_lengths.begin() + NumDimM + NumDimK,
                                    ck::index_t{1},
                                    std::multiplies<ck::index_t>{});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << op.GetTypeString() << std::endl;

    c_ms_ns_device_buf.FromDevice(c_ms_ns_device_result.mData.data());

#if 0
    tensorA.open("tensor_A.txt");
    LogRangeToFile<ADataType>(tensorA, a_ms_ks.mData, ","); 
    LogRangeAsType<ADataType>(std::cout<<"Tensor A elements:\n", a_ms_ks.mData,",");
    std::cout<<std::endl;
    tensorA.close();
    tensorB.open("tensor_B.txt");
    LogRangeToFile<BDataType>(tensorB, b_ks_ns.mData, ","); 
    LogRangeAsType<BDataType>(std::cout<<"Tensor B elements:\n", b_ks_ns.mData,",");
    std::cout<<std::endl;
    tensorB.close();
#endif

    if(do_verification)
    {
        auto ref_gemm    = ReferenceOpInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_ms_ks, b_ks_ns, c_ms_ns_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

#if 0
	    tensorC.open("tensor_C_contraction_host_results.txt");
    	LogRangeToFile<CDataType>(tensorC, c_ms_ns_host_result.mData, ","); 
    	LogRangeAsType<CDataType>(std::cout<<"Tensor C_host elements:\n", c_ms_ns_host_result.mData, ",");
    	std::cout<<std::endl;
	    tensorC.close();

	    tensorC.open("tensor_C_contraction_device_results.txt");
    	LogRangeToFile<CDataType>(tensorC_d, c_ms_ns_device_result.mData, ","); 
    	LogRangeAsType<CDataType>(std::cout<<"Tensor C_device elements:\n", c_ms_ns_device_result.mData, ",");
    	std::cout<<std::endl;
	    tensorC.close();
#endif


        return ck::utils::check_err(c_ms_ns_device_result.mData, c_ms_ns_host_result.mData) ? 0 : 1;
    }

    return 0;
}
