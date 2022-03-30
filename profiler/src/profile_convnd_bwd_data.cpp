#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "profile_convnd_bwd_data_impl.hpp"

enum ConvDataType
{
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
};

enum ConvInputLayout
{
    NCHW, // 0
    NHWC, // 1
};

enum ConvWeightLayout
{
    KCYX, // 0
    KYXC, // 1
};

enum ConvOutputLayout
{
    NKHW, // 0
    NHWK, // 1
};
ck::utils::conv::ConvParams parse_conv_params(int num_dim_spatial, char* argv[], int arg_idx)
{
    // (N, K, C) + num_dim_spatial * 6 (filter, input, strides, dilations, pad left, pad right)
    ck::utils::conv::ConvParams params;

    params.num_dim_spatial = num_dim_spatial;
    params.N               = std::stoi(argv[arg_idx++]);
    params.K               = std::stoi(argv[arg_idx++]);
    params.C               = std::stoi(argv[arg_idx++]);

    params.filter_spatial_lengths.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.filter_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_spatial_lengths.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_strides.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_strides[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_dilations.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_dilations[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_left_pads.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_left_pads[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_right_pads.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_right_pads[i] = std::stoi(argv[arg_idx++]);
    }

    return params;
}

int profile_convnd_bwd_data(int argc, char* argv[], int num_dim_spatial)
{
    const int preParams = 10;
    int conv_args       = 3 + num_dim_spatial * 6;
    int cmdline_nargs   = conv_args + preParams;
    if(cmdline_nargs != argc)
    {
        printf("arg1: tensor operation (conv[1|2|3]d_bwd_data: BackwardConvolution)\n");
        printf("arg2: data type (0: fp32; 1: fp16)\n");
        printf("arg3: input tensor layout (0: NCHW; 1: NHWC)\n");
        printf("arg4: weight tensor layout (0: KCYX; 1: KYXC)\n");
        printf("arg5: output tensor layout (0: NKHW; 1: NHWK)\n");
        printf("arg6: verification (0: no; 1: yes)\n");
        printf("arg7: initialization (0: no init; 1: integer value; 2: decimal value)\n");
        printf("arg8: print tensor value (0: no; 1: yes)\n");
        printf("arg9: run kernel # of times (>1)\n");
        printf("arg10 to 24: N, K, C, Y, X, Hi, Wi, Sy, Sx, Dy, Dx, LeftPy, LeftPx, RightPy, "
               "RightPx\n");
        return 1;
    }

    const int data_type        = static_cast<ConvDataType>(std::stoi(argv[2]));
    const int in_layout        = static_cast<ConvInputLayout>(std::stoi(argv[3]));
    const int wei_layout       = static_cast<ConvWeightLayout>(std::stoi(argv[4]));
    const int out_layout       = static_cast<ConvOutputLayout>(std::stoi(argv[5]));
    const bool do_verification = std::stoi(argv[6]);
    const int init_method      = std::stoi(argv[7]);
    const bool do_log          = std::stoi(argv[8]);
    const int nrepeat          = std::stoi(argv[9]);

    ck::utils::conv::ConvParams params = parse_conv_params(num_dim_spatial, argv, preParams);

    auto Run = [&](auto input_type, auto wei_type, auto out_type, auto acc_type) {
        using InDataType  = decltype(input_type);
        using WeiDataType = decltype(wei_type);
        using OutDataType = decltype(out_type);
        using AccDataType = decltype(acc_type);

        switch(num_dim_spatial)
        {
        case 1:
            ck::profiler::profile_convnd_bwd_data_impl<1,
                                                       InDataType,
                                                       WeiDataType,
                                                       OutDataType,
                                                       AccDataType,
                                                       ck::tensor_layout::convolution::NWC,
                                                       ck::tensor_layout::convolution::KXC,
                                                       ck::tensor_layout::convolution::NWK>(
                do_verification,
                init_method,
                do_log,
                nrepeat,
                params.N,
                params.K,
                params.C,
                params.input_spatial_lengths,
                params.filter_spatial_lengths,
                params.GetOutputSpatialLengths(),
                params.conv_filter_strides,
                params.conv_filter_dilations,
                params.input_left_pads,
                params.input_right_pads);
            break;

        case 2:
            ck::profiler::profile_convnd_bwd_data_impl<2,
                                                       InDataType,
                                                       WeiDataType,
                                                       OutDataType,
                                                       AccDataType,
                                                       ck::tensor_layout::convolution::NHWC,
                                                       ck::tensor_layout::convolution::KYXC,
                                                       ck::tensor_layout::convolution::NHWK>(
                do_verification,
                init_method,
                do_log,
                nrepeat,
                params.N,
                params.K,
                params.C,
                params.input_spatial_lengths,
                params.filter_spatial_lengths,
                params.GetOutputSpatialLengths(),
                params.conv_filter_strides,
                params.conv_filter_dilations,
                params.input_left_pads,
                params.input_right_pads);
            break;

        case 3:
            ck::profiler::profile_convnd_bwd_data_impl<3,
                                                       InDataType,
                                                       WeiDataType,
                                                       OutDataType,
                                                       AccDataType,
                                                       ck::tensor_layout::convolution::NDHWC,
                                                       ck::tensor_layout::convolution::KZYXC,
                                                       ck::tensor_layout::convolution::NDHWK>(
                do_verification,
                init_method,
                do_log,
                nrepeat,
                params.N,
                params.K,
                params.C,
                params.input_spatial_lengths,
                params.filter_spatial_lengths,
                params.GetOutputSpatialLengths(),
                params.conv_filter_strides,
                params.conv_filter_dilations,
                params.input_left_pads,
                params.input_right_pads);
            break;

        default: break;
        }
    };
    if(data_type == ConvDataType::F32_F32_F32 && in_layout == ConvInputLayout::NHWC &&
       wei_layout == ConvWeightLayout::KYXC && out_layout == ConvOutputLayout::NHWK)
    {
        Run(float{}, float{}, float{}, float{});
    }
    else if(data_type == ConvDataType::F16_F16_F16 && in_layout == ConvInputLayout::NHWC &&
            wei_layout == ConvWeightLayout::KYXC && out_layout == ConvOutputLayout::NHWK)
    {
        Run(ck::half_t{}, ck::half_t{}, ck::half_t{}, float{});
    }
    else if(data_type == ConvDataType::BF16_BF16_BF16 && in_layout == ConvInputLayout::NHWC &&
            wei_layout == ConvWeightLayout::KYXC && out_layout == ConvOutputLayout::NHWK)
    {
        Run(ck::bhalf_t{}, ck::bhalf_t{}, ck::bhalf_t{}, float{});
    }
    else if(data_type == ConvDataType::INT8_INT8_INT8 && in_layout == ConvInputLayout::NHWC &&
            wei_layout == ConvWeightLayout::KYXC && out_layout == ConvOutputLayout::NHWK)
    {
        Run(int8_t{}, int8_t{}, int8_t{}, int32_t{});
    }
    else
    {
        std::cout << "wrong! this Conv data_type & layout is not implemented" << std::endl;
        return 1;
    }

    return 0;
}
