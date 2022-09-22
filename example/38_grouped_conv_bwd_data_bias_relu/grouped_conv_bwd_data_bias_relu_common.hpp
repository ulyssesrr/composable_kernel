// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <numeric>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/array.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

void print_helper_msg()
{
    std::cout << "arg1: verification (0=no, 1=yes)\n"
              << "arg2: initialization (0=no init, 1=integer value, 2=decimal value)\n"
              << "arg3: time kernel (0=no, 1=yes)\n"
              << ck::utils::conv::get_conv_param_parser_helper_msg() << std::endl;
}

template <ck::index_t NDimSpatial,
          typename OutDataType,
          typename WeiDataType,
          typename BiasDataType,
          typename InDataType,
          typename OutElementOp,
          typename WeiElementOp,
          typename InElementOp,
          typename DeviceInstance>
int run_conv_bwd_data_bias_relu(bool do_verification,
                                int init_method,
                                bool time_kernel,
                                const ck::utils::conv::ConvParam& conv_param,
                                const HostTensorDescriptor& out_g_n_k_wos_desc,
                                const HostTensorDescriptor& wei_g_k_c_xs_desc,
                                const HostTensorDescriptor& bias_g_n_c_wis_desc,
                                const HostTensorDescriptor& in_g_n_c_wis_desc,
                                const OutElementOp& out_element_op,
                                const WeiElementOp& wei_element_op,
                                const InElementOp& in_element_op)
{
    Tensor<OutDataType> out(out_g_n_k_wos_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<BiasDataType> bias(bias_g_n_c_wis_desc);
    Tensor<InDataType> in_host(in_g_n_c_wis_desc);
    Tensor<InDataType> in_device(in_g_n_c_wis_desc);

    std::cout << "out: " << out.GetDesc() << std::endl;
    std::cout << "wei: " << wei.GetDesc() << std::endl;
    std::cout << "bias: " << bias.GetDesc() << std::endl;
    std::cout << "in: " << in_host.GetDesc() << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        bias.GenerateTensorValue(GeneratorTensor_2<BiasDataType>{-5, 5});
        break;
    default:
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        bias.GenerateTensorValue(GeneratorTensor_3<BiasDataType>{0.0, 1.0});
    }

    DeviceMem out_device_buf(out.GetMemorySize());
    DeviceMem wei_device_buf(wei.GetMemorySize());
    DeviceMem bias_device_buf(bias.GetMemorySize());
    DeviceMem in_device_buf(in_device.GetMemorySize());

    out_device_buf.ToDevice(out.data());
    wei_device_buf.ToDevice(wei.data());
    bias_device_buf.ToDevice(bias.data());

    // reset input to zero
    in_device_buf.SetZero();

    std::array<ck::index_t, NDimSpatial + 3> d0_g_n_c_wis_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> d0_g_n_c_wis_strides{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(bias_g_n_c_wis_desc.GetLengths(), d0_g_n_c_wis_lengths);
    copy(bias_g_n_c_wis_desc.GetStrides(), d0_g_n_c_wis_strides);

    using ck::utils::to_array;

    // do conv
    auto conv     = DeviceInstance{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(out_device_buf.GetDeviceBuffer(),
                                      wei_device_buf.GetDeviceBuffer(),
                                      to_array({bias_device_buf.GetDeviceBuffer()}),
                                      in_device_buf.GetDeviceBuffer(),
                                      to_array(out_g_n_k_wos_desc.GetLengths()),
                                      to_array(out_g_n_k_wos_desc.GetStrides()),
                                      to_array(wei_g_k_c_xs_desc.GetLengths()),
                                      to_array(wei_g_k_c_xs_desc.GetStrides()),
                                      to_array({d0_g_n_c_wis_lengths}),
                                      to_array({d0_g_n_c_wis_strides}),
                                      to_array(in_g_n_c_wis_desc.GetLengths()),
                                      to_array(in_g_n_c_wis_desc.GetStrides()),
                                      to_array(conv_param.conv_filter_strides_),
                                      to_array(conv_param.conv_filter_dilations_),
                                      to_array(conv_param.input_left_pads_),
                                      to_array(conv_param.input_right_pads_),
                                      out_element_op,
                                      wei_element_op,
                                      in_element_op);

    if(!conv.IsSupportedArgument(argument))
    {
        printf("wrong! device_conv with the specified compilation parameters does "
               "not support this Conv problem\n");

        return 1;
    }

    float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    if(do_verification)
    {
        using PassThrough = ck::tensor_operation::element_wise::PassThrough;

        // c doesn't physically exist, any layout is fine
        Tensor<float> c_host(in_g_n_c_wis_desc);

        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                         float,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         PassThrough,
                                                                         WeiElementOp,
                                                                         OutElementOp>();

        auto ref_invoker = ref_conv.MakeInvoker();

        auto ref_argument = ref_conv.MakeArgument(c_host,
                                                  wei,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  PassThrough{},
                                                  wei_element_op,
                                                  out_element_op);

        ref_invoker.Run(ref_argument);

        // TODO: implement elementwise operation for host
        in_host.ForEach(
            [&](auto&, auto idx) { in_element_op(in_host(idx), c_host(idx), bias(idx)); });

        in_device_buf.FromDevice(in_device.data());

        return ck::utils::check_err(in_device, in_host) ? 0 : 1;
    }

    return 0;
}
