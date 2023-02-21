// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {

template <AddressSpaceEnum AddressSpace,
          bool InvalidElementUseNumericalZeroValue,
          typename T,
          typename TensorDescTmp>
struct Tensor
{
    using TensorDescriptor = remove_cvref_t<TensorDescTmp>;
    using DataType         = remove_reference_t<T>;

    static constexpr AddressSpaceEnum kAdressSpace_ = AddressSpace;
    static constexpr bool kInvalidElementUseNumericalZeroValue_ =
        InvalidElementUseNumericalZeroValue;

    __host__ __device__ constexpr Tensor() : buf_{nullptr, 0}, desc_{} {}

    __host__ __device__ constexpr Tensor(DataType* p_data, TensorDescriptor desc)
        : buf_{p_data, desc.GetElementSpaceSize()}, desc_{desc}
    {
    }

    __host__ __device__ constexpr Tensor(DataType* p_data,
                                         TensorDescriptor desc,
                                         DataType invalid_element_value)
        : buf_{p_data, desc.GetElementSpaceSize(), invalid_element_value}, desc_{desc}
    {
    }

    // member
    DynamicBuffer<AddressSpace,
                  DataType,
                  typename TensorDescriptor::ElementSpaceSizeType,
                  InvalidElementUseNumericalZeroValue>
        buf_;

    TensorDescriptor desc_;
};

template <AddressSpaceEnum AddressSpace,
          bool InvalidElementUseNumericalZeroValue,
          typename T,
          typename TensorDesc>
__host__ __device__ constexpr auto make_tensor(const TensorDesc& desc, T* p_data)
{
    return Tensor<AddressSpace, InvalidElementUseNumericalZeroValue, T, remove_cvref_t<TensorDesc>>{
        p_data, desc};
}

template <typename OldTensor,
          typename NewTransforms,
          typename NewLowerDimensionOldVisibleIdss,
          typename NewUpperDimensionNewVisibleIdss>
__host__ __device__ constexpr auto transform_tensor(const OldTensor& old_tensor,
                                                    const NewTransforms& new_transforms,
                                                    NewLowerDimensionOldVisibleIdss,
                                                    NewUpperDimensionNewVisibleIdss)
{
    const auto new_desc = transform_tensor(old_tensor.desc_,
                                           new_transforms,
                                           NewLowerDimensionOldVisibleIdss{},
                                           NewUpperDimensionNewVisibleIdss{});

    return Tensor<OldTensor::kAddressSpace_,
                  OldTensor::kInvalidElementUseNumericalZeroValue,
                  typename OldTensor::DataType,
                  remove_cvref_t<decltype(new_desc)>>{old_tensor.buf_.p_data_, new_desc};
}

} // namespace ck
