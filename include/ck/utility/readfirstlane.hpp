// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/functional2.hpp"
#include "ck/utility/math.hpp"

#include <cstdint>
#include <type_traits>

namespace ck {
namespace detail {

template <std::size_t Size>
struct get_signed_int;

template <>
struct get_signed_int<1>
{
    using type = std::int8_t;
};

template <>
struct get_signed_int<2>
{
    using type = std::int16_t;
};

template <>
struct get_signed_int<4>
{
    using type = std::int32_t;
};

template <std::size_t Size>
using get_signed_int_t = typename get_signed_int<Size>::type;

} // namespace detail

__device__ std::int32_t readfirstlane(std::int32_t value)
{
    return __builtin_amdgcn_readfirstlane(value);
}

template <
    typename Object,
    typename = std::enable_if_t<std::is_class_v<Object> && std::is_trivially_copyable_v<Object>>>
__device__ auto readfirstlane(const Object& obj)
{
    constexpr std::size_t SgprSize   = 4;
    constexpr std::size_t ObjectSize = sizeof(Object);

    using Sgpr = detail::get_signed_int_t<SgprSize>;

    alignas(Object) unsigned char memory[ObjectSize];

    const auto* from = reinterpret_cast<const unsigned char*>(&obj);
    static_for<0, ObjectSize, SgprSize>{}([&](auto offset) {
        *reinterpret_cast<Sgpr*>(memory + offset) =
            readfirstlane(*reinterpret_cast<const Sgpr*>(from + offset));
    });

    constexpr std::size_t RemainedSize = ObjectSize % SgprSize;
    if constexpr(0 < RemainedSize)
    {
        using Carrier = detail::get_signed_int_t<RemainedSize>;

        constexpr std::size_t offset = SgprSize * math::integer_divide_floor(ObjectSize, SgprSize);

        *reinterpret_cast<Carrier>(memory + offset) =
            readfirstlane(*reinterpret_cast<const Carrier*>(from + offset));
    }

    return *reinterpret_cast<Object*>(memory);
}

} // namespace ck
