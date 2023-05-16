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

template <typename Object>
struct sgpr_ptr
{
    static_assert(!std::is_const_v<Object> && !std::is_reference_v<Object> &&
                  std::is_trivially_copyable_v<Object>);

    static constexpr std::size_t SgprSize   = 4;
    static constexpr std::size_t ObjectSize = sizeof(Object);

    using Sgpr = get_signed_int_t<SgprSize>;

    __device__ explicit sgpr_ptr(const Object& obj) noexcept
    {
        const auto* from = reinterpret_cast<const unsigned char*>(&obj);
        static_for<0, ObjectSize, SgprSize>{}([&](auto offset) {
            *reinterpret_cast<Sgpr*>(memory + offset) =
                __builtin_amdgcn_readfirstlane(*reinterpret_cast<const Sgpr*>(from + offset));
        });

        constexpr std::size_t RemainedSize = ObjectSize % SgprSize;
        if constexpr(0 < RemainedSize)
        {
            using Carrier = get_signed_int_t<RemainedSize>;

            constexpr std::size_t offset =
                SgprSize * math::integer_divide_floor(ObjectSize, SgprSize);

            *reinterpret_cast<Carrier>(memory + offset) =
                __builtin_amdgcn_readfirstlane(*reinterpret_cast<const Carrier*>(from + offset));
        }
    }

    __device__ Object& operator*() { return *(this->operator->()); }

    __device__ const Object& operator*() const { return *(this->operator->()); }

    __device__ Object* operator->() { return reinterpret_cast<Object*>(memory); }

    __device__ const Object* operator->() const { return reinterpret_cast<const Object*>(memory); }

    private:
    alignas(
        Object) unsigned char memory[SgprSize * math::integer_divide_ceil(ObjectSize, SgprSize)];
};
} // namespace detail

template <typename T>
__device__ constexpr auto readfirstlane(const T& obj)
{
    return detail::sgpr_ptr<T>(obj);
}

} // namespace ck
