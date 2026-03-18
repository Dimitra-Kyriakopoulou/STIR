/*
    Copyright (C) 2026, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_cuvec_array_H__
#define __stir_cuvec_array_H__

/*!
  \file
  \ingroup CUDA
  \brief Helpers to let stir::Array optionally share ownership of CuVec-backed storage.

  These helpers do not change the underlying Array/VectorWithOffset storage model.
  Instead, they bridge a CuVec allocation into the existing Array constructor that
  already accepts shared ownership of a contiguous data block.
*/

#include "stir/Array.h"
#include "stir/error.h"
#include "stir/shared_ptr.h"

#ifdef STIR_WITH_CUDA
#  include "cuvec.cuh"
#  include <algorithm>
#  include <utility>

START_NAMESPACE_STIR

template <typename elemT>
inline shared_ptr<elemT[]>
make_shared_array_from_cuvec(const shared_ptr<CuVec<elemT>>& cuvec_sptr)
{
  if (!cuvec_sptr)
    error("make_shared_array_from_cuvec: null CuVec owner");

  return shared_ptr<elemT[]>(cuvec_sptr->data(), [owner = cuvec_sptr](elemT*) mutable {
    owner.reset();
  });
}

template <int num_dimensions, typename elemT>
inline Array<num_dimensions, elemT>
make_array_from_cuvec(const IndexRange<num_dimensions>& range, const shared_ptr<CuVec<elemT>>& cuvec_sptr)
{
  if (!cuvec_sptr)
    error("make_array_from_cuvec: null CuVec owner");
  if (cuvec_sptr->size() != range.size_all())
    error("make_array_from_cuvec: CuVec size does not match Array range");

  return Array<num_dimensions, elemT>(range, make_shared_array_from_cuvec(cuvec_sptr));
}

template <int num_dimensions, typename elemT>
inline Array<num_dimensions, elemT>
make_array_from_cuvec(const IndexRange<num_dimensions>& range, CuVec<elemT>&& cuvec)
{
  return make_array_from_cuvec<num_dimensions, elemT>(range, MAKE_SHARED<CuVec<elemT>>(std::move(cuvec)));
}

template <int num_dimensions, typename elemT>
inline Array<num_dimensions, elemT>
make_cuvec_array(const IndexRange<num_dimensions>& range)
{
  auto cuvec_sptr = MAKE_SHARED<CuVec<elemT>>(range.size_all());
  std::fill(cuvec_sptr->begin(), cuvec_sptr->end(), elemT(0));
  return make_array_from_cuvec<num_dimensions, elemT>(range, cuvec_sptr);
}

END_NAMESPACE_STIR
#endif

#endif
