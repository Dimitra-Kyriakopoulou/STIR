//
//
/*
    Copyright (C) 2000- 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2016, University of Hull
    Copyright (C) 2017, 2018, University College London
    Copyright (C) 2018, University of Leeds
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Implementation of non-inline functions of class
  stir::ProjDataInfoCylindricalNoArcCorr

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Palak Wadhwa

*/

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/LORCoordinates.h"
#include "stir/round.h"
#include <algorithm>
#include "stir/error.h"
#include <sstream>

#include <boost/static_assert.hpp>

using std::endl;
using std::ends;
using std::string;
using std::pair;
using std::vector;

START_NAMESPACE_STIR
ProjDataInfoCylindricalNoArcCorr::ProjDataInfoCylindricalNoArcCorr()
{}

ProjDataInfoCylindricalNoArcCorr::ProjDataInfoCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_sptr,
                                                                   const float ring_radius_v,
                                                                   const float angular_increment_v,
                                                                   const VectorWithOffset<int>& num_axial_pos_per_segment,
                                                                   const VectorWithOffset<int>& min_ring_diff_v,
                                                                   const VectorWithOffset<int>& max_ring_diff_v,
                                                                   const int num_views,
                                                                   const int num_tangential_poss,
                                                                   const int tof_mash_factor)
    : ProjDataInfoCylindrical(
        scanner_sptr, num_axial_pos_per_segment, min_ring_diff_v, max_ring_diff_v, num_views, num_tangential_poss),
      ring_radius(ring_radius_v),
      angular_increment(angular_increment_v)
{
  if (!scanner_sptr)
    error("ProjDataInfoCylindricalNoArcCorr: first argument (scanner_ptr) is zero");
  if (num_tangential_poss > scanner_sptr->get_max_num_non_arccorrected_bins())
    error("ProjDataInfoCylindricalNoArcCorr: number of tangential positions exceeds the maximum number of non arc-corrected bins "
          "set for the scanner.");

  uncompressed_view_tangpos_to_det1det2_initialised = false;
  det1det2_to_uncompressed_view_tangpos_initialised = false;
  if (scanner_sptr->is_tof_ready())
    set_tof_mash_factor(tof_mash_factor);
#ifdef STIR_OPENMP_SAFE_BUT_SLOW
  this->initialise_uncompressed_view_tangpos_to_det1det2();
  this->initialise_det1det2_to_uncompressed_view_tangpos();
#endif
}

ProjDataInfoCylindricalNoArcCorr::ProjDataInfoCylindricalNoArcCorr(const shared_ptr<Scanner> scanner_sptr,
                                                                   const VectorWithOffset<int>& num_axial_pos_per_segment,
                                                                   const VectorWithOffset<int>& min_ring_diff_v,
                                                                   const VectorWithOffset<int>& max_ring_diff_v,
                                                                   const int num_views,
                                                                   const int num_tangential_poss,
                                                                   const int tof_mash_factor)
    : ProjDataInfoCylindricalNoArcCorr(scanner_sptr,
                                       scanner_sptr ? scanner_sptr->get_effective_ring_radius()
                                                    : 0.F, // avoid segfault if scanner_sptr==0
                                       scanner_sptr ? static_cast<float>(_PI / scanner_sptr->get_num_detectors_per_ring()) : 0.F,
                                       num_axial_pos_per_segment,
                                       min_ring_diff_v,
                                       max_ring_diff_v,
                                       num_views,
                                       num_tangential_poss,
                                       tof_mash_factor)
{}

ProjDataInfo*
ProjDataInfoCylindricalNoArcCorr::clone() const
{
  return static_cast<ProjDataInfo*>(new ProjDataInfoCylindricalNoArcCorr(*this));
}

bool
ProjDataInfoCylindricalNoArcCorr::operator==(const self_type& that) const
{
  if (!base_type::blindly_equals(&that))
    return false;
  return fabs(this->ring_radius - that.ring_radius) < 0.05F && fabs(this->angular_increment - that.angular_increment) < 0.05F;
}

bool
ProjDataInfoCylindricalNoArcCorr::blindly_equals(const root_type* const that_ptr) const
{
  assert(dynamic_cast<const self_type* const>(that_ptr) != 0);
  return this->operator==(static_cast<const self_type&>(*that_ptr));
}

string
ProjDataInfoCylindricalNoArcCorr::parameter_info() const
{

  std::ostringstream s;
  s << "ProjDataInfoCylindricalNoArcCorr := \n";
  s << ProjDataInfoCylindrical::parameter_info();
  s << "End :=\n";
  return s.str();
}

float
ProjDataInfoCylindricalNoArcCorr::get_psi_offset() const
{
  return this->get_scanner_ptr()->get_intrinsic_azimuthal_tilt();
}

/*
   Warning:
   this code makes use of an implementation dependent feature:
   bit shifting negative ints to the right.
    -1 >> 1 should be -1
    -2 >> 1 should be -1
   This is ok on SUNs (gcc, but probably SUNs cc as well), Parsytec (gcc),
   Pentium (gcc, VC++) and probably every other system which uses
   the 2-complement convention.

   Update: compile time assert is implemented.
*/

/*!
  Go from sinograms to detectors.

  Because sinograms are not arc-corrected, tang_pos_num corresponds
  to an angle as well. Before interleaving we have that
  \verbatim
  det_angle_1 = LOR_angle + bin_angle
  det_angle_2 = LOR_angle + (Pi - bin_angle)
  \endverbatim
  (Hint: understand this first at LOR_angle=0, then realise that
  other LOR_angles follow just by rotation)

  Code gets slightly intricate because:
  - angles have to be defined modulo 2 Pi (so num_detectors)
  - interleaving
*/
void
ProjDataInfoCylindricalNoArcCorr::initialise_uncompressed_view_tangpos_to_det1det2() const
{
  BOOST_STATIC_ASSERT(-1 >> 1 == -1);
  BOOST_STATIC_ASSERT(-2 >> 1 == -1);

  const int num_detectors = get_scanner_ptr()->get_num_detectors_per_ring();

  assert(num_detectors % 2 == 0);
#ifndef NDEBUG
  // check views range from 0 to Pi
  // PW Supports intrinsic tilt.
  const float v_offset = get_azimuthal_angle_offset();
  assert(fabs(get_phi(Bin(0, 0, 0, 0)) - v_offset) < 1.E-4);
  assert(fabs(get_phi(Bin(0, get_num_views(), 0, 0)) - v_offset - _PI) < 1.E-4);
#endif
  const int min_tang_pos_num = -(num_detectors / 2) + 1;
  const int max_tang_pos_num = -(num_detectors / 2) + num_detectors;

  if (this->get_min_tangential_pos_num() < min_tang_pos_num || this->get_max_tangential_pos_num() > max_tang_pos_num)
    {
      error("The tangential_pos range (%d to %d) for this projection data is too large.\n"
            "Maximum supported range is from %d to %d",
            this->get_min_tangential_pos_num(),
            this->get_max_tangential_pos_num(),
            min_tang_pos_num,
            max_tang_pos_num);
    }

  uncompressed_view_tangpos_to_det1det2.grow(0, num_detectors / 2 - 1);
  for (int v_num = 0; v_num <= num_detectors / 2 - 1; ++v_num)
    {
      uncompressed_view_tangpos_to_det1det2[v_num].grow(min_tang_pos_num, max_tang_pos_num);

      for (int tp_num = min_tang_pos_num; tp_num <= max_tang_pos_num; ++tp_num)
        {
          /*
             adapted from CTI code
             Note for implementation: avoid using % with negative numbers
             so add num_detectors before doing modulo num_detectors)
            */
          uncompressed_view_tangpos_to_det1det2[v_num][tp_num].det1_num = (v_num + (tp_num >> 1) + num_detectors) % num_detectors;
          uncompressed_view_tangpos_to_det1det2[v_num][tp_num].det2_num
              = (v_num - ((tp_num + 1) >> 1) + num_detectors / 2) % num_detectors;
        }
    }
    // thanks to yohjp:
    // http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) && _OPENMP >= 201012
#  pragma omp atomic write
#endif
  uncompressed_view_tangpos_to_det1det2_initialised = true;
}

void
ProjDataInfoCylindricalNoArcCorr::initialise_det1det2_to_uncompressed_view_tangpos() const
{
  BOOST_STATIC_ASSERT(-1 >> 1 == -1);
  BOOST_STATIC_ASSERT(-2 >> 1 == -1);

  const int num_detectors = get_scanner_ptr()->get_num_detectors_per_ring();

  if (num_detectors % 2 != 0)
    {
      error("Number of detectors per ring should be even but is %d", num_detectors);
    }
  if (this->get_min_view_num() != 0)
    {
      error("Minimum view number should currently be zero to be able to use get_view_tangential_pos_num_for_det_num_pair()");
    }
#ifndef NDEBUG
  // check views range from 0 to Pi
  // PW Supports intrinsic tilt.
  const float v_offset = get_azimuthal_angle_offset();
  assert(fabs(get_phi(Bin(0, 0, 0, 0)) - v_offset) < 1.E-4);
  assert(fabs(get_phi(Bin(0, get_max_view_num() + 1, 0, 0)) - v_offset - _PI) < 1.E-4);
#endif
  // const int min_tang_pos_num = -(num_detectors/2);
  // const int max_tang_pos_num = -(num_detectors/2)+num_detectors;
  const int max_num_views = num_detectors / 2;

  det1det2_to_uncompressed_view_tangpos.grow(0, num_detectors - 1);
  for (int det1_num = 0; det1_num < num_detectors; ++det1_num)
    {
      det1det2_to_uncompressed_view_tangpos[det1_num].grow(0, num_detectors - 1);

      for (int det2_num = 0; det2_num < num_detectors; ++det2_num)
        {
          if (det1_num == det2_num)
            continue;
          /*
           This somewhat obscure formula was obtained by inverting the code for
           get_det_num_pair_for_view_tangential_pos_num()
           This can be simplified (especially all the branching later on), but
           as we execute this code only occasionally, it's probably not worth it.
          */
          int swap_detectors;
          /*
          Note for implementation: avoid using % with negative numbers
          so add num_detectors before doing modulo num_detectors
          */
          int tang_pos_num = (det1_num - det2_num + 3 * num_detectors / 2) % num_detectors;
          int view_num = (det1_num - (tang_pos_num >> 1) + num_detectors) % num_detectors;

          /* Now adjust ranges for view_num, tang_pos_num.
          The next lines go only wrong in the singular (and irrelevant) case
          det_num1 == det_num2 (when tang_pos_num == num_detectors - tang_pos_num)

            We use the combinations of the following 'symmetries' of
            (tang_pos_num, view_num) == (tang_pos_num+2*num_views, view_num + num_views)
            == (-tang_pos_num, view_num + num_views)
            Using the latter interchanges det_num1 and det_num2, and this leaves
            the LOR the same in the 2D case. However, in 3D this interchanges the rings
            as well. So, we keep track of this in swap_detectors, and return its final
            value.
          */
          if (view_num < max_num_views)
            {
              if (tang_pos_num >= max_num_views)
                {
                  tang_pos_num = num_detectors - tang_pos_num;
                  swap_detectors = 1;
                }
              else
                {
                  swap_detectors = 0;
                }
            }
          else
            {
              view_num -= max_num_views;
              if (tang_pos_num >= max_num_views)
                {
                  tang_pos_num -= num_detectors;
                  swap_detectors = 0;
                }
              else
                {
                  tang_pos_num *= -1;
                  swap_detectors = 1;
                }
            }

          det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].view_num = view_num;
          det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].tang_pos_num = tang_pos_num;
          det1det2_to_uncompressed_view_tangpos[det1_num][det2_num].swap_detectors = swap_detectors == 0;
        }
    }
    // thanks to yohjp:
    // http://stackoverflow.com/questions/27975737/how-to-handle-cached-data-structures-with-multi-threading-e-g-openmp
#if defined(STIR_OPENMP) && _OPENMP >= 201012
#  pragma omp atomic write
#endif
  det1det2_to_uncompressed_view_tangpos_initialised = true;
}

unsigned int
ProjDataInfoCylindricalNoArcCorr::get_num_det_pos_pairs_for_bin(const Bin& bin, bool ignore_non_spatial_dimensions) const
{
  return get_num_ring_pairs_for_segment_axial_pos_num(bin.segment_num(), bin.axial_pos_num()) * get_view_mashing_factor()
         * (ignore_non_spatial_dimensions ? 1 : std::max(1, get_tof_mash_factor()));
}

void
ProjDataInfoCylindricalNoArcCorr::get_all_det_pos_pairs_for_bin(vector<DetectionPositionPair<>>& dps,
                                                                const Bin& bin,
                                                                bool ignore_non_spatial_dimensions) const
{
  this->initialise_uncompressed_view_tangpos_to_det1det2_if_not_done_yet();

  dps.resize(get_num_det_pos_pairs_for_bin(bin, ignore_non_spatial_dimensions));

  const ProjDataInfoCylindrical::RingNumPairs& ring_pairs
      = get_all_ring_pairs_for_segment_axial_pos_num(bin.segment_num(), bin.axial_pos_num());
  // not sure how to handle mashing with non-zero view offset...
  assert(get_min_view_num() == 0);

  int min_timing_pos_num = 0;
  int max_timing_pos_num = 0;
  if (!ignore_non_spatial_dimensions)
    {
      // not sure how to handle even tof mashing
      assert(!is_tof_data() || (get_tof_mash_factor() % 2 == 1)); // TODOTOF
      // we will need to add all (unmashed) timing_pos for the current bin
      min_timing_pos_num = bin.timing_pos_num() * get_tof_mash_factor() - (get_tof_mash_factor() / 2);
      max_timing_pos_num = bin.timing_pos_num() * get_tof_mash_factor() + (get_tof_mash_factor() / 2);
    }

  unsigned int current_dp_num = 0;
  for (int uncompressed_view_num = bin.view_num() * get_view_mashing_factor();
       uncompressed_view_num < (bin.view_num() + 1) * get_view_mashing_factor();
       ++uncompressed_view_num)
    {
      const int det1_num = uncompressed_view_tangpos_to_det1det2[uncompressed_view_num][bin.tangential_pos_num()].det1_num;
      const int det2_num = uncompressed_view_tangpos_to_det1det2[uncompressed_view_num][bin.tangential_pos_num()].det2_num;
      for (auto rings_iter = ring_pairs.begin(); rings_iter != ring_pairs.end(); ++rings_iter)
        {
          for (int uncompressed_timing_pos_num = min_timing_pos_num; uncompressed_timing_pos_num <= max_timing_pos_num;
               ++uncompressed_timing_pos_num)
            {
              assert(current_dp_num < get_num_det_pos_pairs_for_bin(bin, ignore_non_spatial_dimensions));
              dps[current_dp_num].pos1().tangential_coord() = det1_num;
              dps[current_dp_num].pos1().axial_coord() = rings_iter->first;
              dps[current_dp_num].pos2().tangential_coord() = det2_num;
              dps[current_dp_num].pos2().axial_coord() = rings_iter->second;
              dps[current_dp_num].timing_pos() = uncompressed_timing_pos_num;
              ++current_dp_num;
            }
        }
    }
  assert(current_dp_num == get_num_det_pos_pairs_for_bin(bin, ignore_non_spatial_dimensions));
}

Succeeded
ProjDataInfoCylindricalNoArcCorr::find_scanner_coordinates_given_cartesian_coordinates(
    int& det1, int& det2, int& ring1, int& ring2, const CartesianCoordinate3D<float>& c1, const CartesianCoordinate3D<float>& c2)
    const
{
  const int num_detectors = get_scanner_ptr()->get_num_detectors_per_ring();
  const float ring_spacing = get_scanner_ptr()->get_ring_spacing();
  const float ring_radius = get_scanner_ptr()->get_effective_ring_radius();

#if 0
  const CartesianCoordinate3D<float> d = c2 - c1;
  /* parametrisation of LOR is 
     c = l*d+c1
     l has to be such that c.x^2 + c.y^2 = R^2
     i.e.
     (l*d.x+c1.x)^2+(l*d.y+c1.y)^2==R^2
     l^2*(d.x^2+d.y^2) + 2*l*(d.x*c1.x + d.y*c1.y) + c1.x^2+c2.y^2-R^2==0
     write as a*l^2+2*b*l+e==0
     l = (-b +- sqrt(b^2-a*e))/a
     argument of sqrt simplifies to
     R^2*(d.x^2+d.y^2)-(d.x*c1.y-d.y*c1.x)^2
  */
  const float dxy2 = (square(d.x())+square(d.y()));
  assert(dxy2>0); // otherwise parallel to z-axis, which is gives ill-defined bin-coordinates
  const float argsqrt=
    (square(ring_radius)*dxy2-square(d.x()*c1.y()-d.y()*c1.x()));
  if (argsqrt<=0)
    return Succeeded::no; // LOR is outside detector radius
  const float root = sqrt(argsqrt);

  const float l1 = (- (d.x()*c1.x() + d.y()*c1.y())+root)/dxy2;
  const float l2 = (- (d.x()*c1.x() + d.y()*c1.y())-root)/dxy2;
  const CartesianCoordinate3D<float> coord_det1 = d*l1 + c1;
  const CartesianCoordinate3D<float> coord_det2 = d*l2 + c1;
  assert(fabs(square(coord_det1.x())+square(coord_det1.y())-square(ring_radius))<square(ring_radius)*10.E-5);
  assert(fabs(square(coord_det2.x())+square(coord_det2.y())-square(ring_radius))<square(ring_radius)*10.E-5);

  det1 = modulo(round(atan2(coord_det1.x(),-coord_det1.y())/(2.*_PI/num_detectors)), num_detectors);
  det2 = modulo(round(atan2(coord_det2.x(),-coord_det2.y())/(2.*_PI/num_detectors)), num_detectors);
  ring1 = round(coord_det1.z()/ring_spacing);
  ring2 = round(coord_det2.z()/ring_spacing);
#else
  LORInCylinderCoordinates<float> cyl_coords;
  if (find_LOR_intersections_with_cylinder(cyl_coords, LORAs2Points<float>(c1, c2), ring_radius) == Succeeded::no)
    return Succeeded::no;

  det1 = modulo(round((cyl_coords.p1().psi() - this->get_psi_offset()) / (2. * _PI / num_detectors)), num_detectors);
  det2 = modulo(round((cyl_coords.p2().psi() - this->get_psi_offset()) / (2. * _PI / num_detectors)), num_detectors);
  ring1 = round(cyl_coords.p1().z() / ring_spacing);
  ring2 = round(cyl_coords.p2().z() / ring_spacing);

#endif

  assert(det1 >= 0 && det1 < get_scanner_ptr()->get_num_detectors_per_ring());
  assert(det2 >= 0 && det2 < get_scanner_ptr()->get_num_detectors_per_ring());

  return (ring1 >= 0 && ring1 < get_scanner_ptr()->get_num_rings() && ring2 >= 0 && ring2 < get_scanner_ptr()->get_num_rings())
             ? Succeeded::yes
             : Succeeded::no;
}

void
ProjDataInfoCylindricalNoArcCorr::find_cartesian_coordinates_of_detection(CartesianCoordinate3D<float>& coord_1,
                                                                          CartesianCoordinate3D<float>& coord_2,
                                                                          const Bin& bin) const
{
  // find detectors
  DetectionPositionPair<> dpp;
  get_det_pos_pair_for_bin(dpp, bin);

  /* TODO
   best to use Scanner::get_coordinate_for_det_pos().
   Sadly, the latter is not yet implemented for Cylindrical scanners.
  */
  // find corresponding cartesian coordinates
  find_cartesian_coordinates_given_scanner_coordinates(coord_1,
                                                       coord_2,
                                                       dpp.pos1().axial_coord(),
                                                       dpp.pos2().axial_coord(),
                                                       dpp.pos1().tangential_coord(),
                                                       dpp.pos2().tangential_coord(),
                                                       dpp.timing_pos());
}

void
ProjDataInfoCylindricalNoArcCorr::find_cartesian_coordinates_given_scanner_coordinates(CartesianCoordinate3D<float>& coord_1,
                                                                                       CartesianCoordinate3D<float>& coord_2,
                                                                                       const int Ring_A,
                                                                                       const int Ring_B,
                                                                                       const int det1,
                                                                                       const int det2,
                                                                                       const int timing_pos_num) const
{
  const int num_detectors_per_ring = get_scanner_ptr()->get_num_detectors_per_ring();

  int d1, d2, r1, r2;
  int tpos = timing_pos_num;

  this->initialise_det1det2_to_uncompressed_view_tangpos_if_not_done_yet();

  if (!det1det2_to_uncompressed_view_tangpos[det1][det2].swap_detectors)
    {
      d1 = det2;
      d2 = det1;
      r1 = Ring_B;
      r2 = Ring_A;
      tpos *= -1;
    }
  else
    {
      d1 = det1;
      d2 = det2;
      r1 = Ring_A;
      r2 = Ring_B;
    }

#if 0
  const float df1 = (2.*_PI/num_detectors_per_ring)*(det1);
  const float df2 = (2.*_PI/num_detectors_per_ring)*(det2);
  const float x1 = get_scanner_ptr()->get_effective_ring_radius()*cos(df1);
  const float y1 = get_scanner_ptr()->get_effective_ring_radius()*sin(df1);
  const float x2 = get_scanner_ptr()->get_effective_ring_radius()*cos(df2);
  const float y2 = get_scanner_ptr()->get_effective_ring_radius()*sin(df2);
  const float z1 = Ring_A*get_scanner_ptr()->get_ring_spacing();
  const float z2 = Ring_B*get_scanner_ptr()->get_ring_spacing();
  // make sure the return values are in STIR coordinates
  coord_1.z() = z1;
  coord_1.y() = -x1;
  coord_1.x() = y1;

  coord_2.z() = z2;
  coord_2.y() = -x2;
  coord_2.x() = y2;
#else
  LORInCylinderCoordinates<float> cyl_coords(get_scanner_ptr()->get_effective_ring_radius());
  cyl_coords.p1().psi() = to_0_2pi(static_cast<float>((2. * _PI / num_detectors_per_ring) * (d1)) + this->get_psi_offset());
  cyl_coords.p2().psi() = to_0_2pi(static_cast<float>((2. * _PI / num_detectors_per_ring) * (d2)) + this->get_psi_offset());
  cyl_coords.p1().z() = r1 * get_scanner_ptr()->get_ring_spacing();
  cyl_coords.p2().z() = r2 * get_scanner_ptr()->get_ring_spacing();
  LORAs2Points<float> lor(cyl_coords);
  coord_1 = lor.p1();
  coord_2 = lor.p2();

#endif
  if (tpos < 0)
    std::swap(coord_1, coord_2);
}

void
ProjDataInfoCylindricalNoArcCorr::find_bin_given_cartesian_coordinates_of_detection(
    Bin& bin, const CartesianCoordinate3D<float>& coord_1, const CartesianCoordinate3D<float>& coord_2) const
{
  int det_num_a;
  int det_num_b;
  int ring_a;
  int ring_b;

  // given two CartesianCoordinates find the intersection
  if (find_scanner_coordinates_given_cartesian_coordinates(det_num_a, det_num_b, ring_a, ring_b, coord_1, coord_2)
      == Succeeded::no)
    {
      bin.set_bin_value(-1);
      return;
    }

  // check rings are in valid range
  // this should have been done by find_scanner_coordinates_given_cartesian_coordinates
  assert(!(ring_a < 0 || ring_a >= get_scanner_ptr()->get_num_rings() || ring_b < 0
           || ring_b >= get_scanner_ptr()->get_num_rings()));

  if (get_bin_for_det_pair(bin, det_num_a, ring_a, det_num_b, ring_b) == Succeeded::no
      || bin.tangential_pos_num() < get_min_tangential_pos_num() || bin.tangential_pos_num() > get_max_tangential_pos_num())
    bin.set_bin_value(-1);
}

Bin
ProjDataInfoCylindricalNoArcCorr::get_bin(const LOR<float>& lor, const double delta_time) const
{
  Bin bin;
#ifndef STIR_DEVEL
  // find nearest bin by going to nearest detectors first
  LORInCylinderCoordinates<float> cyl_coords;
  if (lor.change_representation(cyl_coords, get_ring_radius()) == Succeeded::no)
    {
      bin.set_bin_value(-1);
      return bin;
    }
  const int num_detectors_per_ring = get_scanner_ptr()->get_num_detectors_per_ring();
  const int num_rings = get_scanner_ptr()->get_num_rings();

  const int det1 = modulo(round((cyl_coords.p1().psi() - this->get_psi_offset()) / (2. * _PI / num_detectors_per_ring)),
                          num_detectors_per_ring);
  const int det2 = modulo(round((cyl_coords.p2().psi() - this->get_psi_offset()) / (2. * _PI / num_detectors_per_ring)),
                          num_detectors_per_ring);
  // TODO WARNING LOR coordinates are w.r.t. centre of scanner, but the rings are numbered with the first ring at 0
  const int ring1 = round(cyl_coords.p1().z() / get_ring_spacing() + (num_rings - 1) / 2.F);
  const int ring2 = round(cyl_coords.p2().z() / get_ring_spacing() + (num_rings - 1) / 2.F);

  assert(det1 >= 0 && det1 < num_detectors_per_ring);
  assert(det2 >= 0 && det2 < num_detectors_per_ring);

  if (ring1 >= 0 && ring1 < num_rings && ring2 >= 0 && ring2 < num_rings
      && get_bin_for_det_pair(bin, det1, ring1, det2, ring2, (cyl_coords.is_swapped() ? -1 : 1) * get_tof_bin(delta_time))
             == Succeeded::yes
      && bin.tangential_pos_num() >= get_min_tangential_pos_num() && bin.tangential_pos_num() <= get_max_tangential_pos_num())
    {
      bin.set_bin_value(1);
      return bin;
    }
  else
    {
      bin.set_bin_value(-1);
      return bin;
    }

#else
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor_coords;
  if (lor.change_representation(lor_coords, get_ring_radius()) == Succeeded::no)
    {
      bin.set_bin_value(-1);
      return bin;
    }

  // first find view
  // unfortunately, phi ranges from [0,Pi[, but the rounding can
  // map this to a view which corresponds to Pi anyway.
  // PW Accurate bin view number = phi - intrinsic_tilt.
  bin.view_num() = round(to_0_2pi(lor_coords.phi() - get_azimuthal_angle_offset()) / get_azimuthal_angle_sampling());
  assert(bin.view_num() >= 0);
  assert(bin.view_num() <= get_num_views());
  const bool swap_direction = bin.view_num() > get_max_view_num();
  if (swap_direction)
    bin.view_num() -= get_num_views();

  bin.tangential_pos_num() = round(lor_coords.beta() / angular_increment);
  if (swap_direction)
    bin.tangential_pos_num() *= -1;

  if (bin.tangential_pos_num() < get_min_tangential_pos_num() || bin.tangential_pos_num() > get_max_tangential_pos_num())
    {
      bin.set_bin_value(-1);
      return bin;
    }

#  if 0
  const int num_rings = 
    get_scanner_ptr()->get_num_rings();
  // TODO WARNING LOR coordinates are w.r.t. centre of scanner, but the rings are numbered with the first ring at 0
  int ring1, ring2;
  if (!swap_direction)
    {
      ring1 = round(lor_coords.z1()/get_ring_spacing() + (num_rings-1)/2.F);
      ring2 = round(lor_coords.z2()/get_ring_spacing() + (num_rings-1)/2.F);
    }
  else
    {
      ring2 = round(lor_coords.z1()/get_ring_spacing() + (num_rings-1)/2.F);
      ring1 = round(lor_coords.z2()/get_ring_spacing() + (num_rings-1)/2.F);
    }

  if (!(ring1 >=0 && ring1<get_scanner_ptr()->get_num_rings() &&
	ring2 >=0 && ring2<get_scanner_ptr()->get_num_rings() &&
	get_segment_axial_pos_num_for_ring_pair(bin.segment_num(),
						bin.axial_pos_num(),
						ring1,
						ring2) == Succeeded::yes)
      )
    {
      bin.set_bin_value(-1);
      return bin;
    }
#  else
  // find nearest segment
  {
    if (delta_time != 0)
      {
        error("TODO TOF");
      }
    const float delta
        = (swap_direction ? lor_coords.z1() - lor_coords.z2() : lor_coords.z2() - lor_coords.z1()) / get_ring_spacing();
    // check if out of acquired range
    // note the +1 or -1, which takes the size of the rings into account
    if (delta > get_max_ring_difference(get_max_segment_num()) + 1 || delta < get_min_ring_difference(get_min_segment_num()) - 1)
      {
        bin.set_bin_value(-1);
        return bin;
      }
    if (delta >= 0)
      {
        for (bin.segment_num() = 0; bin.segment_num() < get_max_segment_num(); ++bin.segment_num())
          {
            if (delta < get_max_ring_difference(bin.segment_num()) + .5)
              break;
          }
      }
    else
      {
        // delta<0
        for (bin.segment_num() = 0; bin.segment_num() > get_min_segment_num(); --bin.segment_num())
          {
            if (delta > get_min_ring_difference(bin.segment_num()) - .5)
              break;
          }
      }
  }
  // now find nearest axial position
  {
    const float m = (lor_coords.z2() + lor_coords.z1()) / 2;
#    if 0
    // this uses private member of ProjDataInfoCylindrical
    // enable when moved
    initialise_ring_diff_arrays_if_not_done_yet();

#      ifndef NDEBUG
    bin.axial_pos_num()=0;
    assert(get_m(bin)==- m_offset[bin.segment_num()]);
#      endif
    bin.axial_pos_num() =
      round((m + m_offset[bin.segment_num()])/
	    get_axial_sampling(bin.segment_num()));
#    else
    bin.axial_pos_num() = 0;
    bin.axial_pos_num() = round((m - get_m(bin)) / get_axial_sampling(bin.segment_num()));
#    endif
    if (bin.axial_pos_num() < get_min_axial_pos_num(bin.segment_num())
        || bin.axial_pos_num() > get_max_axial_pos_num(bin.segment_num()))
      {
        bin.set_bin_value(-1);
        return bin;
      }
  }
#  endif

  bin.set_bin_value(1);
  return bin;
#endif
}

END_NAMESPACE_STIR
