//
//
#ifndef __stir_analytic_SRT2D_SRT2DReconstruction_H__
#define __stir_analytic_SRT2D_SRT2DReconstruction_H__
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup SRT2D
 
  \brief declares the stir::SRT2DReconstruction class

  \author Kris Thielemans
  \author PARAPET project
 
*/

#include "stir/recon_buildblock/AnalyticReconstruction.h"
//#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include <vector>
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;
class ProjData;

class SRT2DReconstruction :
        public
            RegisteredParsingObject<
                SRT2DReconstruction,
                    Reconstruction < DiscretisedDensity < 3,float> >,
                    AnalyticReconstruction
                 >
{
  //typedef AnalyticReconstruction base_type;
    typedef
    RegisteredParsingObject<
        SRT2DReconstruction,
            Reconstruction < DiscretisedDensity < 3,float> >,
            AnalyticReconstruction
         > base_type;
#ifdef SWIG
  // work-around swig problem. It gets confused when using a private (or protected)
  // typedef in a definition of a public typedef/member
 public:
#else
 private: 
#endif  
    typedef DiscretisedDensity < 3,float> TargetT;
public:
    //! Name which will be used when parsing a ProjectorByBinPair object
    static const char * const registered_name;

  //! Default constructor (calls set_defaults())
  SRT2DReconstruction (); 
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    SRT2DReconstruction(const std::string& parameter_filename);

 SRT2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, const float thres_restr_bound_v=-pow(10,6), const int num_segments_to_combine=-1, const float zoom=1, const int filter_wiener=1, const int filter_median=0, const int filter_gamma=1);

  virtual std::string method_info() const;

  virtual void ask_parameters();

  virtual Succeeded set_up(shared_ptr <TargetT > const& target_data_sptr);

 protected: // make parameters protected such that doc shows always up in doxygen
  // parameters used for parsing


  //! number of segments to combine (with SSRB) before starting 2D reconstruction
  /*! if -1, a value is chosen depending on the axial compression.
      If there is no axial compression, num_segments_to_combine is
      effectively set to 3, otherwise it is set to 1.
      \see SSRB
  */
  int num_segments_to_combine;
  //! potentially display data
  /*! allowed values: \c display_level=0 (no display), 1 (only final image), 
      2 (filtered-viewgrams). Defaults to 0.
   */
  int display_level;
  float zoom;
  int filter_wiener; 
  int filter_median; 
  int filter_gamma; 
	float thres_restr_bound;
	std::vector<double> thres_restr_bound_vector;  

  private:
  Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & target_image_ptr);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing(); 

 inline void spline(float x[],float y[],int n, float y2[]);

inline float hilbert_der(float x, float f[], float ddf[], float p[], int sp, float lg[], float termC);
inline float integ(float dist, int max, float ff[]);

void wiener(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa); 
void median(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa); 
void gamma(VoxelsOnCartesianGrid<float>& image, int sx, int sy, int sa); 
};


END_NAMESPACE_STIR
    
#endif