// clang-format off
// @HEADER
// *****************************************************************************
//                            Tacho package
//
// Copyright 2022 NTESS and the Tacho contributors.
// SPDX-License-Identifier: BSD-2-Clause
// *****************************************************************************
// @HEADER
// clang-format on
#ifndef __TACHO_DRIVER_HPP__
#define __TACHO_DRIVER_HPP__

/// \file Tacho_Driver.hpp
/// \brief temporary solver interface for refactoring
/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "Tacho.hpp"
#include "Tacho_Util.hpp"

#include "mwm2.hpp"
#if defined(TACHO_HAVE_SUPERLUDIST)
// FIX this
extern  "C"
{
  void mc64id_dist(int *);

  void mc64ad_dist(int*, int*, int*, int*, int*,  double*,
                   int*, int*, int*, int*, int*,  double*,
                   int*, int*);
}
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_Timer.hpp>

#include <trilinos_btf_decl.h>

namespace Tacho {

/// forward decl
class Graph;
#if defined(TACHO_HAVE_METIS) || defined(TACHO_HAVE_TRILINOS_SS)
class GraphTools_Metis;
#else
class GraphTools;
#endif

class SymbolicTools;
template <typename ValueType, typename DeviceType> class CrsMatrixBase;
template <typename ValueType, typename DeviceType> class NumericToolsBase;
template <typename ValueType, typename DeviceType> class NumericToolsSerial;
template <typename ValueType, typename DeviceType, int Var> class NumericToolsLevelSet;

///
/// Tacho Solver interface
///
template <typename ValueType, typename DeviceType> struct Driver {
public:
  using value_type = ValueType;
  using mag_type = typename ArithTraits<ValueType>::mag_type;
  using device_type = DeviceType;
  using exec_space = typename device_type::execution_space;
  using exec_memory_space = typename device_type::memory_space;

  using host_device_type = typename UseThisDevice<Kokkos::DefaultHostExecutionSpace>::type;
  using host_space = typename host_device_type::execution_space;
  using host_memory_space = typename host_device_type::memory_space;

  using size_type_array = Kokkos::View<size_type *, device_type>;
  using ordinal_type_array = Kokkos::View<ordinal_type *, device_type>;
  using value_type_array = Kokkos::View<value_type *, device_type>;
  using value_type_matrix = Kokkos::View<value_type **, Kokkos::LayoutLeft, device_type>;

  using size_type_array_host = Kokkos::View<size_type *, host_device_type>;
  using ordinal_type_array_host = Kokkos::View<ordinal_type *, host_device_type>;
  using value_type_array_host = Kokkos::View<value_type *, host_device_type>;
  using mag_type_array_host = Kokkos::View<mag_type *, host_device_type>;
  using value_type_matrix_host = Kokkos::View<value_type **, Kokkos::LayoutLeft, host_device_type>;

  using crs_matrix_type = CrsMatrixBase<value_type, device_type>;
  using crs_matrix_type_host = CrsMatrixBase<value_type, host_device_type>;

#if defined(TACHO_HAVE_METIS) || defined(TACHO_HAVE_TRILINOS_SS)
  using graph_tools_type = GraphTools_Metis;
#else
  using graph_tools_type = GraphTools;
#endif

  using symbolic_tools_type = SymbolicTools;
  using numeric_tools_base_type = NumericToolsBase<value_type, device_type>;
  using numeric_tools_serial_type = NumericToolsSerial<value_type, device_type>;
  using numeric_tools_levelset_var0_type = NumericToolsLevelSet<value_type, device_type, 0>;
  using numeric_tools_levelset_var1_type = NumericToolsLevelSet<value_type, device_type, 1>;
  using numeric_tools_levelset_var2_type = NumericToolsLevelSet<value_type, device_type, 2>;

private:
  enum : int { Cholesky = 1, LDL = 2, SymLU = 3, LU = 4, SkewLDL = 5 };

  // ** solver mode
  ordinal_type _method;

  // ** ordering options
  ordinal_type _order_connected_graph_separately;

  // ** problem
  ordinal_type _m;
  size_type _nnz;

  size_type_array _ap;
  size_type_array_host _h_ap;
  ordinal_type_array _aj;
  ordinal_type_array_host _h_aj;

  // ** matrix scaling
  bool _scale_mat;
  mag_type_array_host _d;

  // ** matching
  ordinal_type _num_swaps;

  // ** fill-reducing perm
  ordinal_type_array _perm;
  ordinal_type_array_host _h_perm;
  ordinal_type_array _peri;
  ordinal_type_array_host _h_peri;

  // ** condensed graph
  ordinal_type _m_graph;
  size_type _nnz_graph;

  size_type_array_host _h_ap_graph;
  ordinal_type_array_host _h_aj_graph;
  ordinal_type_array_host _h_aw_graph;

  ordinal_type_array_host _h_perm_graph;
  ordinal_type_array_host _h_peri_graph;

  // ** symbolic factorization output
  ordinal_type _nnz_u;
  // supernodes output
  ordinal_type _nsupernodes;
  ordinal_type_array _supernodes;

  // dof mapping to sparse matrix
  size_type_array _gid_super_panel_ptr;
  ordinal_type_array _gid_super_panel_colidx;

  // supernode map and panel size configuration
  size_type_array _sid_super_panel_ptr;
  ordinal_type_array _sid_super_panel_colidx, _blk_super_panel_colidx;

  // supernode elimination tree (parent - children)
  size_type_array _stree_ptr;
  ordinal_type_array _stree_children;

  // supernode elimination tree (child - parent)
  ordinal_type_array _stree_parent;

  // roots of supernodes
  ordinal_type_array_host _stree_level, _stree_roots;

  // ** numeric factorization output
  numeric_tools_base_type *_N;

  // small dense matrix
  // - chol A is used
  // - ldl A D P are used
  value_type_matrix_host _A, _D;
  ordinal_type_array_host _P;

  // ** options
  ordinal_type _verbose;             // print
  ordinal_type _small_problem_thres; // smaller than this, use lapack

#ifdef TACHO_DEPRECATED_PARAMETERS
  // // ** tasking options
  ordinal_type _serial_thres_size; // serialization threshold size
  ordinal_type _mb;                // block size for byblocks algorithms
  ordinal_type _nb;                // panel size for panel algorithms
  ordinal_type _front_update_mode; // front update mode 0 - lock, 1 - atomic
#endif

  // ** levelset options
#ifdef TACHO_DEPRECATED_PARAMETERS
  bool _levelset;                    // use level set code instead of tasking
#endif
  ordinal_type _device_level_cut;    // above this level, matrices are computed on device
  ordinal_type _device_factor_thres; // bigger than this threshold, device function is used
  ordinal_type _device_solve_thres;  // bigger than this threshold, device function is used
  ordinal_type _variant;             // algorithmic variant in levelset 0: naive, 1: invert diagonals
  ordinal_type _nstreams;            // on cuda, multi streams are used

  bool _pivot;                       // turn on/off local pivoting
  mag_type _pivot_tol;               // tolerance for tiny pivot perturbation
  bool _store_transpose;             // store transpose explicitly

#ifdef TACHO_DEPRECATED_PARAMETERS
  // parallelism and memory constraint is made via this parameter
  ordinal_type _max_num_superblocks; // # of superblocks in the memoyrpool
#endif

public:
  Driver();
  /// delete copy constructor and assignment operator
  /// sharing numeric tools for different inputs does not make sense
  Driver(const Driver &) = default;
  Driver &operator=(const Driver &) = default;

  /// duplicate the solver with sharing symbolic factorization
  Driver duplicate();

  ///
  /// common options
  ///
  void setVerbose(const ordinal_type verbose = 1);
  void setSmallProblemThresholdsize(const ordinal_type small_problem_thres = 1024);
  void setMatrixType(const int symmetric, // 0 - unsymmetric, 1 - structure sym, 2 - symmetric
                     const bool is_positive_definite);
  void setSolutionMethod(const int method); /// 1 - cholesky, 2 - LDL, 3 - LU

  ///
  /// Graph options
  ///
  void setOrderConnectedGraphSeparately(const ordinal_type order_connected_graph_separately = 1);

  ///
  /// tasking options
  ///
  void setSerialThresholdsize(const ordinal_type serial_thres_size = -1);
  void setBlocksize(const ordinal_type mb = -1);
  void setPanelsize(const ordinal_type nb = -1);
  void setFrontUpdateMode(const ordinal_type front_update_mode = 1);
  void setMaxNumberOfSuperblocks(const ordinal_type max_num_superblocks = -1);

  ///
  /// Level set tools options
  ///
  void setLevelSetScheduling(const bool levelset);
  void setLevelSetOptionDeviceLevelCut(const ordinal_type device_level_cut);
  void setLevelSetOptionDeviceFunctionThreshold(const ordinal_type device_factor_thres,
                                                const ordinal_type device_solve_thres);
  void setLevelSetOptionNumStreams(const ordinal_type nstreams);
  void setLevelSetOptionAlgorithmVariant(const ordinal_type variant);

  void doLocalPivot(const bool pivot);
  void setPivotTolerance(const mag_type pivot_tol);
  void useNoPivotTolerance();
  void useDefaultPivotTolerance();
  void storeExplicitTranspose(bool flag);

  ///
  /// get interface
  ///
  ordinal_type getNumNonZerosU() const;
  ordinal_type getNumSupernodes() const;
  ordinal_type_array getSupernodes() const;
  ordinal_type_array getPermutationVector() const;
  ordinal_type_array getInversePermutationVector() const;

  // internal only
  int analyze();
  int analyze_linear_system();
  int analyze_condensed_graph();

  template <typename arg_size_type_array, typename arg_ordinal_type_array>
  int analyze(const ordinal_type m, const arg_size_type_array &ap, const arg_ordinal_type_array &aj,
              const bool duplicate = false) {

    _m = m;
    _num_swaps = 0;

    if (duplicate) {
      /// for most cases, ap and aj are from host; so construct ap and aj and mirror to device
      _h_ap = size_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_ap"), ap.extent(0));
      Kokkos::deep_copy(_h_ap, ap);
      _h_aj = ordinal_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_aj"), aj.extent(0));
      Kokkos::deep_copy(_h_aj, aj);

      _ap = Kokkos::create_mirror_view(exec_memory_space(), _h_ap);
      Kokkos::deep_copy(_ap, _h_ap);
      _aj = Kokkos::create_mirror_view(exec_memory_space(), _h_aj);
      Kokkos::deep_copy(_aj, _h_aj);
    } else {
      /// this does not make any extra deep copy; users should hold the graph data
      _ap = Kokkos::create_mirror_view(exec_memory_space(), ap);
      Kokkos::deep_copy(_ap, ap);
      _aj = Kokkos::create_mirror_view(exec_memory_space(), aj);
      Kokkos::deep_copy(_aj, aj);

      _h_ap = Kokkos::create_mirror_view(host_memory_space(), ap);
      Kokkos::deep_copy(_h_ap, ap);
      _h_aj = Kokkos::create_mirror_view(host_memory_space(), aj);
      Kokkos::deep_copy(_h_aj, aj);
    }

    _h_perm = ordinal_type_array_host();
    _h_peri = ordinal_type_array_host();

    _nnz = _h_ap(m);

    _m_graph = 0;
    _nnz_graph = 0;

    _h_ap_graph = size_type_array_host();
    _h_aj_graph = ordinal_type_array_host();

    _h_perm_graph = ordinal_type_array_host();
    _h_peri_graph = ordinal_type_array_host();

    return analyze();
  }

  template <typename arg_size_type_array, typename arg_ordinal_type_array, typename arg_perm_type_array>
  int analyze(const ordinal_type m, const arg_size_type_array &ap, const arg_ordinal_type_array &aj,
              const arg_perm_type_array &perm, const arg_perm_type_array &peri, const bool duplicate = false) {
    _m = m;
    _num_swaps = 0;

    // this takes the user-specified perm, such that analyze() won't call graph partitioner
    if (duplicate) {
      /// for most cases, ap and aj are from host; so construct ap and aj and mirror to device
      _h_ap = size_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_ap"), ap.extent(0));
      Kokkos::deep_copy(_h_ap, ap);
      _h_aj = ordinal_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_aj"), aj.extent(0));
      Kokkos::deep_copy(_h_aj, aj);

      _ap = Kokkos::create_mirror_view(exec_memory_space(), _h_ap);
      Kokkos::deep_copy(_ap, _h_ap);
      _aj = Kokkos::create_mirror_view(exec_memory_space(), _h_aj);
      Kokkos::deep_copy(_aj, _h_aj);

      _h_perm = ordinal_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_perm"), perm.extent(0));
      _h_peri = ordinal_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_peri"), peri.extent(0));
    } else {
      /// this does not make any extra deep copy; users should hold the graph data
      _ap = Kokkos::create_mirror_view(exec_memory_space(), ap);
      Kokkos::deep_copy(_ap, ap);
      _aj = Kokkos::create_mirror_view(exec_memory_space(), aj);
      Kokkos::deep_copy(_aj, aj);

      _h_ap = Kokkos::create_mirror_view(host_memory_space(), ap);
      Kokkos::deep_copy(_h_ap, ap);
      _h_aj = Kokkos::create_mirror_view(host_memory_space(), aj);
      Kokkos::deep_copy(_h_aj, aj);

      _h_perm = Kokkos::create_mirror_view(host_memory_space(), perm);
      _h_peri = Kokkos::create_mirror_view(host_memory_space(), peri);
    }

    Kokkos::deep_copy(_h_perm, perm);
    Kokkos::deep_copy(_h_peri, peri);

    _nnz = _h_ap(m);

    _m_graph = 0;
    _nnz_graph = 0;

    _h_ap_graph = size_type_array_host();
    _h_aj_graph = ordinal_type_array_host();

    _h_perm_graph = ordinal_type_array_host();
    _h_peri_graph = ordinal_type_array_host();

    return analyze();
  }

  template <typename arg_size_type_array, typename arg_ordinal_type_array>
  int analyze(const ordinal_type m, const arg_size_type_array &ap, const arg_ordinal_type_array &aj,
              const ordinal_type m_graph, const arg_size_type_array &ap_graph, const arg_ordinal_type_array &aj_graph,
              const arg_ordinal_type_array &aw_graph, const bool duplicate = false) {
    _m = m;
    _num_swaps = 0;

    if (duplicate) {
      /// for most cases, ap and aj are from host; so construct ap and aj and mirror to device
      _h_ap = size_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_ap"), ap.extent(0));
      Kokkos::deep_copy(_h_ap, ap);
      _h_aj = ordinal_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_aj"), aj.extent(0));
      Kokkos::deep_copy(_h_aj, aj);

      _ap = Kokkos::create_mirror_view(exec_memory_space(), _h_ap);
      Kokkos::deep_copy(_ap, _h_ap);
      _aj = Kokkos::create_mirror_view(exec_memory_space(), _h_aj);
      Kokkos::deep_copy(_aj, _h_aj);
    } else {
      /// this does not make any extra deep copy; users should hold the graph data
      _ap = Kokkos::create_mirror_view(exec_memory_space(), ap);
      Kokkos::deep_copy(_ap, ap);
      _aj = Kokkos::create_mirror_view(exec_memory_space(), aj);
      Kokkos::deep_copy(_aj, aj);

      _h_ap = Kokkos::create_mirror_view(host_memory_space(), ap);
      Kokkos::deep_copy(_h_ap, ap);
      _h_aj = Kokkos::create_mirror_view(host_memory_space(), aj);
      Kokkos::deep_copy(_h_aj, aj);
    }

    _h_perm = ordinal_type_array_host();
    _h_peri = ordinal_type_array_host();

    _nnz = _h_ap(m);

    _m_graph = m_graph;
    if (duplicate) {
      _h_ap_graph = size_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_ap_graph"), ap_graph.extent(0));
      _h_aj_graph = ordinal_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_aj_graph"), aj_graph.extent(0));
      _h_aw_graph = ordinal_type_array_host(Kokkos::ViewAllocateWithoutInitializing("h_aw_graph"), aw_graph.extent(0));
    } else {
      _h_ap_graph = Kokkos::create_mirror_view(host_memory_space(), ap_graph);
      _h_aj_graph = Kokkos::create_mirror_view(host_memory_space(), aj_graph);
      _h_aw_graph = Kokkos::create_mirror_view(host_memory_space(), aw_graph);
    }

    Kokkos::deep_copy(_h_ap_graph, ap_graph);
    Kokkos::deep_copy(_h_aj_graph, aj_graph);
    Kokkos::deep_copy(_h_aw_graph, aw_graph);

    _h_perm_graph = ordinal_type_array_host();
    _h_peri_graph = ordinal_type_array_host();

    _nnz_graph = _h_ap_graph(m_graph);

    return analyze();
  }

  template <typename arg_size_type_array, typename arg_ordinal_type_array>
  int analyze(const ordinal_type m, const ordinal_type blk_size,
              const arg_size_type_array &ap, const arg_ordinal_type_array &aj,
              const int max_match = -1, const bool duplicate = false) {

    value_type_array av;
    return analyze(m, blk_size, ap, aj, av, max_match, false, duplicate);
  }

  template <typename arg_size_type_array, typename arg_ordinal_type_array, typename arg_value_type_array>
  int analyze(const ordinal_type m, const ordinal_type blk_size,
              const arg_size_type_array &ap, const arg_ordinal_type_array &aj, const arg_value_type_array &av,
              const int max_match = -1, const bool scale_mat = false, const bool duplicate = false) {

    _num_swaps = 0;

    int rval = 0;
    Kokkos::Timer timer;
    printf( "\n >> in analyze (blk-size=%d, max-match=%d%s) <<\n",blk_size,max_match,(scale_mat ? ", scale" : "") ); fflush(stdout);
    if (blk_size > 1) {
      double t_shrink = 0.0;
      double t_match  = 0.0;

      // ** max cardinarity matchng
      int num_swaps = 0;
      ordinal_type_array_host h_match("h_match",0);
      ordinal_type_array_host h_imatch("h_imatch",0);
      if (max_match >= 0) {
        _h_ap = Kokkos::create_mirror_view(host_memory_space(), ap);
        _h_aj = Kokkos::create_mirror_view(host_memory_space(), aj);
        auto h_av = Kokkos::create_mirror_view(host_memory_space(), av);
        Kokkos::deep_copy(_h_ap, ap);
        Kokkos::deep_copy(_h_aj, aj);
        int num_match = -1;
        double maxwork = 0.0;
        double work;
        bool do_mwm = (av.extent(0) == aj.extent(0));
        printf( "   > using %s\n",(do_mwm ? "max-weight matching" : "max-cardinarity matching") );
        if (do_mwm) {
        }
        size_type_array_host iwork("iwork", 5*m);
        {
          // compress & extract
          int nnz = ap(m);
          size_type_array_host  h_ap_odd_upp("h_ap_odd", 1+m/2);
          size_type_array_host  h_aj_odd_upp("h_aj_odd", nnz);
          value_type_array_host h_av_odd_upp("h_av_odd", nnz);
          nnz = 0;
          h_ap_odd_upp(0) = nnz;
          for (int i=0; i<m; i+=2) {
            for (int k=_h_ap(i); k<_h_ap(i+1); k++) {
              if (_h_aj(k)%2 == 1) {
                if (true) {
                  // all odd rows
                  h_aj_odd_upp(nnz) = (_h_aj(k)-1)/2;
                  if (do_mwm) h_av_odd_upp(nnz) = h_av(k);
                  nnz++;
                } else if (_h_aj(k) > i) {
                  // only upper
                  h_aj_odd_upp(nnz) = (_h_aj(k)-1)/2;
                  if (do_mwm) h_av_odd_upp(nnz) = h_av(k);
                  nnz++;
                }
              }
            }
            h_ap_odd_upp(i/2+1) = nnz;
          }
          Kokkos::resize(h_aj_odd_upp, nnz);
          // input for cardinarity imatchng
          value_type_array_host h_av_odd;
          size_type_array_host h_aj_odd;
          size_type_array_host h_ap_odd("h_ap_odd", 1+m/2);
          if (false) {
            // just use extracted (upper, or odd) part
            Kokkos::resize(h_aj_odd, nnz);
            Kokkos::deep_copy(h_ap_odd, h_ap_odd_upp);
            Kokkos::deep_copy(h_aj_odd, h_aj_odd_upp);
          } else {
            // expand to full, or transpose
            bool expand_full = false;
            Kokkos::resize(h_aj_odd, (expand_full ? 2*nnz : nnz));
            if (do_mwm)
              Kokkos::resize(h_av_odd, (expand_full ? 2*nnz : nnz));
            h_ap_odd(0) = 0;
            for (int i=0; i<m/2; i++) {
              if (expand_full) {
                // expand to full
                h_ap_odd(i+1) = h_ap_odd_upp(i+1)-h_ap_odd_upp(i);
              } else {
                h_ap_odd(i+1) = 0;
                for (int k=h_ap_odd_upp(i); k<h_ap_odd_upp(i+1); k++) {
                  if (h_aj_odd_upp(k) == i) {
                    // just diagonal
                    h_ap_odd(i+1) = 1;
                  }
                }
              }
            }
            // to expand to strictly-lower, or to transpose
            for (int i=0; i<m/2; i++) {
              for (int k=h_ap_odd_upp(i); k<h_ap_odd_upp(i+1); k++) {
                if (h_aj_odd_upp(k) != i) {
                  h_ap_odd(h_aj_odd_upp(k)+1) ++;
                }
              }
            }
            // insert nz indices
            for (int i=0; i<m/2; i++) h_ap_odd(i+1) += h_ap_odd(i);
            for (int i=0; i<m/2; i++) {
              for (int k=h_ap_odd_upp(i); k<h_ap_odd_upp(i+1); k++) {
                if (expand_full) {
                  // upper
                  h_aj_odd(h_ap_odd(i)) = h_aj_odd_upp(k);
                  if (do_mwm) h_av_odd(h_ap_odd(i)) = abs(h_av_odd_upp(k));
                  h_ap_odd(i) ++;
                } else if (h_aj_odd_upp(k) == i) {
                  // just diagonal
                  h_aj_odd(h_ap_odd(i)) = h_aj_odd_upp(k);
                  if (do_mwm) h_av_odd(h_ap_odd(i)) = abs(h_av_odd_upp(k));
                  h_ap_odd(i) ++;
                }
                if (h_aj_odd_upp(k) != i) {
                  // strictly-lower
                  h_aj_odd(h_ap_odd(h_aj_odd_upp(k))) = i;
                  if (do_mwm) h_av_odd(h_ap_odd(h_aj_odd_upp(k))) = abs(h_av_odd_upp(k));
                  h_ap_odd(h_aj_odd_upp(k)) ++;
                }
              }
            }
            for (int i=m/2-1; i>0; i--) h_ap_odd(i) = h_ap_odd(i-1);
            h_ap_odd(0) = 0;
          }
          t_shrink = timer.seconds();;
          timer.reset();
          //printf( " calling maxtrans\n" ); fflush(stdout);
          /*{
            printf("A=[\n");
            if (do_mwm)
              for (int i=0; i<m; i++) for (int k=_h_ap(i); k<_h_ap(i+1); k++) printf("%d %d %e\n",i,_h_aj(k),h_av(k));
            else
              for (int i=0; i<m; i++) for (int k=_h_ap(i); k<_h_ap(i+1); k++) printf("%d %d\n",i,_h_aj(k));
            printf("];\n");
          }*/
          /*{
            printf("U=[\n");
            if (do_mwm)
              for (int i=0; i<m/2; i++) for (int k=h_ap_odd_upp(i); k<h_ap_odd_upp(i+1); k++) printf("%d %d %e\n",i,h_aj_odd_upp(k),h_av_odd_upp(k));
            else
              for (int i=0; i<m/2; i++) for (int k=h_ap_odd_upp(i); k<h_ap_odd_upp(i+1); k++) printf("%d %d\n",i,h_aj_odd_upp(k));
            printf("];\n"); fflush(stdout);
          }*/
          /*{
            printf("B=[\n");
            if (do_mwm)
              for (int i=0; i<m/2; i++) for (int k=h_ap_odd(i); k<h_ap_odd(i+1); k++) printf("%d %d %e\n",i,h_aj_odd(k),abs(h_av_odd(k)));
            else
              for (int i=0; i<m/2; i++) for (int k=h_ap_odd(i); k<h_ap_odd(i+1); k++) printf("%d %d\n",i,h_aj_odd(k));
            printf("];\n"); fflush(stdout);
          }*/
          // compute max cardinarity matching
          size_type_array_host match_odd("match_odd",m/2);
          if (do_mwm) {
#if defined(TACHO_HAVE_SUPERLUDIST)
            using int_type_array_host = Kokkos::View<int    *, host_device_type>;
            using dbl_type_array_host = Kokkos::View<double *, host_device_type>;

            int liw, ldw;
            int icntl[10], info[10];
            /* Possible values for JOB are: */
            /*   1 Compute a column permutation of the matrix so that the */
            /*     permuted matrix has as many entries on its diagonal as possible. */
            /*     The values on the diagonal are of arbitrary size. HSL subroutine */
            /*     MC21A/AD is used for this. See [1]. */
            /*   2 Compute a column permutation of the matrix so that the smallest */
            /*     value on the diagonal of the permuted matrix is maximized. */
            /*     See [3]. */
            /*   3 Compute a column permutation of the matrix so that the smallest */
            /*     value on the diagonal of the permuted matrix is maximized. */
            /*     The algorithm differs from the one used for JOB = 2 and may */
            /*     have quite a different performance. See [2]. */
            /*   4 Compute a column permutation of the matrix so that the sum */
            /*     of the diagonal entries of the permuted matrix is maximized. */
            /*     See [3]. */
            /*   5 Compute a column permutation of the matrix so that the product */
            /*     of the diagonal entries of the permuted matrix is maximized */
            /*     and vectors to scale the matrix so that the nonzero diagonal */
            /*     entries of the permuted matrix are one in absolute value and */
            /*     all the off-diagonal entries are less than or equal to one in */
            /*     absolute value. See [3]. */
            int job = 4; // job = 4 seems to do best
            //int job = 5; // job = 5 with scaling may do better
            int scaling_option = 2; // 2 seems to do the best
            int n = m/2;
            if (max_match >= 1 && max_match <= 5) {
              job = max_match;
            }

            liw = 5*n;
            if(job == 3) 
              { liw = 10*n + n; }
            ldw = 3*n+nnz;
            int_type_array_host iw("iw", liw);
            dbl_type_array_host dw("dw", ldw);

            // Abs nzvals
            dbl_type_array_host nzval_abs("nzval_abs", nnz);
            for(int i = 0; i < h_ap_odd(n); ++i)
              nzval_abs(i) = abs(h_av_odd(i));

            //Convert to 1 formatting
            for(int i = 0; i < h_ap_odd(n); ++i)
              h_aj_odd(i) = h_aj_odd(i)+1;
            for(int i = 0; i <= n; ++i)
              h_ap_odd(i) = h_ap_odd(i)+1;

            printf( "   > calling SuperLU_DIST MC64(job = %d) \n",job ); fflush(stdout);
            mc64id_dist(icntl);
            mc64ad_dist(&job, &n, &nnz, h_ap_odd.data(), h_aj_odd.data(), nzval_abs.data(),
                        &num_match, match_odd.data(), &liw, iw.data(), &ldw, dw.data(), icntl, info);

            if (job == 5) {
              printf( "   > using (scaling option = %d) \n",scaling_option ); fflush(stdout);
              Kokkos::resize(_d, m);
              for (int i = 0; i < n; ++i) {
                double r = exp(dw(n+i));
                double c = exp(dw(i));
                if (scaling_option == 1) {
                    _d(2*i)   = r;
                    _d(2*i+1) = c;
                } else if (scaling_option == 2) {
                    _d(2*i)   = c;
                    _d(2*i+1) = r;
                } else {
                    _d(2*i)   = c*r;
                    _d(2*i+1) = 1.0;
                }
                //_d(2*i+1) = _d(2*i) = 1.0;
              }
            }
            if (scale_mat) {
              // Find matrix scaling
              mag_type_array_host d_new("d_new",m);
	      if (job != 5) {
                Kokkos::resize(_d, m);
                Kokkos::deep_copy(_d, mag_type(1.0));
	      }
              for (int itr=0; itr<3; itr++)
              {
                for (int i=0; i<m; i++) {
                  d_new(i) = mag_type(0.0);
                  for (int k=_h_ap(i); k<_h_ap(i+1); k++) {
                    mag_type val = abs(h_av(k)) / (_d(i) * _d(_h_aj(k)));
                    if (val > d_new(i)) d_new(i) = val;
                  }
                  d_new(i) = sqrt(d_new(i));
                }
                for (int i=0; i<m; i++) _d(i) *= d_new(i);
              }
              _scale_mat = scale_mat;
            }
            if (job == 5) {
              _scale_mat = true;
            }
            //convert indexing back
            for(int i=0; i <= n; ++i)
            { h_ap_odd(i) = h_ap_odd(i)-1; }
            for(int i=0; i < h_ap_odd(n); ++i)
            { h_aj_odd(i) = h_aj_odd(i)-1; }
            for(int i=0; i < n; ++i)
            { match_odd(i) = match_odd(i)-1; }
#else
            printf( "   > calling ShyLU-Basker MWM\n" );
            int rval = mwm(m/2, nnz, h_ap_odd.data(), h_aj_odd.data(), h_av_odd.data(), match_odd.data(), num_match);
#endif
          } else {
            num_match = trilinos_btf_maxtrans (m/2, m/2, h_ap_odd.data(), h_aj_odd.data(), maxwork, &work, match_odd.data(), iwork.data());
          }
          t_match = timer.seconds();
          if (num_match < m/2) {
            printf( "\n ** WARNING : num_match = %d, m = %d **\n\n",num_match,m/2 );
            ordinal_type_array_host h_match_check("h_match_check",m/2);
            ordinal_type_array_host h_match_not("h_match_check",m/2 - num_match);
            int not_match = 0;
            for (int i=0; i<m/2; i++) {
              if (match_odd(i) < 0) {
                h_match_not(not_match) = i;
                not_match ++;
              } else {
                h_match_check(match_odd(i)) = 1;
              }
            }
            not_match = 0;
            for (int i=0; i<m/2; i++) {
              if (h_match_check(i) == 0) {
                //printf( " > match_odd(%d)) = %d -> %d\n",h_match_not(not_match),match_odd(h_match_not(not_match)), -i );
                match_odd(h_match_not(not_match)) = -i;
                not_match ++;
              }
            }
          }
          /*{
            printf("p=[\n");
            for (int i=0; i<m/2; i++) printf("%d %d\n",i,match_odd(i));
            printf("];\n");
          }*/
          // expand matching to full matrix
          Kokkos::resize(h_match, m);
          Kokkos::resize(h_imatch, m);
          for (int i=0; i<m; i++) {
            if (i%2 == 0) {
              h_match(i) = i;
            } else {
              int match = match_odd((i-1)/2);
              if (match >= 0) {
                h_match(i) = 2*match+1;
              } else {
                match = -match;
                h_match(i) = -(2*match+1);
              }
            }
            h_imatch(abs(h_match(i))) = i;
          }
          {
            size_type_array_host visited("visited", m/2);
            for (int i=1; i<m/2; i++) visited(i) = 0;
            for (int i=1; i<m; i+=2) {
              int i1 = (i-1)/2;
              if (visited(i1) == 0) {
                visited(i1) = 1;

                // follow the chain
                int j = h_match(i);
                int i2 = (j-1)/2;
                while (i1 != i2) {
                  num_swaps ++;
		  if (visited(i2) != 0) printf( " %d already visited ?\n" );
                  visited(i2) = 1;

                  j = h_match(j);
                  i2 = (j-1)/2;
                }
              }
            }
          }
          //printf( " num_swaps = %d / %d\n",num_swaps,m/2 );
          /*{
            printf("q=[\n");
            for (int i=0; i<m; i++) printf("%d %d\n",i,h_match(i));
            printf("];\n"); fflush(stdout);
          }*/
          /*{
            if (_d.extent(0) == m) {
              printf("d=[\n");
              for (int i=0; i<m; i++) printf("%d %e\n",i,_d(i));
              printf("];\n"); fflush(stdout);
            }
	  }*/
	  /*{}
            //printf("T=[\n");
            //for (int i=0; i<m; i++) {
            //  for (int k=_h_ap(h_match(i)); k<_h_ap(h_match(i)+1); k++) {
            //    if (do_mwm) {
            //      printf("%d %d %e\n",i,h_imatch(_h_aj(k)),h_av(k));
            //    } else {
            //      printf("%d %d\n",i,h_imatch(_h_aj(k)));
            //    }
            //  }
            //}
            //printf("];\n");
          }*/
          //printf( " calling maxtrans, done\n" ); fflush(stdout);
        }
        if (_verbose) {
          printf("===========================\n");
          printf("  Time for matching (num_match = %d, num_swap = %d, n = %d)\n",num_match,num_swaps,m/2);
          printf("             time to compress: %10.6f s\n", t_shrink);
          printf("             time to match   : %10.6f s\n", t_match);
        }
      } // end of max match

      timer.reset();
      const size_type nnz = ap(m);
      ordinal_type m_graph = m / blk_size;

      ordinal_type_array_host aw_graph
        (Kokkos::ViewAllocateWithoutInitializing("wgs"), m_graph);
      size_type_array_host ap_graph
        (Kokkos::ViewAllocateWithoutInitializing("ap_graph"), 1+m_graph);
      ordinal_type_array_host aj_graph;

      //condense graph before calling analyze
      bool explicit_zeros = true; // TODO: move it as a user-specified paremeter
      if (explicit_zeros) {
        TACHO_TEST_FOR_EXCEPTION((m != blk_size * m_graph),
          std::logic_error, "Failed to initialize the condensed graph");
        size_type nnz_graph = nnz; // TODO: count nnz
        Kokkos::resize(aj_graph, nnz_graph);

        // condense the graph
        //printf( " condensing graph\n" ); fflush(stdout);
        size_type_array_host col_graph("col_graph", m_graph);
        nnz_graph = 0;
        ap_graph(0) = 0;
        for (ordinal_type b = 0; b < m; b += blk_size) {
          ordinal_type bi = b/blk_size;
          if (b > 0) {
            // zero out check
            for (ordinal_type i = ap_graph(bi-1); i < ap_graph(bi); i++) {
              col_graph(aj_graph(i)) = 0;
            }
          }
          for (ordinal_type i = b; i < b+blk_size; i++) {
            ordinal_type row = (max_match >= 0 ? h_match(i) : i);
            if (row < 0) {
              row = -row;
            }
            for (size_type k = ap(row); k < ap(row+1); k++) {
              ordinal_type col = (max_match >= 0 ? h_imatch(aj(k)) : aj(k));
              size_t bj = col/blk_size;
              if (col_graph(bj) == 0) {
                aj_graph(nnz_graph) = bj;
                col_graph(bj) = 1;
                nnz_graph++;
              }
            }
          }
          aw_graph(bi) = blk_size;
          ap_graph(bi+1) = nnz_graph;
        }
        Kokkos::resize(aj_graph, nnz_graph);
      } else {
        size_type nnz_graph = nnz / (blk_size*blk_size);
        TACHO_TEST_FOR_EXCEPTION((m != blk_size * m_graph || nnz != size_type(blk_size*blk_size) * nnz_graph),
          std::logic_error, "Failed to initialize the condensed graph");
        Kokkos::resize(aj_graph, nnz_graph);

        // condense the graph
        nnz_graph = 0;
        ap_graph(0) = 0;
        for (ordinal_type i = 0; i < m; i += blk_size) {
          for (size_type k = ap(i); k < ap(i+1); k++) {
            if (aj(k)%blk_size == 0) {
              aj_graph(nnz_graph) = aj(k)/blk_size;
              nnz_graph++;
            }
            aw_graph(i/blk_size) = blk_size;
            ap_graph((i/blk_size)+1) = nnz_graph;
          }
        }
        TACHO_TEST_FOR_EXCEPTION((nnz != size_type(blk_size*blk_size) * nnz_graph),
          std::logic_error, "Failed to condense graph");
      }
      //printf( " graph condensed\n" ); fflush(stdout);
      double t_compress = timer.seconds();
      if (_verbose) {
        printf("===========================\n");
        printf("  Time for Condensation(blk_size = %d%s): %10.6f s\n",blk_size,t_compress,(explicit_zeros ? " with explicit zeros" : ""));
        printf("\n");
      }
      /*{
        printf("ac=[\n");
        for (int i=0; i<m_graph; i++) for (int k=ap_graph(i); k<ap_graph(i+1); k++) printf("%d %d\n",i,aj_graph(k));
        printf("];\n");
      }*/
      rval = analyze(m, ap, aj, m_graph, ap_graph, aj_graph, aw_graph, duplicate);
      _num_swaps = num_swaps;
      if (max_match >= 0) {
        // * integrate the max-matching into fill-reducing perm
        //printf("perm0=[\n");
        //for (int i=0; i<m; i++) printf("%d %d\n",_h_perm(i),_h_peri(i));
        //printf("];\n");
        size_type_array perm("perm",m);
        for (int i=0; i<m; i++) perm(i) = h_match(_h_perm(i));
        for (int i=0; i<m; i++) {
          _h_perm(i) = perm(i);
          _h_peri(perm(i)) = i;
        }
        Kokkos::deep_copy(_perm, _h_perm);
        Kokkos::deep_copy(_peri, _h_peri);
        //printf("perm1=[\n");
        //for (int i=0; i<m; i++) printf("%d %d\n",_h_perm(i),_h_peri(i));
        //printf("];\n");
      }
    } else {
      rval = analyze(m, ap, aj, duplicate);
    }
    return rval;
  }

  int initialize();

  int factorize(const value_type_array &ax);
  int factorize_small_host(const value_type_array &ax);

  int solve(const value_type_matrix &x, const value_type_matrix &b, const value_type_matrix &t);
  int solve_small_host(const value_type_matrix &x, const value_type_matrix &b, const value_type_matrix &t);

  int diag(const value_type_array &d);
  int pfaffian();

  double computeRelativeResidual(const value_type_array &ax, const value_type_matrix &x, const value_type_matrix &b);
  void   computeSpMV(const value_type_array &ax, const value_type_matrix &x, value_type_matrix &b);

  int exportFactorsToCrsMatrix(crs_matrix_type &A);
  int release();

  void printParameters();
};

} // namespace Tacho

//#include "Tacho_Driver_Impl.hpp"

#endif
