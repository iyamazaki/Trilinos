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

  // ** max cardinarity imatchng
  ordinal_type_array_host _h_match;
  ordinal_type_array_host _h_imatch;
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
    _h_match = ordinal_type_array_host();

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
              const bool max_match = false, const bool duplicate = false) {

    int rval = 0;
    if (blk_size > 1) {
      if (max_match) {
        _h_ap = Kokkos::create_mirror_view(host_memory_space(), ap);
        _h_aj = Kokkos::create_mirror_view(host_memory_space(), aj);
        Kokkos::deep_copy(_h_ap, ap);
        Kokkos::deep_copy(_h_aj, aj);
        double maxwork = 0.0;
        double work;
        size_type_array iwork("iwork", 5*m);
        {
            // compress & extract
            int nnz = ap(m);
            size_type_array h_ap_odd_upp("h_ap_odd", 1+m/2);
            size_type_array h_aj_odd_upp("h_aj_odd", nnz);
            nnz = 0;
            h_ap_odd_upp(0) = nnz;
            for (int i=0; i<m; i+=2) {
              for (int k=_h_ap(i); k<_h_ap(i+1); k++) {
                if (_h_aj(k)%2 == 1) {
                  if (true) {
                    // all odd rows
                    h_aj_odd_upp(nnz) = (_h_aj(k)-1)/2;
                    nnz++;
                  } else if (_h_aj(k) > i) {
                    // only upper
                    h_aj_odd_upp(nnz) = (_h_aj(k)-1)/2;
                    nnz++;
                  }
                }
              }
              h_ap_odd_upp(i/2+1) = nnz;
            }
            Kokkos::resize(h_aj_odd_upp, nnz);
            // input for cardinarity imatchng
            size_type_array h_aj_odd;
            size_type_array h_ap_odd("h_ap_odd", 1+m/2);
            if (false) {
              // just use extracted (upper, or odd) part
              Kokkos::resize(h_aj_odd, nnz);
              Kokkos::deep_copy(h_ap_odd, h_ap_odd_upp);
              Kokkos::deep_copy(h_aj_odd, h_aj_odd_upp);
            } else {
              // expand to full, or transpose
              bool expand_full = false;
              Kokkos::resize(h_aj_odd, (expand_full ? 2*nnz : nnz));
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
                    h_ap_odd(i) ++;
                  } else if (h_aj_odd_upp(k) == i) {
                    // just diagonal
                    h_aj_odd(h_ap_odd(i)) = h_aj_odd_upp(k);
                    h_ap_odd(i) ++;
                  }
                  if (h_aj_odd_upp(k) != i) {
                    // strictly-lower
                    h_aj_odd(h_ap_odd(h_aj_odd_upp(k))) = i;
                    h_ap_odd(h_aj_odd_upp(k)) ++;
                  }
                }
              }
              for (int i=m/2-1; i>0; i--) h_ap_odd(i) = h_ap_odd(i-1);
              h_ap_odd(0) = 0;
            }
            //printf( " calling maxtrans\n" ); fflush(stdout);
            /*{
              printf("A=[\n");
              for (int i=0; i<m; i++) for (int k=_h_ap(i); k<_h_ap(i+1); k++) printf("%d %d\n",i,_h_aj(k));
              printf("];\n");
              printf("U=[\n");
              for (int i=0; i<m/2; i++) for (int k=h_ap_odd_upp(i); k<h_ap_odd_upp(i+1); k++) printf("%d %d\n",i,h_aj_odd_upp(k));
              printf("];\n"); fflush(stdout);
              printf("B=[\n");
              for (int i=0; i<m/2; i++) for (int k=h_ap_odd(i); k<h_ap_odd(i+1); k++) printf("%d %d\n",i,h_aj_odd(k));
              printf("];\n"); fflush(stdout);
            }*/
            // compute max cardinarity imatchng
            size_type_array match_odd("match_odd",m/2);
            int num_match = trilinos_btf_maxtrans (m/2, m/2, h_ap_odd.data(), h_aj_odd.data(), maxwork, &work, match_odd.data(), iwork.data());
            //printf( " > num_match = %d, m = %d\n",num_match,m/2 );
            Kokkos::resize(_h_match, m);
            Kokkos::resize(_h_imatch, m);
            for (int i=0; i<m; i++) {
              if (i%2 == 0) {
                _h_match(i) = i;
              } else {
                int match = match_odd((i-1)/2);
                _h_match(i) = 2*match+1;
              }
              _h_imatch(_h_match(i)) = i;
            }
            /*{
              printf("p=[\n");
              for (int i=0; i<m/2; i++) printf("%d %d\n",i,match_odd(i));
              printf("];\n");
              printf("T=[\n");
              for (int i=0; i<m; i++) for (int k=_h_ap(_h_match(i)); k<_h_ap(_h_match(i)+1); k++) printf("%d %d\n",i,_h_imatch(_h_aj(k)));
              printf("];\n");
            }*/
            //printf( " calling maxtrans, done\n" ); fflush(stdout);
        }
        /*{
          printf("q=[\n");
          for (int i=0; i<m; i++) printf("%d %d\n",i,_h_match(i));
          printf("];\n"); fflush(stdout);
        }*/
      } // end of max match

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
        size_type_array_host col_graph
          (Kokkos::ViewAllocateWithoutInitializing("col_graph"), m_graph);
        nnz_graph = 0;
        ap_graph(0) = 0;
        for (ordinal_type b = 0; b < m; b += blk_size) {
          // TODO: zero out using ap_graph & aj_graph
          for (ordinal_type i = 0; i < m_graph; i++) {
            col_graph(i) = 0;
          }
          for (ordinal_type i = b; i < b+blk_size; i++) {
            ordinal_type row = (max_match ? _h_match(i) : i);
            for (size_type k = ap(row); k < ap(row+1); k++) {
              ordinal_type col = (max_match ? _h_imatch(aj(k)) : aj(k));
              size_t bj = col/blk_size;
              if (col_graph(bj) == 0) {
                aj_graph(nnz_graph) = bj;
                col_graph(bj) = 1;
                nnz_graph++;
              }
            }
          }
          aw_graph(b/blk_size) = blk_size;
          ap_graph((b/blk_size)+1) = nnz_graph;
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
      /*{
        printf("a=[\n");
        for (int i=0; i<m_graph; i++) for (int k=ap_graph(i); k<ap_graph(i+1); k++) printf("%d %d\n",i,aj_graph(k));
        printf("];\n");
      }*/
      rval = analyze(m, ap, aj, m_graph, ap_graph, aj_graph, aw_graph, duplicate);
      if (max_match) {
        //printf("perm0=[\n");
        //for (int i=0; i<m; i++) printf("%d %d\n",_h_perm(i),_h_peri(i));
        //printf("];\n");
        size_type_array perm("perm",m);
        for (int i=0; i<m; i++) perm(i) = _h_match(_h_perm(i));
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

  double computeRelativeResidual(const value_type_array &ax, const value_type_matrix &x, const value_type_matrix &b);
  void   computeSpMV(const value_type_array &ax, const value_type_matrix &x, value_type_matrix &b);

  int exportFactorsToCrsMatrix(crs_matrix_type &A);
  int release();

  void printParameters();
};

} // namespace Tacho

//#include "Tacho_Driver_Impl.hpp"

#endif
