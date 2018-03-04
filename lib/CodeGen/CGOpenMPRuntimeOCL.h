//===----- CGOpenMPRuntimeOCL.h - Interface to OpenCL/SPIR Runtimes ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to
// Opencl/spir64 targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMEOCL_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMEOCL_H

//#include <llvm/ADT/SmallBitVector.h>
#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
//#include "clang/Basic/IdentifierTable.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/CallSite.h"

namespace clang {
namespace CodeGen {

class CGOpenMPRuntimeOCL : public CGOpenMPRuntime {

public:
  llvm::BasicBlock *MasterContBlock;
  CodeGenFunction *currentCGF;

protected:
  enum OpenMPRTLKernelFunction {
    _cl_create_read_only,
    _cl_create_write_only,
    _cl_offloading_read_only,
    _cl_offloading_write_only,
    _cl_create_read_write,
    _cl_offloading_read_write,
    _cl_read_buffer,
    _cl_write_buffer,
    _cl_create_program,
    _cl_create_kernel,
    _cl_set_kernel_args,
    _cl_set_kernel_arg,
    _cl_set_kernel_hostArg,
    _cl_execute_kernel,
    _cl_execute_tiled_kernel,
    _cl_release_buffer,
    _cl_get_threads_blocks
  };

  llvm::DenseSet<DeclarationName> globals;
  llvm::BasicBlock *NumThreadsContBlock;
  llvm::Value *NumThreads;
  bool isTargetParallel;
  bool inParallel;

  void emitMasterHeader(CodeGenFunction &CGF);

  void emitMasterFooter();

  void emitNumThreadsHeader(CodeGenFunction &CGF, llvm::Value *NumThreads);

  void emitNumThreadsFooter(CodeGenFunction &CGF);

  void GenOpenCLArgMetadata(const RecordDecl *FD, llvm::Function *Fn,
                            CodeGenModule &CGM);

  llvm::Constant *createRuntimeFunction(OpenMPRTLKernelFunction Function);

  bool targetHasInnerOutlinedFunction(OpenMPDirectiveKind kind);

  bool teamsHasInnerOutlinedFunction(OpenMPDirectiveKind kind);

  /// \brief Creates offloading entry for the provided entry ID \a ID,
  /// address \a Addr, size \a Size, and flags \a Flags.
  virtual void createOffloadEntry(llvm::Constant *ID, llvm::Constant *Addr,
                                  uint64_t Size, uint64_t Flags = 0u);

public:
  // \brief Group the captures information for a given context.
  struct DataSharingInfo {
    enum DataSharingType {
      // A value allocated in the current function - the alloca has to be
      // replaced by the address in shared memory.
      DST_Val,
      // A reference captured into this function - the reference has to be
      // shared as is.
      DST_Ref,
      // A value allocated in the current function but required a cast in the
      // header - it has to be replaced by the address in shared memory and the
      // pointee has to be copied there.
      DST_Cast,
    };
    // The local values of the captures. The boolean indicates that what is
    // being shared is a reference and not the variable original storage.
    llvm::SmallVector<std::pair<const VarDecl *, DataSharingType>, 8>
        CapturesValues;
    llvm::SmallVector<std::pair<const Expr*, const VarDecl *>, 8>
        VLADeclMap;

    void add(const VarDecl *VD, DataSharingType DST) {
      CapturesValues.push_back(std::make_pair(VD, DST));
    }

    void addVLADecl(const Expr* VATExpr, const VarDecl *VD) {
      // VLADeclMap[VATExpr] = VD;
      VLADeclMap.push_back(std::make_pair(VATExpr, VD));
    }

    const VarDecl *getVLADecl(const Expr* VATExpr) const {
      for (auto ExprDeclPair : VLADeclMap) {
        if (ExprDeclPair.first == VATExpr) {
          return ExprDeclPair.second;
        }
      }
      assert(false && "No VAT expression that matches the input");
      return nullptr;
    }

    bool isVLADecl(const VarDecl* VD) const {
      for (auto ExprDeclPair : VLADeclMap) {
        if (ExprDeclPair.second == VD) {
          return true;
        }
      }
      return false;
    }

    // The record type of the sharing region if shared by the master.
    QualType MasterRecordType;
  };


  explicit CGOpenMPRuntimeOCL(CodeGenModule &CGM);

  /// \brief Emit outlined function for 'target' directive on the opencl/spir
  /// device.
  /// \param D Directive to emit.
  /// \param ParentName Name of the function that encloses the target region.
  /// \param OutlinedFn Outlined function value to be defined by this call.
  /// \param OutlinedFnID Outlined function ID value to be defined by this call.
  /// \param IsOffloadEntry True if the outlined function is an offload entry.
  /// An outlined function may not be an entry if, e.g. the if clause always
  /// evaluates to false.
  void emitTargetOutlinedFunction(const OMPExecutableDirective &D,
                                  StringRef ParentName,
                                  llvm::Function *&OutlinedFn,
                                  llvm::Constant *&OutlinedFnID,
                                  bool IsOffloadEntry,
                                  const RegionCodeGenTy &CodeGen,
                                  unsigned capturedLevel = 0) override;

  /// \brief Emits outlined function for the specified OpenMP parallel directive
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  virtual llvm::Value *emitParallelOutlinedFunction(
      const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
      OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
      unsigned CaptureLevel = 1, unsigned ImplicitParamStop = 0);

  /// \brief Emits code for parallel or serial call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run in parallel threads. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  /// \param IfCond Condition in the associated 'if' clause, if it was
  /// specified, nullptr otherwise.
  ///
  virtual void emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                llvm::Value *OutlinedFn,
                                ArrayRef<llvm::Value *> CapturedVars,
                                const Expr *IfCond);

  /// \brief Emits outlined function for the specified OpenMP teams directive
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  llvm::Value *emitTeamsOutlinedFunction(const OMPExecutableDirective &D,
                                         const VarDecl *ThreadIDVar,
                                         OpenMPDirectiveKind InnermostKind,
                                         const RegionCodeGenTy &CodeGen,
                                         unsigned CaptureLevel = 1,
                                         unsigned ImplicitParamStop = 0);

  /// \brief Emits code for teams call of the \a OutlinedFn with
  /// variables captured in a record which address is stored in \a
  /// CapturedStruct.
  /// \param OutlinedFn Outlined function to be run by team masters. Type of
  /// this function is void(*)(kmp_int32 *, kmp_int32, struct context_vars*).
  /// \param CapturedVars A pointer to the record with the references to
  /// variables used in \a OutlinedFn function.
  ///
  void emitTeamsCall(CodeGenFunction &CGF, const OMPExecutableDirective &D,
                     SourceLocation Loc, llvm::Value *OutlinedFn,
                     ArrayRef<llvm::Value *> CapturedVars) override;

  /// \brief Emits a master region.
  /// \param MasterOpGen Generator for the statement associated with the given
  /// master region.
  virtual void emitMasterRegion(CodeGenFunction &CGF,
                                const RegionCodeGenTy &MasterOpGen,
                                SourceLocation Loc);

  /// \brief Emit an implicit/explicit barrier for OpenMP threads.
  /// \param Kind Directive for which this implicit barrier call must be
  /// generated. Must be OMPD_barrier for explicit barrier generation.
  /// \param EmitChecks true if need to emit checks for cancellation barriers.
  /// \param ForceSimpleCall true simple barrier call must be emitted, false if
  /// runtime class decides which one to emit (simple or with cancellation
  /// checks).
  ///
  virtual void emitBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                               OpenMPDirectiveKind Kind, bool EmitChecks = true,
                               bool ForceSimpleCall = false);

  /// \brief Call the appropriate runtime routine to initialize it before start
  /// of loop.
  ///
  /// This is used only in case of static schedule, when the user did not
  /// specify a ordered clause on the loop construct.
  /// Depending on the loop schedule, it is necessary to call some runtime
  /// routine before start of the OpenMP loop to get the loop upper / lower
  /// bounds LB and UB and stride ST.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param DKind Kind of the directive.
  /// \param ScheduleKind Schedule kind, specified by the 'schedule' clause.
  /// \param Values Input arguments for the construct.
  ///
  virtual void emitForStaticInit(CodeGenFunction &CGF, SourceLocation Loc,
                                 OpenMPDirectiveKind DKind,
                                 const OpenMPScheduleTy &ScheduleKind,
                                 const StaticRTInput &Values) override;

  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param SchedKind Schedule kind, specified by the 'dist_schedule' clause.
  /// \param Values Input arguments for the construct.
  /// \param CoalescedDistSchedule Indicates if coalesced scheduling type is
  /// required.
  ///
  virtual void emitDistributeStaticInit(CodeGenFunction &CGF,
                                        SourceLocation Loc,
                                        OpenMPDistScheduleClauseKind SchedKind,
                                        const StaticRTInput &Values,
                                        bool Coalesced = false) override;

  /// \brief Call the appropriate runtime routine to notify that we finished
  /// all the work with current loop.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param Loc Clang source location.
  /// \param DKind Kind of the directive for which the static finish is emitted.
  ///
  virtual void emitForStaticFinish(CodeGenFunction &CGF, SourceLocation Loc,
                                   bool CoalescedDistSchedule = false);

  // Iteration of dynamic loop, i.e., Schedule (dynamic)
  virtual void emitForDispatchInit(CodeGenFunction &CGF, SourceLocation Loc,
                                   const OpenMPScheduleTy &ScheduleKind,
                                   unsigned IVSize, bool IVSigned, bool Ordered,
                                   llvm::Value *LB, llvm::Value *UB,
                                   llvm::Value *Chunk = nullptr) override;

  /// Call the appropriate runtime routine to notify that we finished
  /// iteration of the dynamic loop.
  ///
  /// \param CGF Reference to current CodeGenFunction.
  /// \param OpenMP Directive.
  /// \param Loc Clang source location.
  /// \param IVSize Size of the iteration variable in bits.
  /// \param IVSigned Sign of the interation variable.
  ///
  virtual void emitForDispatchFinish(CodeGenFunction &CGF,
                                     const OMPLoopDirective &S,
                                     SourceLocation Loc, unsigned IVSize,
                                     bool IVSigned);

  /// \brief Emits call to void __kmpc_push_num_threads(ident_t *loc, kmp_int32
  /// global_tid, kmp_int32 num_threads) to generate code for 'num_threads'
  /// clause.
  /// \param NumThreads An integer value of threads.
  virtual void emitNumThreadsClause(CodeGenFunction &CGF,
                                    llvm::Value *NumThreads,
                                    SourceLocation Loc);

  /// \brief Emits call to void __kmpc_push_num_teams(ident_t *loc, kmp_int32
  /// global_tid, kmp_int32 num_teams, kmp_int32 thread_limit) to generate code
  /// for num_teams clause.
  /// \param NumTeams An integer expression of teams.
  /// \param ThreadLimit An integer expression of threads.
  virtual void emitNumTeamsClause(CodeGenFunction &CGF, const Expr *NumTeams,
                                  const Expr *ThreadLimit, SourceLocation Loc);

  /// \brief Check if the specified \a ScheduleKind is static non-chunked.
  /// This kind of worksharing directive is emitted without outer loop.
  /// \param ScheduleKind Schedule kind specified in the 'schedule' clause.
  /// \param Chunked True if chunk is specified in the clause.
  ///
  virtual bool isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                  bool Chunked) const;

  /// \brief Check if the specified \a ScheduleKind is static non-chunked.
  /// This kind of distribute directive is emitted without outer loop.
  /// \param ScheduleKind Schedule kind specified in the 'dist_schedule' clause.
  /// \param Chunked True if chunk is specified in the clause.
  ///
  virtual bool isStaticNonchunked(OpenMPDistScheduleClauseKind ScheduleKind,
                                  bool Chunked) const;

  /// \brief Check if the specified \a ScheduleKind is dynamic.
  /// This kind of worksharing directive is emitted without outer loop.
  /// \param ScheduleKind Schedule Kind specified in the 'schedule' clause.
  ///
  virtual bool isDynamic(OpenMPScheduleClauseKind ScheduleKind) const;

  /// \brief Emit code for the directive that does not require outlining.
  ///
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  /// \param HasCancel true if region has inner cancel directive, false
  /// otherwise.
  virtual void emitInlinedDirective(CodeGenFunction &CGF,
                                    OpenMPDirectiveKind InnermostKind,
                                    const RegionCodeGenTy &CodeGen,
                                    bool HasCancel = false);

  /// \brief Registers the context of a parallel region with the runtime
  /// codegen implementation.
  void registerParallelContext(CodeGenFunction &CGF,
                               const OMPExecutableDirective &S) override;

  private:

  // \brief Map between a context and its data sharing information.
  typedef llvm::DenseMap<const Decl *, DataSharingInfo> DataSharingInfoMapTy;
  DataSharingInfoMapTy DataSharingInfoMap;

  // \brief Obtain the data sharing info for the current context.
  const DataSharingInfo &getDataSharingInfo(const Decl *Context);

  // \brief Create the data sharing info for the current context.
  void createDataSharingInfo(CodeGenFunction &CGF);

  // \brief Set of all functions that are offload entry points.
  llvm::SmallPtrSet<llvm::Function *, 16> EntryPointFunctionSet;

  // \brief Map between a function and its associated data sharing related
  // values.
  struct DataSharingFunctionInfo {
    bool RequiresOMPRuntime;
    bool IsEntryPoint;
    llvm::Function *EntryWorkerFunction;
    llvm::BasicBlock *EntryExitBlock;
    llvm::BasicBlock *InitDSBlock;
    llvm::Function *InitializationFunction;
    SmallVector<std::pair<llvm::Value *, bool>, 16> ValuesToBeReplaced;
    DataSharingFunctionInfo()
        : RequiresOMPRuntime(true), IsEntryPoint(false),
          EntryWorkerFunction(nullptr), EntryExitBlock(nullptr),
          InitDSBlock(nullptr), InitializationFunction(nullptr) {}
  };
  typedef llvm::DenseMap<llvm::Function *, DataSharingFunctionInfo>
      DataSharingFunctionInfoMapTy;
  DataSharingFunctionInfoMapTy DataSharingFunctionInfoMap;

  // \brief Create the data sharing replacement pairs at the top of a function
  // with parallel regions. If they were created already, do not do anything.
  void
  createDataSharingPerFunctionInfrastructure(CodeGenFunction &EnclosingCGF);

  // \brief Create the data sharing arguments and call the parallel outlined
  // function.
  llvm::Function *createDataSharingParallelWrapper(
      llvm::Function &OutlinedParallelFn, const OMPExecutableDirective &D,
      const Decl *CurrentContext, bool IsSimd = false);

  // \brief Map between an outlined function and its data-sharing-wrap version.
  llvm::DenseMap<llvm::Function *, llvm::Function *> WrapperFunctionsMap;

  // \brief Context that is being currently used for purposes of parallel region
  // code generarion.
  const Decl *CurrentParallelContext = nullptr;

};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMEOCL_H
