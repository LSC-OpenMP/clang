//===----- CGOpenMPRuntimeNVPTX.h - Interface to OpenMP NVPTX Runtimes ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to NVPTX
// targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMESPARK_H
#define LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMESPARK_H

#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/StmtOpenMP.h"
#include "llvm/IR/CallSite.h"

namespace clang {
namespace CodeGen {

enum OpenMPOffloadMappingFlags {
  /// \brief No flags
  OMP_MAP_NONE = 0x0,
  /// \brief Allocate memory on the device and move data from host to device.
  OMP_MAP_TO = 0x01,
  /// \brief Allocate memory on the device and move data from device to host.
  OMP_MAP_FROM = 0x02,
  /// \brief Always perform the requested mapping action on the element, even
  /// if it was already mapped before.
  OMP_MAP_ALWAYS = 0x04,
  /// \brief Delete the element from the device environment, ignoring the
  /// current reference count associated with the element.
  OMP_MAP_DELETE = 0x08,
  /// \brief The element being mapped is a pointer-pointee pair; both the
  /// pointer and the pointee should be mapped.
  OMP_MAP_PTR_AND_OBJ = 0x10,
  /// \brief This flags signals that the base address of an entry should be
  /// passed to the target kernel as an argument.
  OMP_MAP_TARGET_PARAM = 0x20,
  /// \brief Signal that the runtime library has to return the device pointer
  /// in the current position for the data being mapped. Used when we have the
  /// use_device_ptr clause.
  OMP_MAP_RETURN_PARAM = 0x40,
  /// \brief This flag signals that the reference being passed is a pointer to
  /// private data.
  OMP_MAP_PRIVATE = 0x80,
  /// \brief Pass the element to the device by value.
  OMP_MAP_LITERAL = 0x100,
  /// \brief States the map is implicit.
  OMP_MAP_IMPLICIT = 0x200,
  /// \brief The 16 MSBs of the flags indicate whether the entry is member of
  /// some struct/class.
  OMP_MAP_MEMBER_OF = 0xffff000000000000
};

class CGOpenMPRuntimeSpark : public CGOpenMPRuntime {

public:
  explicit CGOpenMPRuntimeSpark(CodeGenModule &CGM);

  /// \brief Emit outlined function for 'target' directive on the NVPTX
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
                                  unsigned CaptureLevel) override;

  struct OMPSparkMappingInfo {
    llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 8>>
        InOutVarUse;
    llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 8>>
        InVarUse;
    llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 8>>
        OutVarDef;
    llvm::DenseMap<const VarDecl *, const OMPArraySectionExpr *> RangedVar;
    llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 8>>
        RangedArrayAccess;
    llvm::DenseMap<const Expr *, llvm::Value *> RangeIndexes;
    llvm::SmallVector<const VarDecl *, 8> ReducedVar;
    llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 8>>
        CounterUse;
    llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 4>>
        CounterInfo;
    llvm::DenseMap<const VarDecl *, llvm::Value *> KernelArgVars;
    unsigned Identifier;
  };

  unsigned CurrentIdentifier=0;
  llvm::SmallVector<OMPSparkMappingInfo *, 16> SparkMappingFunctions;

  bool IsSparkTargetRegion;

  llvm::DenseMap<const ValueDecl *, unsigned> OffloadingMapVarsIndex;
  llvm::DenseMap<const ValueDecl *, unsigned> OffloadingMapVarsType;

  int getMapType(const VarDecl *VD) {
    if (OffloadingMapVarsType.find(VD) != OffloadingMapVarsType.end()) {
      return OffloadingMapVarsType[VD];
    }
    return -1;
  }
  unsigned getOffloadingMapCurrentIdentifier() {
    return CurrentIdentifier++;
  }

  Expr *ActOnIntegerConstant(SourceLocation Loc, uint64_t Val);
  bool isNotSupportedLoopForm(Stmt *S, OpenMPDirectiveKind Kind, Expr *&InitVal,
                              Expr *&StepVal, Expr *&CheckVal, VarDecl *&VarCnt,
                              Expr *&CheckOp, BinaryOperatorKind &OpKind);

  void DefineJNITypes();
  void GenerateMappingKernel(const OMPExecutableDirective &S);
  void GenerateReductionKernel(const OMPReductionClause &C,
                               const OMPExecutableDirective &S);

  void EmitSparkJob();
  void EmitSparkNativeKernel(llvm::raw_fd_ostream &SPARK_FILE);
  void EmitSparkInput(llvm::raw_fd_ostream &SPARK_FILE);
  void EmitSparkMapping(llvm::raw_fd_ostream &SPARK_FILE,
                        OMPSparkMappingInfo &info, bool isLast);
  void EmitSparkOutput(llvm::raw_fd_ostream &SPARK_FILE);
  std::string getSparkVarName(const ValueDecl *VD);

  void addOpenMPKernelArgVar(const VarDecl *VD, llvm::Value *Addr) {
    assert(!SparkMappingFunctions.empty() &&
           "OpenMP private variables region is not started.");
    SparkMappingFunctions.back()->KernelArgVars[VD] = Addr;
  }
  void addOpenMPKernelArgRange(const Expr *E, llvm::Value *Addr) {
    SparkMappingFunctions.back()->RangeIndexes[E] = Addr;
  }
  void delOpenMPKernelArgVar(const VarDecl *VD) {
    assert(!SparkMappingFunctions.empty() &&
           "OpenMP private variables region is not started.");
    SparkMappingFunctions.back()->KernelArgVars[VD] = 0;
  }
  void addOffloadingMapVariable(
      const ValueDecl *VD, unsigned Type) {
    OffloadingMapVarsType[VD] = Type;
    OffloadingMapVarsIndex[VD] =
        getOffloadingMapCurrentIdentifier();
  }
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMESPARK_H
