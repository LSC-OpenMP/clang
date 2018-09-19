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
#include "llvm/IR/Constant.h"

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

  static bool classof(const CGOpenMPRuntime *Runtime) { return true; }

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

  /// \brief Emits outlined function for the specified OpenMP parallel directive
  /// \a D. This outlined function has type void(*)(kmp_int32 *ThreadID,
  /// kmp_int32 BoundID, struct context_vars*).
  /// \param D OpenMP directive.
  /// \param ThreadIDVar Variable for thread id in the current OpenMP region.
  /// \param InnermostKind Kind of innermost directive (for simple directives it
  /// is a directive itself, for combined - its innermost directive).
  /// \param CodeGen Code generation sequence for the \a D directive.
  llvm::Value *emitParallelOutlinedFunction(
      const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
      OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
      unsigned CaptureLevel = 1, unsigned ImplicitParamStop = 0) override;

  llvm::Function *emitFakeOpenMPFunction(const CapturedStmt &S,
                                         bool UseCapturedArgumentsOnly = false,
                                         unsigned CaptureLevel = 1,
                                         unsigned ImplicitParamStop = 0,
                                         bool NonAliasedMaps = false);

  llvm::Function *
  outlineTargetDirective(const OMPExecutableDirective &D, StringRef Name,
                         const RegionCodeGenTy &CodeGen) override;

  class OMPSparkMappingInfo {
  public:
    const OMPLoopDirective *OMPDirective;
    llvm::SmallSet<const VarDecl *, 8> StrictlyInputs;
    llvm::SmallSet<const VarDecl *, 8> StrictlyOutputs;
    llvm::SmallSet<const VarDecl *, 8> BothInputsOutputs;
    llvm::SmallSet<const VarDecl *, 8> AllInputs;
    llvm::SmallSet<const VarDecl *, 8> AllOutputs;
    llvm::SmallSet<const VarDecl *, 8> AllArgs;
    llvm::DenseMap<const VarDecl *, const OMPArraySectionExpr *> RangedVar;
    llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 8>>
        RangedArrayAccess;
    llvm::DenseMap<const Expr *, llvm::Value *> RangeIndexes;
    llvm::SmallVector<const VarDecl *, 8> ReducedVar;
    llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 4>>
        CounterInfo;
    llvm::DenseMap<const VarDecl *, llvm::Value *> KernelArgVars;
    const int Identifier;
    static int _NextId;

    OMPSparkMappingInfo(const OMPLoopDirective *Directive)
        : OMPDirective(Directive), Identifier(_NextId++) {}
    ~OMPSparkMappingInfo() {}

    void addOpenMPKernelArgVar(const VarDecl *VD, llvm::Value *Addr) {
      KernelArgVars[VD] = Addr;
    }

    void addOpenMPKernelArgRange(const Expr *E, llvm::Value *Addr) {
      RangeIndexes[E] = Addr;
    }
  };

  unsigned CurrentScalaIdentifier = 0;
  llvm::SmallVector<OMPSparkMappingInfo, 16> SparkMappingFunctions;

  bool ShouldAccessJNIArgs = false;

  llvm::DenseMap<const ValueDecl *, unsigned> OffloadingMapVarsIndex;
  llvm::StringMap<unsigned> VarNameToScalaID;

  unsigned getNewScalaID() { return CurrentScalaIdentifier++; }

  Expr *ActOnIntegerConstant(SourceLocation Loc, uint64_t Val);
  bool isNotSupportedLoopForm(Stmt *S, OpenMPDirectiveKind Kind, Expr *&InitVal,
                              Expr *&StepVal, Expr *&CheckVal, VarDecl *&VarCnt,
                              Expr *&CheckOp, BinaryOperatorKind &OpKind);

  QualType jintQTy;
  QualType jbyteQTy;
  QualType jsizeQTy;
  QualType jbooleanQTy;
  QualType jlongQTy;

  QualType _jobjectQTy;
  QualType jobjectQTy;
  QualType jarrayQTy;
  QualType jbyteArrayQTy;

  QualType _jmethodIDQTy;
  QualType jmethodIDQTy;

  QualType _JNINativeInterfaceQTy;
  QualType JNIEnvQTy;
  QualType jclassQTy;

  llvm::Type *jintTy;
  llvm::Type *jbyteTy;
  llvm::Type *jsizeTy;
  llvm::Type *jbooleanTy;
  llvm::Type *jobjectTy;
  llvm::Type *jlongTy;
  llvm::Type *jarrayTy;
  llvm::Type *jbyteArrayTy;
  llvm::Type *jmethodIDTy;
  llvm::Type *JNIEnvTy;
  llvm::Type *jclassTy;

  void DefineJNITypes();
  void BuildJNITy();
  llvm::Constant *createJNIRuntimeFunction(unsigned Function);
  llvm::Value *EmitJNINewByteArray(CodeGenFunction &CGF, llvm::Value *Env,
                                   llvm::Value *Size);
  llvm::Value *EmitJNIReleaseByteArrayElements(CodeGenFunction &CGF,
                                               llvm::Value *Env,
                                               llvm::Value *Array,
                                               llvm::Value *Elems,
                                               llvm::Value *Mode);
  llvm::Value *EmitJNIGetByteArrayElements(CodeGenFunction &CGF,
                                           llvm::Value *Env, llvm::Value *Array,
                                           llvm::Value *IsCopy);
  llvm::Value *EmitJNIReleasePrimitiveArrayCritical(CodeGenFunction &CGF,
                                                    llvm::Value *Env,
                                                    llvm::Value *Array,
                                                    llvm::Value *Carray,
                                                    llvm::Value *Mode);
  llvm::Value *EmitJNIGetPrimitiveArrayCritical(CodeGenFunction &CGF,
                                                llvm::Value *Env,
                                                llvm::Value *Array,
                                                llvm::Value *IsCopy);
  llvm::Value *EmitJNICreateNewTuple(CodeGenFunction &CGF, llvm::Value *Env,
                                     ArrayRef<llvm::Value *> Elements);

  llvm::Function *GenerateMappingKernel(const OMPExecutableDirective &S);
  void GenerateReductionKernel(const OMPReductionClause &C,
                               const OMPExecutableDirective &S);

  void EmitSparkJob();
  void EmitSparkNativeKernel(llvm::raw_fd_ostream &SPARK_FILE);
  void EmitSparkInput(llvm::raw_fd_ostream &SPARK_FILE);
  void EmitSparkMapping(llvm::raw_fd_ostream &SPARK_FILE,
                        OMPSparkMappingInfo &info, bool isLast);
  void EmitSparkOutput(llvm::raw_fd_ostream &SPARK_FILE);
  static std::string getSparkVarName(const ValueDecl *VD);

  void addOffloadingMapVariable(const ValueDecl *VD) {
    if (OffloadingMapVarsIndex.find(VD) == OffloadingMapVarsIndex.end()) {
      unsigned ScalaId = getNewScalaID();
      OffloadingMapVarsIndex[VD] = ScalaId;
      VarNameToScalaID[VD->getNameAsString()] = ScalaId;
    }
  }

  /// \brief Checks, if the specified variable is currently an argument.
  /// \return 0 if the variable is not an argument, or address of the arguments
  /// otherwise.
  llvm::Value *getOpenMPKernelArgVar(const VarDecl *VD) {
    if (SparkMappingFunctions.empty())
      return 0;
    llvm::errs() << "Look for " << VD->getNameAsString() << "\n";
    return SparkMappingFunctions.back().KernelArgVars[VD];
  }
  /// \brief Checks, if the specified variable is currently an argument.
  /// \return 0 if the variable is not an argument, or address of the arguments
  /// otherwise.
  llvm::Value *getOpenMPKernelArgRange(const Expr *VarExpr) {
    if (SparkMappingFunctions.empty())
      return 0;
    return SparkMappingFunctions.back().RangeIndexes[VarExpr];
  }
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGOPENMPRUNTIMESPARK_H
