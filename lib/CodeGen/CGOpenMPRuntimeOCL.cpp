//===--- CGOpenMPRuntimeOCL.h - Interface to OpenMP OpenCL/SPIR Runtimes --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation specialized to
// OpenCL / SPIR targets.
//
//===----------------------------------------------------------------------===//

#include "CGCleanup.h"
#include "CGOpenMPRuntimeOCL.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenMP.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

using namespace clang;
using namespace CodeGen;

namespace {

/// \brief Base class for handling code generation inside OpenMP regions.
class CGOpenMPRegionInfo : public CodeGenFunction::CGCapturedStmtInfo {

public:
  /// \brief Kinds of OpenMP regions used in codegen.
  enum CGOpenMPRegionKind {
    /// \brief Region with outlined function for standalone 'parallel'
    /// directive.
    ParallelOutlinedRegion,
    /// \brief Region with outlined function for standalone 'simd'
    /// directive.
    SimdOutlinedRegion,
    /// \brief Region with outlined function for standalone 'task' directive.
    TaskOutlinedRegion,
    /// \brief Region for constructs that do not require function outlining,
    /// like 'for', 'sections', 'atomic' etc. directives.
    InlinedRegion,
    /// \brief Region with outlined function for standalone 'target' directive.
    TargetRegion,
  };

  CGOpenMPRegionInfo(const CapturedStmt &CS,
                     const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CS, CR_OpenMP), RegionKind(RegionKind),
        CodeGen(CodeGen), Kind(Kind), HasCancel(HasCancel) {}

  CGOpenMPRegionInfo(const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CR_OpenMP), RegionKind(RegionKind), CodeGen(CodeGen),
        Kind(Kind), HasCancel(HasCancel) {}

  /// \brief Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, const Stmt *S) override;

  CGOpenMPRegionKind getRegionKind() const { return RegionKind; }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  bool hasCancel() const { return HasCancel; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }

  ~CGOpenMPRegionInfo() override = default;

protected:

  CGOpenMPRegionKind RegionKind;
  RegionCodeGenTy CodeGen;
  OpenMPDirectiveKind Kind;
  bool HasCancel;

};

class CGOpenMPOutlinedRegionInfo final : public CGOpenMPRegionInfo {
public:
  CGOpenMPOutlinedRegionInfo(const CapturedStmt &CS, const VarDecl *ThreadIDVar,
                             const RegionCodeGenTy &CodeGen,
                             OpenMPDirectiveKind Kind, bool HasCancel)
      : CGOpenMPRegionInfo(CS, ParallelOutlinedRegion, CodeGen, Kind,
                           HasCancel),
        ThreadIDVar(ThreadIDVar) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               ParallelOutlinedRegion;
  }

private:
  /// \brief A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
};

void CGOpenMPRegionInfo::EmitBody(CodeGenFunction &CGF, const Stmt *S) {
  llvm::outs() << "OCL::EmitBody\n";
  if (!CGF.HaveInsertPoint())
    return;
  /* CGF.EHStack.pushTerminate(); */

  CodeGenModule &CGM = CGF.CGM;
  llvm::raw_fd_ostream CLOS(CGM.OpenMPSupport.createTempFile(), true);
  const std::string FileName = CGM.OpenMPSupport.getTempName();
  std::string includeContents = CGM.OpenMPSupport.getIncludeStr();
  std::string argsStr = CGM.OpenMPSupport.getArgsStr();

  // Add the basic c header files.
  CLOS << "#include <stdlib.h>\n";
  CLOS << "#include <stdint.h>\n";
  CLOS << "#include <math.h>\n\n";

  if (includeContents != "") {
      // There are OpenMP Declare statements
      CLOS << includeContents << "\n";
  }

  CLOS << "void foo (\n";
  CLOS << argsStr << ") {\n";
  // Dump the args and the loop body for clang-pcg
  CLOS << "\n#pragma scop\n";
  S->printPretty(CLOS, nullptr, PrintingPolicy(CGF.getContext().getLangOpts()), 4);
  CLOS << "\n#pragma endscop\n}\n";
  CLOS.close();

  // Insert call to clang-pcg, the polyhedral code generation module
  const std::string cName = FileName + ".c";
  rename(FileName.c_str(), cName.c_str());
  std::string pcg = "clang-pcg " + cName;
  std::system(pcg.c_str());

  /* CGF.EHStack.popTerminate(); */
}

/// Run the provided function with the shared loop bounds of a loop directive.
static void DoOnSharedLoopBounds(
    const OMPExecutableDirective &D,
    const llvm::function_ref<void(const VarDecl *, const VarDecl *)> &Exec) {
  // Is this a loop directive?
  if (isOpenMPLoopBoundSharingDirective(D.getDirectiveKind())) {
    auto *LDir = dyn_cast<OMPLoopDirective>(&D);
    // Do the bounds of the associated loop need to be shared? This check is the
    // same as checking the existence of an expression that refers to a previous
    // (enclosing) loop.
    if (LDir->getPrevLowerBoundVariable()) {
      const VarDecl *LB = cast<VarDecl>(
          cast<DeclRefExpr>(LDir->getLowerBoundVariable())->getDecl());
      const VarDecl *UB = cast<VarDecl>(
          cast<DeclRefExpr>(LDir->getUpperBoundVariable())->getDecl());
      Exec(LB, UB);
    }
  }
}

} // namespace

//
// Some assets used by OpenMPRuntimeOCL
//

enum DATA_SHARING_SIZES {
  // The maximum number of threads per block.
  DS_Max_Worker_Threads = 1024,
  // The max number of dimmensions
  DS_Slots = 3,
  // The maximum number of blocks.
  DS_Max_Blocks = 32
};

// bufferNames keeps the order in which the buffers are
// created during the offloading data
std::vector<std::pair<int, std::string>> bufferNames;
// paramNames keeps the order in which the parameters are
// in the kernel function signature
std::vector<std::pair<int, std::string>> paramNames[8];
// pairCompare compare two strings in the above structures
bool pairCompare(const std::pair<int, std::string> &p1,
                 const std::pair<int, std::string> &p2) {
  return p1.second < p2.second;
}

struct Required {
  explicit Required(std::string val) : val_(val) {}

  bool operator()(const std::pair<int, std::string> &elem) const {
    return val_ == elem.second;
  }

private:
  std::string val_;
};

///
/// Get the Variable Name inside the Value argument
///
llvm::StringRef getVarNameAsString(llvm::Value *FV) {
  llvm::Value *LV = FV;
  if (isa<llvm::CastInst>(LV))
    LV = cast<llvm::CastInst>(LV)->getOperand(0);
  if (isa<llvm::GetElementPtrInst>(LV))
    LV = cast<llvm::GetElementPtrInst>(LV)->getPointerOperand();
  if (isa<llvm::LoadInst>(LV))
    LV = cast<llvm::LoadInst>(LV)->getPointerOperand();
  return LV->getName();
}

///
/// Get the Variable Type inside the Value argument
///
llvm::Type *getVarType(llvm::Value *FV) {
  llvm::Type *LTy;
  if (isa<llvm::AllocaInst>(FV))
    LTy = dyn_cast<llvm::AllocaInst>(FV)->getAllocatedType();
  else if (isa<llvm::CastInst>(FV))
    LTy = dyn_cast<llvm::CastInst>(FV)->getSrcTy();
  else
    LTy = dyn_cast<llvm::Instruction>(FV)->getOperand(0)->getType();

  return LTy;
}

/// \brief RAII for emitting code of OpenMP constructs.
class OutlinedFunctionRAII {
  CGOpenMPRuntimeOCL &RT;
  CodeGenModule &CGM;
  llvm::BasicBlock *oldMCB;
  CodeGenFunction *oldCGF;

public:
  OutlinedFunctionRAII(CGOpenMPRuntimeOCL &RT, CodeGenModule &CGM)
      : RT(RT), CGM(CGM), oldMCB(RT.MasterContBlock), oldCGF(RT.currentCGF) {

    RT.MasterContBlock = nullptr;
    RT.currentCGF = nullptr;
  }
  ~OutlinedFunctionRAII() {
    assert(RT.MasterContBlock == nullptr && "master region was not closed.");
    RT.MasterContBlock = oldMCB;
    RT.currentCGF = oldCGF;
  }
};

static llvm::Value *emitParallelOCLOutlinedFunction(
    CodeGenModule &CGM, const OMPExecutableDirective &D,
    const VarDecl *ThreadIDVar, OpenMPDirectiveKind InnermostKind,
    const RegionCodeGenTy &CodeGen, unsigned CaptureLevel,
    unsigned ImplicitParamStop) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");
  const auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  if (D.hasClausesOfKind<OMPDependClause>() &&
      isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
    CS = cast<CapturedStmt>(CS->getCapturedStmt());
  CodeGenFunction CGF(CGM, true);
  bool HasCancel = false;
  if (auto *OPD = dyn_cast<OMPParallelDirective>(&D))
    HasCancel = OPD->hasCancel();
  else if (auto *OPSD = dyn_cast<OMPParallelSectionsDirective>(&D))
    HasCancel = OPSD->hasCancel();
  else if (auto *OPFD = dyn_cast<OMPParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();
  CGOpenMPOutlinedRegionInfo CGInfo(*CS, ThreadIDVar,
                            CodeGen, InnermostKind, HasCancel);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  return CGF.GenerateOpenMPCapturedStmtFunction(
      *CS, /*UseCapturedArgumentsOnly=*/false, CaptureLevel, ImplicitParamStop);
}


CGOpenMPRuntimeOCL::CGOpenMPRuntimeOCL(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM) {
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP opencl/spir can only handle device code.");
}

void CGOpenMPRuntimeOCL::emitMasterHeader(CodeGenFunction &CGF) {
}

void CGOpenMPRuntimeOCL::emitMasterFooter() {
  // only close master region, if one is open
  if (MasterContBlock == nullptr)
    return;
  currentCGF->EmitBranch(MasterContBlock);
  currentCGF->EmitBlock(MasterContBlock, true);
  MasterContBlock = nullptr;
  return;
}

void CGOpenMPRuntimeOCL::emitNumThreadsHeader(CodeGenFunction &CGF,
                                              llvm::Value *NumThreads) {
}

void CGOpenMPRuntimeOCL::emitNumThreadsFooter(CodeGenFunction &CGF) {
}

bool CGOpenMPRuntimeOCL::targetHasInnerOutlinedFunction(
    OpenMPDirectiveKind kind) {
  switch (kind) {
  case OpenMPDirectiveKind::OMPD_target_parallel:
  case OpenMPDirectiveKind::OMPD_target_parallel_for:
  case OpenMPDirectiveKind::OMPD_target_parallel_for_simd:
  case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for:
  case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for_simd:
    isTargetParallel = true;
  case OpenMPDirectiveKind::OMPD_target_teams:
  case OpenMPDirectiveKind::OMPD_target_teams_distribute:
  case OpenMPDirectiveKind::OMPD_target_teams_distribute_simd:
    return true;
  default:
    return false;
  }
}

bool CGOpenMPRuntimeOCL::teamsHasInnerOutlinedFunction(
    OpenMPDirectiveKind kind) {
  switch (kind) {
  case OpenMPDirectiveKind::OMPD_teams_distribute_parallel_for:
  case OpenMPDirectiveKind::OMPD_teams_distribute_parallel_for_simd:
  case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for:
  case OpenMPDirectiveKind::OMPD_target_teams_distribute_parallel_for_simd:
    isTargetParallel = true;
    return true;
  default:
    return false;
  }
}

void CGOpenMPRuntimeOCL::GenOpenCLArgMetadata(const RecordDecl *FD,
                                              llvm::Function *Fn,
                                              CodeGenModule &CGM) {
  llvm::outs() << "OCL::GenOpenCLArgMetadata\n";
  CodeGenFunction CGF(CGM);
  llvm::LLVMContext &Context = CGM.getLLVMContext();
  CGBuilderTy Builder = CGF.Builder;
  SmallVector<llvm::Metadata *, 8> opSource = {
      llvm::ConstantAsMetadata::get(Builder.getInt32(1)), // OpenCL C
      llvm::ConstantAsMetadata::get(
          Builder.getInt32(20000))}; // OpenCL C Version
  llvm::MDNode *srcMD = llvm::MDNode::get(Context, opSource);
  Fn->getParent()->getOrInsertNamedMetadata("spir.Source")->addOperand(srcMD);
  // Create MDNodes that represent the kernel arg metadata.
  // Each MDNode is a list in the form of "key", N number of values which is
  // the same number of values as their are kernel arguments.
  ASTContext &ASTCtx = CGM.getContext();

  const PrintingPolicy &Policy = ASTCtx.getPrintingPolicy();

  // MDNode for the kernel argument address space qualifiers.
  SmallVector<llvm::Metadata *, 8> addressQuals;

  // MDNode for the kernel argument access qualifiers (images only).
  SmallVector<llvm::Metadata *, 8> accessQuals;

  // MDNode for the kernel argument type names.
  SmallVector<llvm::Metadata *, 8> argTypeNames;

  // MDNode for the kernel argument base type names.
  SmallVector<llvm::Metadata *, 8> argBaseTypeNames;

  // MDNode for the kernel argument type qualifiers.
  SmallVector<llvm::Metadata *, 8> argTypeQuals;

  // MDNode for the kernel argument names.
  SmallVector<llvm::Metadata *, 8> argNames;

  // for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
  //  const ParmVarDecl *parm = FD->getParamDecl(i);
  for (auto parm : FD->fields()) {
    QualType ty = parm->getType();
    std::string typeQuals;

    if (ty->isPointerType()) {
      QualType pointeeTy = ty->getPointeeType();

      // Get address qualifier.
      addressQuals.push_back(llvm::ConstantAsMetadata::get(
          Builder.getInt32(CGM.getContext().getTargetAddressSpace(
              pointeeTy.getAddressSpace()))));

      // Get argument type name.
      std::string typeName =
          pointeeTy.getUnqualifiedType().getAsString(Policy) + "*";

      // Turn "unsigned type" to "utype"
      std::string::size_type pos = typeName.find("unsigned");
      if (pointeeTy.isCanonical() && pos != std::string::npos)
        typeName.erase(pos + 1, 8);

      argTypeNames.push_back(llvm::MDString::get(Context, typeName));

      std::string baseTypeName =
          pointeeTy.getUnqualifiedType().getCanonicalType().getAsString(
              Policy) +
          "*";

      // Turn "unsigned type" to "utype"
      pos = baseTypeName.find("unsigned");
      if (pos != std::string::npos)
        baseTypeName.erase(pos + 1, 8);

      argBaseTypeNames.push_back(llvm::MDString::get(Context, baseTypeName));

      // Get argument type qualifiers:
      if (ty.isRestrictQualified())
        typeQuals = "restrict";
      if (pointeeTy.isConstQualified() ||
          (pointeeTy.getAddressSpace() == LangAS::opencl_constant))
        typeQuals += typeQuals.empty() ? "const" : " const";
      if (pointeeTy.isVolatileQualified())
        typeQuals += typeQuals.empty() ? "volatile" : " volatile";
    } else {
      uint32_t AddrSpc = 0;
      bool isPipe = ty->isPipeType();
      if (ty->isImageType() || isPipe)
        AddrSpc = CGM.getContext().getTargetAddressSpace(LangAS::opencl_global);

      addressQuals.push_back(
          llvm::ConstantAsMetadata::get(Builder.getInt32(AddrSpc)));

      // Get argument type name.
      std::string typeName;
      if (isPipe)
        typeName = ty.getCanonicalType()
                       ->getAs<PipeType>()
                       ->getElementType()
                       .getAsString(Policy);
      else
        typeName = ty.getUnqualifiedType().getAsString(Policy);

      // Turn "unsigned type" to "utype"
      std::string::size_type pos = typeName.find("unsigned");
      if (ty.isCanonical() && pos != std::string::npos)
        typeName.erase(pos + 1, 8);

      std::string baseTypeName;
      if (isPipe)
        baseTypeName = ty.getCanonicalType()
                           ->getAs<PipeType>()
                           ->getElementType()
                           .getCanonicalType()
                           .getAsString(Policy);
      else
        baseTypeName =
            ty.getUnqualifiedType().getCanonicalType().getAsString(Policy);

      argTypeNames.push_back(llvm::MDString::get(Context, typeName));

      // Turn "unsigned type" to "utype"
      pos = baseTypeName.find("unsigned");
      if (pos != std::string::npos)
        baseTypeName.erase(pos + 1, 8);

      argBaseTypeNames.push_back(llvm::MDString::get(Context, baseTypeName));

      argTypeQuals.push_back(llvm::MDString::get(Context, typeQuals));

      // Get image and pipe access qualifier:
      if (ty->isImageType() || ty->isPipeType()) {
        const OpenCLAccessAttr *A = parm->getAttr<OpenCLAccessAttr>();
        if (A && A->isWriteOnly())
          accessQuals.push_back(llvm::MDString::get(Context, "write_only"));
        else if (A && A->isReadWrite())
          accessQuals.push_back(llvm::MDString::get(Context, "read_write"));
        else
          accessQuals.push_back(llvm::MDString::get(Context, "read_only"));
      } else
        accessQuals.push_back(llvm::MDString::get(Context, "none"));

      // Get argument name.
      argNames.push_back(llvm::MDString::get(Context, parm->getName()));
    }

    Fn->setMetadata("kernel_arg_addr_space",
                    llvm::MDNode::get(Context, addressQuals));
    Fn->setMetadata("kernel_arg_access_qual",
                    llvm::MDNode::get(Context, accessQuals));
    Fn->setMetadata("kernel_arg_type",
                    llvm::MDNode::get(Context, argTypeNames));
    Fn->setMetadata("kernel_arg_base_type",
                    llvm::MDNode::get(Context, argBaseTypeNames));
    Fn->setMetadata("kernel_arg_type_qual",
                    llvm::MDNode::get(Context, argTypeQuals));
    if (CGM.getCodeGenOpts().EmitOpenCLArgMetadata)
      Fn->setMetadata("kernel_arg_name", llvm::MDNode::get(Context, argNames));
  }
}

llvm::Constant *
CGOpenMPRuntimeOCL::createRuntimeFunction(OpenMPRTLKernelFunction Function) {

  llvm::outs() << "CGOpenMPRuntimeOCL::createRuntimeFunction\n";

  llvm::Constant *RTLFn = nullptr;
  switch (Function) {
      case _cl_create_read_only: {
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, CGM.Int64Ty, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_read_only");
        break;
      }
      case _cl_create_write_only: {
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, CGM.Int64Ty, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_write_only");
        break;
      }
      case _cl_offloading_read_only: {
        llvm::Type *TParams[] = {CGM.Int64Ty, CGM.VoidPtrTy};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_offloading_read_only");
        break;
      }
      case _cl_offloading_write_only: {
        llvm::Type *TParams[] = {CGM.Int64Ty, CGM.VoidPtrTy};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_offloading_write_only");
        break;
      }
      case _cl_create_read_write: {
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, CGM.Int64Ty, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_read_write");
        break;
      }
      case _cl_offloading_read_write: {
        llvm::Type *TParams[] = {CGM.Int64Ty, CGM.VoidPtrTy};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_offloading_read_write");
        break;
      }
      case _cl_read_buffer: {
        llvm::Type *TParams[] = {CGM.Int64Ty, CGM.Int32Ty, CGM.VoidPtrTy};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_read_buffer");
        break;
      }
      case _cl_write_buffer: {
        llvm::Type *TParams[] = {CGM.Int64Ty, CGM.Int32Ty, CGM.VoidPtrTy};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_write_buffer");
        break;
      }
      case _cl_create_program: {
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, CGM.Int8PtrTy, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_program");
        break;
      }
      case _cl_create_kernel: {
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, CGM.Int8PtrTy, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_create_kernel");
        break;
      }
      case _cl_set_kernel_args: {
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, CGM.Int32Ty, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_kernel_args");
        break;
      }
      case _cl_set_kernel_arg: {
        llvm::Type *TParams[] = {CGM.Int32Ty, CGM.Int32Ty};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_kernel_arg");
        break;
      }
      case _cl_set_kernel_hostArg: {
        llvm::Type *TParams[] = {CGM.Int32Ty, CGM.Int32Ty, CGM.VoidPtrTy};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_kernel_hostArg");
        break;
      }
      case _cl_execute_kernel: {
        llvm::Type *TParams[] = {CGM.Int64Ty, CGM.Int64Ty, CGM.Int64Ty,
                                 CGM.Int32Ty};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_execute_kernel");
        break;
      }
      case _cl_execute_tiled_kernel: {
        llvm::Type *TParams[] = {CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty,
                                 CGM.Int32Ty, CGM.Int32Ty, CGM.Int32Ty};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_execute_tiled_kernel");
        break;
      }
      case _cl_release_buffer: {
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.VoidTy, CGM.Int32Ty, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_set_release_buffer");
        break;
      }
      case _cl_get_threads_blocks: {
        llvm::Type *TParams[] = {CGM.IntPtrTy, CGM.IntPtrTy, CGM.IntPtrTy,
                                 CGM.IntPtrTy, CGM.Int64Ty,  CGM.Int32Ty};
        llvm::FunctionType *FnTy =
            llvm::FunctionType::get(CGM.Int32Ty, TParams, false);
        RTLFn = CGM.CreateRuntimeFunction(FnTy, "_cl_get_threads_blocks");
        break;
      }
  }
  return RTLFn;
}

void CGOpenMPRuntimeOCL::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen, unsigned capturedLevel) {

  if (!IsOffloadEntry) // Nothing to do.
    return;

  llvm::outs() << "OCL::emitTargetOutlinedFunction\n";

  assert(!ParentName.empty() && "Invalid target region parent name!");
  /* CapturedStmt &CS = *cast<CapturedStmt>(D.getAssociatedStmt()); */
  /* for (auto capture : CS.captures()) { */
  /*   globals.insert(capture.getCapturedVar()->getDeclName()); */
  /* } */

  /* OutlinedFunctionRAII RAII(*this, CGM); */
  /* class MasterPrePostActionTy : public PrePostActionTy { */
  /*   CGOpenMPRuntimeOCL &RT; */

  /* public: */
  /*   MasterPrePostActionTy(CGOpenMPRuntimeOCL &RT) : RT(RT) {} */
  /*   void Enter(CodeGenFunction &CGF) override { RT.emitMasterHeader(CGF); } */
  /*   void Exit(CodeGenFunction &CGF) override { RT.emitMasterFooter(); } */
  /* } Action(*this); */

  /* if (!targetHasInnerOutlinedFunction(D.getDirectiveKind())) { */
  /*   CodeGen.setAction(Action); */
  /* } */
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen, 0);

  OutlinedFn->setCallingConv(llvm::CallingConv::C);
  OutlinedFn->addFnAttr(llvm::Attribute::NoUnwind);
  OutlinedFn->removeFnAttr(llvm::Attribute::OptimizeNone);

  llvm::outs() << "OCL::Dumping outlined function:\n";
  OutlinedFn->dump();

  /* TODO: Check if this is necessary */
  /* GenOpenCLArgMetadata(CS.getCapturedRecordDecl(), OutlinedFn, CGM); */

}

llvm::Value *CGOpenMPRuntimeOCL::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    unsigned CaptureLevel, unsigned ImplicitParamStop) {

  llvm::outs() << "OCL::emitParallelOutlinedFunction\n";
  this->emitMasterFooter();

  llvm::DenseSet<const VarDecl *> Lastprivates;
  for (const auto *C : D.getClausesOfKind<OMPLastprivateClause>()) {
    for (const auto *D : C->varlists())
      Lastprivates.insert(
          cast<VarDecl>(cast<DeclRefExpr>(D)->getDecl())->getCanonicalDecl());
  }

  llvm::DenseSet<DeclarationName> FirstPrivates;
  for (const auto *C : D.getClausesOfKind<OMPFirstprivateClause>()) {
    for (const auto *D : C->varlists()) {
      auto *OrigVD = cast<VarDecl>(cast<DeclRefExpr>(D)->getDecl());
      bool ThisFirstprivateIsLastprivate =
          Lastprivates.count(OrigVD->getCanonicalDecl()) > 0;
      if (!ThisFirstprivateIsLastprivate)
        FirstPrivates.insert(OrigVD->getDeclName());
    }
  }

  bool wasAlreadyParallel = inParallel;
  inParallel = true;
  OutlinedFunctionRAII RAII(*this, CGM);

  Stmt *Body = D.getAssociatedStmt();
  if (auto *CS = dyn_cast_or_null<CapturedStmt>(Body)) {
    Body = CS->getCapturedStmt();
  }

  llvm::Value *OutlinedFn = emitParallelOCLOutlinedFunction(
      CGM, D, ThreadIDVar, InnermostKind, CodeGen, CaptureLevel, ImplicitParamStop);

  inParallel = wasAlreadyParallel;
  if (auto Fn = dyn_cast<llvm::Function>(OutlinedFn)) {
    Fn->removeFnAttr(llvm::Attribute::NoInline);
    Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
    Fn->addFnAttr(llvm::Attribute::NoUnwind);
  }

  return OutlinedFn;
}

void CGOpenMPRuntimeOCL::emitParallelCall(CodeGenFunction &CGF,
                                          SourceLocation Loc,
                                          llvm::Value *OutlinedFn,
                                          ArrayRef<llvm::Value *> CapturedVars,
                                          const Expr *IfCond) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::outs() << "OCL::emitParallelCall\n";

  llvm::SmallVector<llvm::Value *, 16> RealArgs;
  /*
  RealArgs.push_back(
      llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(
          CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  RealArgs.push_back(
      llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(
          CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  */
  RealArgs.append(CapturedVars.begin(), CapturedVars.end());

  llvm::Function *F = cast<llvm::Function>(OutlinedFn);

  emitMasterHeader(CGF);

  llvm::DenseMap<llvm::Value *, llvm::GlobalVariable *> sharedVariables;
  unsigned addrSpaceLocal =
      CGM.getContext().getTargetAddressSpace(LangAS::opencl_local);
  unsigned addrSpaceGeneric =
      CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic);

  emitMasterFooter();

  /* for (llvm::Value *arg : RealArgs) { */
  /*   arg->dump(); */
  /* } */

  // call outlined parallel function:
  // CGF.EmitCallOrInvoke(OutlinedFn, RealArgs);

  // Emit code to reference the file that contain the kernels
  const std::string FileName = CGM.OpenMPSupport.getTempName();
  llvm::Value *FileStr = CGF.Builder.CreateGlobalStringPtr(FileName);
  CGF.EmitRuntimeCall(createRuntimeFunction(_cl_create_program), FileStr);

  int workSizes[8][3];
  int blockSizes[8][3];
  int kernelId, upperKernel = 0;
  int k = 0;
  for (kernelId = 0; kernelId < 8; ++kernelId) {
     for (int j = 0; j < 3; j++) {
         workSizes[kernelId][j] = 0;
         blockSizes[kernelId][j] = 0;
     }
     paramNames[kernelId].clear();
  }
  // sort the bufferNames. Is this necessary??
  std::sort(bufferNames.begin(), bufferNames.end(), pairCompare);

  std::ifstream argFile(FileName);
  if (argFile.is_open()) {
      int kind, index;
      std::string arg_name;
      int last_KernelId = -1;
      while (argFile >> kernelId) {
          assert(kernelId < 8 && "Invalid kernel identifier");
          if (kernelId != last_KernelId) {
              last_KernelId = kernelId;
              argFile >> workSizes[kernelId][0] >> workSizes[kernelId][1] >> workSizes[kernelId][2];
              argFile >> kernelId;
              assert(kernelId == last_KernelId && "Invalid kernel structure");
              argFile >> blockSizes[kernelId][0] >> blockSizes[kernelId][1] >> blockSizes[kernelId][2];
              argFile >> kernelId;
              assert(kernelId == last_KernelId && "Invalid kernel structure");
          }
          argFile >> kind >> index >> arg_name;
          if (kind == 1) {
              paramNames[kernelId].push_back(std::pair<int, std::string>(index, arg_name));
          } else if (kind == 2) {
              // Note: scalar names not yet supported in this release
              // scalarNames[kernelId].push_back(std::pair<int, std::string>(index, arg_name));
          } else
              assert (false && "Invalid kernel structure");
      }
      upperKernel = kernelId;
      argFile.close();
  }

  for (kernelId = 0; kernelId <= upperKernel; kernelId++) {
      llvm::Value *KernelStr = CGF.Builder.CreateGlobalStringPtr(FileName + std::to_string(kernelId));
      CGF.EmitRuntimeCall(createRuntimeFunction(_cl_create_kernel), KernelStr);

      // Set kernel args according pos & index of buffer
      k = 0;
      for (std::vector<std::pair<int, std::string>>::iterator I = bufferNames.begin(),
                   E = bufferNames.end();
           I != E; ++I) {
          std::vector<std::pair<int, std::string>>::iterator it =
                  std::find_if(paramNames[kernelId].begin(),
                               paramNames[kernelId].end(), Required((I)->second));
          if (it == paramNames[kernelId].end()) {
              // the arg is not required
          } else {
              llvm::Value *Args[] = {CGF.Builder.getInt32(k), CGF.Builder.getInt32((I)->first)};
              CGF.EmitRuntimeCall(createRuntimeFunction(_cl_set_kernel_arg), Args);
              k++;
          }
      }
      int workDim;
      if (workSizes[kernelId][2] != 0) workDim = 3;
      else if (workSizes[kernelId][1] != 0) workDim = 2;
      else workDim = 1;
      llvm::Value *GroupSize[] = {CGF.Builder.getInt32(workSizes[kernelId][0]),
                                  CGF.Builder.getInt32(workSizes[kernelId][1]),
                                  CGF.Builder.getInt32(workSizes[kernelId][2]),
                                  CGF.Builder.getInt32(blockSizes[kernelId][0]),
                                  CGF.Builder.getInt32(blockSizes[kernelId][1]),
                                  CGF.Builder.getInt32(blockSizes[kernelId][2]),
                                  CGF.Builder.getInt32(workDim)};
      CGF.EmitRuntimeCall(createRuntimeFunction(_cl_execute_tiled_kernel), GroupSize);
  }



  if (isTargetParallel) {
    return;
  }

  // copy back shared variables to threadlocal
  emitMasterHeader(CGF);
  for (auto pair : sharedVariables) {
    llvm::PointerType *argType = cast<llvm::PointerType>(pair.first->getType());
    llvm::Value *sharedVar = CGF.Builder.CreateAlignedLoad(
        pair.second,
        CGM.getDataLayout().getPrefTypeAlignment(argType->getElementType()));
    CGF.Builder.CreateAlignedStore(
        sharedVar, pair.first,
        CGM.getDataLayout().getPrefTypeAlignment(sharedVar->getType()));
  }
  if (inParallel) {
    emitMasterFooter();
  } else if (NumThreads) {
    emitNumThreadsFooter(CGF);
    NumThreads = nullptr;
  }
}

llvm::Value *CGOpenMPRuntimeOCL::emitTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    unsigned CaptureLevel, unsigned ImplicitParamStop) {

  llvm::outs() << "OCL::emitTeamsOutlinedFunction\n";

  OutlinedFunctionRAII RAII(*this, CGM);
  class TeamsPrePostActionTy : public PrePostActionTy {
    CGOpenMPRuntimeOCL &RT;

  public:
    TeamsPrePostActionTy(CGOpenMPRuntimeOCL &RT) : RT(RT) {}
    void Enter(CodeGenFunction &CGF) override { RT.emitMasterHeader(CGF); }
    void Exit(CodeGenFunction &CGF) override { RT.emitMasterFooter(); }
  } Action(*this);

  if (!teamsHasInnerOutlinedFunction(D.getDirectiveKind()))
    CodeGen.setAction(Action);

  llvm::Value *OutlinedFn = CGOpenMPRuntime::emitTeamsOutlinedFunction(
      D, ThreadIDVar, InnermostKind, CodeGen, CaptureLevel, ImplicitParamStop);
  if (auto Fn = dyn_cast<llvm::Function>(OutlinedFn)) {
    Fn->removeFnAttr(llvm::Attribute::NoInline);
    Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  }
  return OutlinedFn;
}

void CGOpenMPRuntimeOCL::emitTeamsCall(CodeGenFunction &CGF,
                                       const OMPExecutableDirective &D,
                                       SourceLocation Loc,
                                       llvm::Value *OutlinedFn,
                                       ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::outs() << "OCL::emitTeamsCall\n";
  emitMasterFooter();
  llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
  OutlinedFnArgs.push_back(
      llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(
          CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  OutlinedFnArgs.push_back(
      llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(
          CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());
  CGF.EmitCallOrInvoke(OutlinedFn, OutlinedFnArgs);
}

void CGOpenMPRuntimeOCL::emitMasterRegion(CodeGenFunction &CGF,
                                          const RegionCodeGenTy &MasterOpGen,
                                          SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::outs() << "OCL::emitMasterRegion\n";
  emitMasterHeader(CGF);
  emitInlinedDirective(CGF, OMPD_master, MasterOpGen);
  emitMasterFooter();
}

void CGOpenMPRuntimeOCL::emitBarrierCall(CodeGenFunction &CGF,
                                         SourceLocation Loc,
                                         OpenMPDirectiveKind Kind,
                                         bool EmitChecks,
                                         bool ForceSimpleCall) {
    llvm::errs() << "OCL::emitBarrierCall\n";
}

void CGOpenMPRuntimeOCL::emitForStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc, OpenMPDirectiveKind DKind,
    const OpenMPScheduleTy &ScheduleKind,
    const CGOpenMPRuntime::StaticRTInput &Values) {

  llvm::outs() << "OCL::EmitForStaticInit\n";
}

void CGOpenMPRuntimeOCL::emitForStaticFinish(CodeGenFunction &CGF,
                                             SourceLocation Loc,
                                             bool CoalescedDistSchedule) {

  llvm::outs() << "OCL::EmitForStaticFinish\n";
}

void CGOpenMPRuntimeOCL::emitForDispatchInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    const OpenMPScheduleTy &ScheduleKind, unsigned IVSize, bool IVSigned,
    bool Ordered, llvm::Value *LB, llvm::Value *UB, llvm::Value *Chunk) {
  llvm::errs()
      << "For opencl/spir target, dynamic dispatch is not supported.\n";
}

void CGOpenMPRuntimeOCL::emitForDispatchFinish(CodeGenFunction &CGF,
                                               const OMPLoopDirective &S,
                                               SourceLocation Loc,
                                               unsigned IVSize, bool IVSigned) {

   llvm::outs() << "CGOpenMPRuntimeOCL::emitForDispatchFinish\n";
}

void CGOpenMPRuntimeOCL::emitDistributeStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDistScheduleClauseKind SchedKind,
    const CGOpenMPRuntime::StaticRTInput &Values, bool Coalesced) {

  llvm::outs() << "OCL::emitDistributeStaticInit\n";
}

void CGOpenMPRuntimeOCL::emitNumThreadsClause(CodeGenFunction &CGF,
                                              llvm::Value *NumThreads,
                                              SourceLocation Loc) {

  llvm::outs() << "OCL::emitNumThreadsClause\n";
}

void CGOpenMPRuntimeOCL::emitNumTeamsClause(CodeGenFunction &CGF,
                                            const Expr *NumTeams,
                                            const Expr *ThreadLimit,
                                            SourceLocation Loc) {

  llvm::outs() << "OCL::emitNumTeamsClause\n";
}

bool CGOpenMPRuntimeOCL::isStaticNonchunked(
    OpenMPScheduleClauseKind ScheduleKind, bool Chunked) const {
  // In case of OMPC_SCHEDULE_unknown we return false
  // as we want to emit schedule(static,1) if no schedule clause is specified
  // more precise: the case below is the only one, for which we partition the
  // iteration space into chunks of equal size only to be conformant with the
  // specification
  return (ScheduleKind == OpenMPScheduleClauseKind::OMPC_SCHEDULE_static &&
          !Chunked);
}

bool CGOpenMPRuntimeOCL::isStaticNonchunked(
    OpenMPDistScheduleClauseKind ScheduleKind, bool Chunked) const {
  return ScheduleKind ==
             OpenMPDistScheduleClauseKind::OMPC_DIST_SCHEDULE_static &&
         !Chunked;
}

bool CGOpenMPRuntimeOCL::isDynamic(
    OpenMPScheduleClauseKind ScheduleKind) const {
  // we don't support real dynamic scheduling and just emit everything as static
  return false;
}

void CGOpenMPRuntimeOCL::emitInlinedDirective(CodeGenFunction &CGF,
                                              OpenMPDirectiveKind InnerKind,
                                              const RegionCodeGenTy &CodeGen,
                                              bool HasCancel) {

  llvm::outs() << "OCL::emitInLinedDirective\n";
}

void CGOpenMPRuntimeOCL::createOffloadEntry(llvm::Constant *ID,
                                            llvm::Constant *Addr, uint64_t Size,
                                            uint64_t Flags) {

  llvm::outs() << "OCL::createOffloadEntry\n";
}

static FieldDecl *addFieldToRecordDecl(ASTContext &C, DeclContext *DC,
                                       QualType FieldTy) {
  auto *Field = FieldDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Id=*/nullptr, FieldTy,
      C.getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
      /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
  Field->setAccess(AS_public);
  DC->addDecl(Field);
  return Field;
}


/// \brief Registers the context of a parallel region with the runtime
/// codegen implementation.
void CGOpenMPRuntimeOCL::registerParallelContext(
    CodeGenFunction &CGF, const OMPExecutableDirective &S) {
  CurrentParallelContext = CGF.CurCodeDecl;

  if (isOpenMPParallelDirective(S.getDirectiveKind()) ||
      isOpenMPSimdDirective(S.getDirectiveKind()))
    createDataSharingInfo(CGF);
}

void CGOpenMPRuntimeOCL::createDataSharingInfo(CodeGenFunction &CGF) {
    llvm::outs() << "OCL::createDataSharingInfo\n";
  auto &Context = CGF.CurCodeDecl;
  assert(Context &&
         "A parallel region is expected to be enclosed in a context.");

  ASTContext &C = CGM.getContext();

  if (DataSharingInfoMap.find(Context) != DataSharingInfoMap.end())
    return;

  auto &Info = DataSharingInfoMap[Context];

  // Get the body of the region. The region context is either a function or a
  // captured declaration.
  const Stmt *Body;
  if (auto *D = dyn_cast<CapturedDecl>(Context))
    Body = D->getBody();
  else
    Body = cast<FunctionDecl>(Context)->getBody();

  // Track if in this region one has to share
  // Find all the captures in all enclosed regions and obtain their captured
  // statements.
  SmallVector<const OMPExecutableDirective *, 8> CapturedDirs;
  SmallVector<const Stmt *, 64> WorkList;
  WorkList.push_back(Body);
  while (!WorkList.empty()) {
    const Stmt *CurStmt = WorkList.pop_back_val();
    if (!CurStmt)
      continue;

    // Is this a parallel region.
    if (auto *Dir = dyn_cast<OMPExecutableDirective>(CurStmt)) {
      if (isOpenMPParallelDirective(Dir->getDirectiveKind()) ||
          isOpenMPSimdDirective(Dir->getDirectiveKind())) {
        CapturedDirs.push_back(Dir);
      } else {
        if (Dir->hasAssociatedStmt()) {
          // Look into the associated statement of OpenMP directives.
          const CapturedStmt &CS =
              *cast<CapturedStmt>(Dir->getAssociatedStmt());
          CurStmt = CS.getCapturedStmt();

          WorkList.push_back(CurStmt);
        }
      }

      continue;
    }

    // Keep looking for other regions.
    WorkList.append(CurStmt->child_begin(), CurStmt->child_end());
  }

  assert(!CapturedDirs.empty() && "Expecting at least one parallel region!");

  // Scan the captured statements and generate a record to contain all the data
  // to be shared. Make sure we do not share the same thing twice.
  auto *SharedMasterRD =
      C.buildImplicitRecord("__openmp_spir_data_sharing_master_record");
  auto *SharedThreadRD =
      C.buildImplicitRecord("__openmp_spir_data_sharing_thread_record");
  SharedMasterRD->startDefinition();
  SharedThreadRD->startDefinition();

  llvm::SmallSet<const VarDecl *, 32> AlreadySharedDecls;
  ASTContext &Ctx = CGF.getContext();
  for (auto *Dir : CapturedDirs) {
    const auto *CS = cast<CapturedStmt>(Dir->getAssociatedStmt());
    if (Dir->hasClausesOfKind<OMPDependClause>() &&
        isOpenMPTargetExecutionDirective(Dir->getDirectiveKind()))
      CS = cast<CapturedStmt>(CS->getCapturedStmt());
    const RecordDecl *RD = CS->getCapturedRecordDecl();
    auto CurField = RD->field_begin();
    auto CurCap = CS->capture_begin();

    int idx = 0;
    // store the shared data types and names to be used by clang-pcg
    std::string incStr;
    llvm::raw_string_ostream Inc(incStr);
    // also, store the order of the shared data, same as buffer creation
    bufferNames.clear();

    for (CapturedStmt::const_capture_init_iterator I = CS->capture_init_begin(),
                                                   E = CS->capture_init_end();
         I != E; ++I, ++CurField, ++CurCap) {

      const VarDecl *CurVD = nullptr;
      QualType ElemTy;
      if (*I)
        ElemTy = (*I)->getType();

      // Track the data sharing type.
      DataSharingInfo::DataSharingType DST = DataSharingInfo::DST_Val;

      if (CurCap->capturesThis()) {
        // We use null to indicate 'this'.
        CurVD = nullptr;
      } else {
        // Get the variable that is initializing the capture.
        if (CurField->hasCapturedVLAType()) {
          auto VAT = CurField->getCapturedVLAType();
          ElemTy = Ctx.getSizeType();
          CurVD = ImplicitParamDecl::Create(Ctx, ElemTy,
              ImplicitParamDecl::Other);
          CGF.EmitVarDecl(*CurVD);
          CGF.Builder.CreateAlignedStore(CGF.Builder.CreateIntCast(
              CGF.VLASizeMap[VAT->getSizeExpr()], CGM.SizeTy, false),
              CGF.GetAddrOfLocalVar(CurVD).getPointer(),
              CGF.getPointerAlign());
          Info.addVLADecl(VAT->getSizeExpr(), CurVD);
        } else
          CurVD = CurCap->getCapturedVar();

        // If this is an OpenMP capture declaration, we need to look at the
        // original declaration.
        const VarDecl *OrigVD = CurVD;
        if (auto *OD = dyn_cast<OMPCapturedExprDecl>(OrigVD))
          OrigVD = cast<VarDecl>(
              cast<DeclRefExpr>(OD->getInit()->IgnoreImpCasts())->getDecl());

        if (idx > 0) Inc << ",\n";
        cast<Decl>(OrigVD)->print(Inc);
        bufferNames.push_back(std::pair<int, std::string>(idx, cast<NamedDecl>(OrigVD)->getNameAsString()));
        idx++;

        // If the variable does not have local storage it is always a reference.
        if (!OrigVD->hasLocalStorage())
          DST = DataSharingInfo::DST_Ref;
        else {
          // If we have an alloca for this variable, then we need to share the
          // storage too, not only the reference.
          auto *Val = cast<llvm::Instruction>(
              CGF.GetAddrOfLocalVar(OrigVD).getPointer());
          if (isa<llvm::LoadInst>(Val))
            DST = DataSharingInfo::DST_Ref;
          // If the variable is a bitcast, it is being encoded in a pointer
          // and should be treated as such.
          else if (isa<llvm::BitCastInst>(Val))
            DST = DataSharingInfo::DST_Cast;
          // If the variable is a reference, we also share it as is,
          // i.e., consider it a reference to something that can be shared.
          else if (OrigVD->getType()->isReferenceType())
            DST = DataSharingInfo::DST_Ref;
        }
      }

      // Do not insert the same declaration twice.
      if (AlreadySharedDecls.count(CurVD))
        continue;

      AlreadySharedDecls.insert(CurVD);
      Info.add(CurVD, DST);

      if (DST == DataSharingInfo::DST_Ref)
        ElemTy = C.getPointerType(ElemTy);

      addFieldToRecordDecl(C, SharedMasterRD, ElemTy);
      llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),
                           DS_Max_Blocks);
      auto QTy = C.getConstantArrayType(ElemTy, NumElems, ArrayType::Normal,
                                        /*IndexTypeQuals=*/0);
      addFieldToRecordDecl(C, SharedThreadRD, QTy);
    }

    // Save the data type & names of args.
    // They will be used by CGOpenMPRegionInfo::EmitBody
    CGF.CGM.OpenMPSupport.saveArgsStr(Inc.str());

    // Add loop bounds if required.
     DoOnSharedLoopBounds(*Dir, [&AlreadySharedDecls, &C, &Info, &SharedMasterRD,
                                &SharedThreadRD, &Dir,
                                &CGF](const VarDecl *LB, const VarDecl *UB) {
      // Do not insert the same declaration twice.
      if (AlreadySharedDecls.count(LB))
        return;

      // We assume that if the lower bound is not to be shared, the upper
      // bound is not shared as well.
      assert(!AlreadySharedDecls.count(UB) &&
             "Not expecting shared upper bound.");

      QualType ElemTy = LB->getType();

      // Bounds are shared by value.
      Info.add(LB, DataSharingInfo::DST_Val);
      Info.add(UB, DataSharingInfo::DST_Val);
      addFieldToRecordDecl(C, SharedMasterRD, ElemTy);
      addFieldToRecordDecl(C, SharedMasterRD, ElemTy);

      llvm::APInt NumElems(C.getTypeSize(C.getUIntPtrType()),
                           DS_Max_Blocks);
      auto QTy = C.getConstantArrayType(ElemTy, NumElems, ArrayType::Normal,
                                        /*IndexTypeQuals=*/0);
      addFieldToRecordDecl(C, SharedThreadRD, QTy);
      addFieldToRecordDecl(C, SharedThreadRD, QTy);

      // Emit the preinits to make sure the initializers are properly emitted.
      // FIXME: This is a hack - it won't work if declarations being shared
      // appear after the first parallel region.
      const OMPLoopDirective *L = cast<OMPLoopDirective>(Dir);
      if (auto *PreInits = cast_or_null<DeclStmt>(L->getPreInits()))
        for (const auto *I : PreInits->decls()) {
          CGF.EmitOMPHelperVar(cast<VarDecl>(I));
        }
    });
 }

  SharedMasterRD->completeDefinition();
  Info.MasterRecordType = C.getRecordType(SharedMasterRD);
  return;
}
