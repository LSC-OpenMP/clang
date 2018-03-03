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

  std::string KernelName;
  CGOpenMPRegionKind RegionKind;
  RegionCodeGenTy CodeGen;
  OpenMPDirectiveKind Kind;
  bool HasCancel;

  int createTempFile() {
    char *tmpName = strdup("kernel_XXXXXX");
    int fd = mkstemp(tmpName);
    KernelName = std::string(tmpName);
    return fd;
  }

  std::string getTempName() { return KernelName; }

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
  llvm::errs() << "OCL::EmitBody for\n";
  if (!CGF.HaveInsertPoint())
    return;
  /* CGF.EHStack.pushTerminate(); */

  llvm::raw_fd_ostream CLOS(CGOpenMPRegionInfo::createTempFile(), true);
  const std::string FileName = CGOpenMPRegionInfo::getTempName();

  CodeGenModule &CGM = CGF.CGM;
  std::string includeContents = CGM.OpenMPSupport.getIncludeStr();

  if (includeContents != "") {
    CLOS << includeContents << "\n";
  }

  S->printPretty(CLOS, nullptr, PrintingPolicy(CGF.getContext().getLangOpts()), 4);
  CLOS.close();

  /* Insert call to pcg: polyhedral code generation */
  /* std::string pcg = "clang-pcg " + FileName; */
  /* std::system(pcg.c_str()); */
  /* std::ifstream argFile(FileName); */
  /* if (argFile.is_open()) { */
  /*   /1* CodeGen(CGF); *1/ */
  /*   argFile.close(); */
  /* } */
  /* std::remove(FileName.c_str()); */

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

std::vector<std::pair<int, std::string>> vectorNames[8];
std::vector<std::pair<int, std::string>> scalarNames[8];

std::map<llvm::Value *, std::string> vectorMap;
std::map<std::string, llvm::Value *> scalarMap;

llvm::SmallVector<QualType, 16> deftypes;

bool dumpedDefType(const QualType *T) {
  for (ArrayRef<QualType>::iterator I = deftypes.begin(), E = deftypes.end();
       I != E; ++I) {
    if ((*I).getAsString() == T->getAsString())
      return true;
  }
  deftypes.push_back(*T);
  return false;
}

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

int GetTypeSizeInBits(llvm::Type *ty) {
  int typeSize = 0;
  if (ty->isSized()) {
    if (ty->isStructTy()) {
      int nElements = ty->getStructNumElements();
      for (int i = 0; i < nElements; i++) {
        llvm::Type *elTy = ty->getStructElementType(i);
        typeSize += GetTypeSizeInBits(elTy);
      }
    } else {
      typeSize = ty->getScalarSizeInBits();
    }
  } else {
    llvm_unreachable("Unsupported data type for reduction clause");
    typeSize = 32; /* assume i32 data type */
  }
  return typeSize;
}

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
    llvm::errs() << "OCL::emitParallelOCLOutlinedFunction\n";
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
  llvm::errs() << "OCL::Calling GenerateOpenMPCapturedStmtFunction\n";
  return CGF.GenerateOpenMPCapturedStmtFunction(
      *CS, /*UseCapturedArgumentsOnly=*/false, CaptureLevel, ImplicitParamStop);
}


CGOpenMPRuntimeOCL::CGOpenMPRuntimeOCL(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM) {
  llvm::errs() << "Creating OpenMPRuntimeOCL object\n";
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
    llvm::errs() << "OCL::targetHasInnerOutlinedFunction\n";
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
    llvm::errs() << "OCL::teamsHasInnerOutlinedFunction\n";
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
  llvm::errs() << "OCL::GenOpenCLArgMetadata\n";
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
CGOpenMPRuntimeOCL::createRuntimeFunction(OpenMPRTLFunctionSPIR Function) {
  // llvm::errs() << "CGOpenMPRuntimeOCL::createRuntimeFunction\n";
}

void CGOpenMPRuntimeOCL::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen, unsigned capturedLevel) {

  if (!IsOffloadEntry) // Nothing to do.
    return;

  llvm::errs() << "OCL::emitTargetOutlinedFunction {\n";
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

  llvm::errs() << "OCL::Dumping outlined function:\n";
  OutlinedFn->dump();

  /* TODO: Check if this is necessary */
  /* GenOpenCLArgMetadata(CS.getCapturedRecordDecl(), OutlinedFn, CGM); */

}

llvm::Value *CGOpenMPRuntimeOCL::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    unsigned CaptureLevel, unsigned ImplicitParamStop) {

  llvm::errs() << "OCL::emitParallelOutlinedFunction\n";
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

  llvm::errs() << "OCL::emitParallelCall\n";

  llvm::SmallVector<llvm::Value *, 16> RealArgs;
  RealArgs.push_back(
      llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(
          CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  RealArgs.push_back(
      llvm::ConstantPointerNull::get(CGF.CGM.Int32Ty->getPointerTo(
          CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic))));
  RealArgs.append(CapturedVars.begin(), CapturedVars.end());

  llvm::Function *F = cast<llvm::Function>(OutlinedFn);

  llvm::errs() << "Function Type:\n";
  F->getFunctionType()->dump();
  llvm::errs() << "  number of params: " << F->getFunctionType()->getNumParams()
               << "\n";

  emitMasterHeader(CGF);

  llvm::DenseMap<llvm::Value *, llvm::GlobalVariable *> sharedVariables;
  bool emitBarrier = false;
  unsigned addrSpaceLocal =
      CGM.getContext().getTargetAddressSpace(LangAS::opencl_local);
  unsigned addrSpaceGeneric =
      CGM.getContext().getTargetAddressSpace(LangAS::opencl_generic);

  emitMasterFooter();
  // memory fence to wait for stores to local mem:
  if (emitBarrier) {
    // call opencl write_mem_fence
    llvm::Value *arg[] = {
        CGF.Builder.getInt32(1 << 0)}; //CLK_LOCAL_MEM_FENCE   0x01
    CGF.EmitRuntimeCall(createRuntimeFunction(write_mem_fence), arg);
  }
  llvm::errs() << "Args:\n";
  for (llvm::Value *arg : RealArgs) {
    arg->dump();
  }
  // call outlined parallel function:
  CGF.EmitCallOrInvoke(OutlinedFn, RealArgs);

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
  if (emitBarrier) {
    // call opencl read_mem_fence
    llvm::Value *arg[] = {
        CGF.Builder.getInt32(1 << 0)}; //CLK_LOCAL_MEM_FENCE   0x01
    CGF.EmitRuntimeCall(createRuntimeFunction(read_mem_fence), arg);
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

  llvm::errs() << "OCL::emitTeamsOutlinedFunction\n";

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

  llvm::errs() << "OCL::emitTeamsCall {\n";
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
  llvm::errs() << "}\n";
}

void CGOpenMPRuntimeOCL::emitMasterRegion(CodeGenFunction &CGF,
                                          const RegionCodeGenTy &MasterOpGen,
                                          SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::errs() << "OCL::emitMasterRegion\n";
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
  if (!CGF.HaveInsertPoint())
    return;

  // call opencl work group barrier
  llvm::Value *arg[] = {
      CGF.Builder.getInt32(1 << 1)}; //CLK_GLOBAL_MEM_FENCE   0x02
  CGF.EmitRuntimeCall(createRuntimeFunction(work_group_barrier), arg);
}

void CGOpenMPRuntimeOCL::emitForStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc, OpenMPDirectiveKind DKind,
    const OpenMPScheduleTy &ScheduleKind,
    const CGOpenMPRuntime::StaticRTInput &Values) {

  llvm::errs() << "OCL::EmitForStaticInit\n";
}

void CGOpenMPRuntimeOCL::emitForStaticFinish(CodeGenFunction &CGF,
                                             SourceLocation Loc,
                                             bool CoalescedDistSchedule) {

  llvm::errs() << "OCL::EmitForStaticFinish\n";
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

   llvm::errs() << "CGOpenMPRuntimeOCL::emitForDispatchFinish\n";
}

void CGOpenMPRuntimeOCL::emitDistributeStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDistScheduleClauseKind SchedKind,
    const CGOpenMPRuntime::StaticRTInput &Values, bool Coalesced) {

  llvm::errs() << "OCL::emitDistributeStaticInit\n";
}

void CGOpenMPRuntimeOCL::emitNumThreadsClause(CodeGenFunction &CGF,
                                              llvm::Value *NumThreads,
                                              SourceLocation Loc) {

  llvm::errs() << "OCL::emitNumThreadsClause\n";
}

void CGOpenMPRuntimeOCL::emitNumTeamsClause(CodeGenFunction &CGF,
                                            const Expr *NumTeams,
                                            const Expr *ThreadLimit,
                                            SourceLocation Loc) {

  llvm::errs() << "OCL::emitNumTeamsClause\n";
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

  llvm::errs() << "OCL::emitInLinedDirective\n";
}

void CGOpenMPRuntimeOCL::createOffloadEntry(llvm::Constant *ID,
                                            llvm::Constant *Addr, uint64_t Size,
                                            uint64_t Flags) {

  llvm::errs() << "OCL::createOffloadEntry\n";
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
    llvm::errs() << "OCL::createDataSharingInfo\n";
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
    // start storing shared data into a string to be used by OpenCl kernel
    std::string incStr;
    llvm::raw_string_ostream Inc(incStr);

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

        idx++;
        Inc << idx << ": ";
        cast<Decl>(OrigVD)->print(Inc);
        Inc << "\n";

        // Debug
        llvm::errs() << "Function arg(" << idx << "):";
        cast<Decl>(OrigVD)->print(llvm::errs()); llvm::errs() << "\n";
        // end Debug

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

    CGF.CGM.OpenMPSupport.appendIncludeStr(Inc.str());

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


void CGOpenMPRuntimeOCL::EmitOMPLoopAsCLKernel(CodeGenFunction &CGF,
                                               const OMPLoopDirective &S) {

  llvm::errs() << "OCL::EmitOMPLoopAsCLKernel\n";

  bool verbose = true; /* change to false on release */
  bool tile = true;
  std::string tileSize = std::to_string(16); // default tile size
  if (auto *C = S.getSingleClause<OMPScheduleClause>()) {
    if (const auto *Ch = C->getChunkSize()) {
      // We only support chunk expression that folds to a constant
      llvm::APSInt Result;
      if (CGF.ConstantFoldsToSimpleInteger(Ch, Result)) {
        tileSize = Result.toString(10);
      }
    }
  }
  std::string ChunkSize = "--tile-size=" + tileSize + " ";
  bool vectorize = isOpenMPSimdDirective(S.getDirectiveKind());

  // Start creating a unique filename that refers to scop function
  llvm::raw_fd_ostream CLOS(CGM.OpenMPSupport.createTempFile(), true);
  const std::string FileName = CGM.OpenMPSupport.getTempName();
  const std::string clName = FileName + ".cl";
  const std::string AuxName = FileName + ".aux";

  // Add the basic c header files.
  CLOS << "#include <stdlib.h>\n";
  CLOS << "#include <stdint.h>\n";
  CLOS << "#include <math.h>\n\n";

  ArrayRef<llvm::Value *> MapClausePointerValues;
  ArrayRef<llvm::Value *> MapClauseSizeValues;
  ArrayRef<QualType> MapClauseQualTypes;

  // Fix-me
  // CGM.OpenMPSupport.getMapPos(MapClausePointerValues, MapClauseSizeValues,
  //                              MapClauseQualTypes);

  // Dump necessary typedefs in scope file
  deftypes.clear();
  for (ArrayRef<QualType>::iterator T = MapClauseQualTypes.begin(),
                                    E = MapClauseQualTypes.end();
       T != E; ++T) {
    QualType Q = (*T);
    if (!Q.isCanonical()) {
      const Type *ty = Q.getTypePtr();
      if (ty->isPointerType() || ty->isReferenceType()) {
        Q = ty->getPointeeType();
      }

      while (Q.getTypePtr()->isArrayType()) {
        Q = dyn_cast<ArrayType>(Q.getTypePtr())->getElementType();
      }

      if (!dumpedDefType(&Q)) {
        std::string defty = Q.getAsString();
        QualType B =
            ty->getCanonicalTypeInternal().getTypePtr()->getPointeeType();

        while (B.getTypePtr()->isArrayType()) {
          B = dyn_cast<ArrayType>(B.getTypePtr())->getElementType();
        }

        ty = B.getTypePtr();
        if (isa<RecordType>(ty)) {
          const auto *RT = dyn_cast<RecordType>(ty);
          RecordDecl *RD = RT->getDecl()->getDefinition();
          RD->print(CLOS);
          CLOS << ";\n";
        }

        if (B.isCanonical() && B.getAsString().compare(defty) != 0) {
          CLOS << "typedef " << B.getAsString() << " " << defty << ";\n";
        }
      }
    }
  }

  // Fix-me
  // CGM.OpenMPSupport.clearScopVars();
  // CGM.OpenMPSupport.clearKernelVars();
  // CGM.OpenMPSupport.clearLocalVars();
  scalarMap.clear();

  CLOS << "void foo (\n";

  int j = 0;
  bool needComma = false;
  for (ArrayRef<llvm::Value *>::iterator I = MapClausePointerValues.begin(),
                                         E = MapClausePointerValues.end();
       I != E; ++I) {

    llvm::Value *KV = dyn_cast<llvm::User>(*I)->getOperand(0);
    QualType QT = MapClauseQualTypes[j];
    std::string KName = vectorMap[KV];

    // Fix-me
    // CGM.OpenMPSupport.addScopVar(KV);
    // CGM.OpenMPSupport.addScopType(QT);
    // CGM.OpenMPSupport.addKernelVar(KV);
    // CGM.OpenMPSupport.addKernelType(QT);

    bool isPointer = false;
    const Type *ty = QT.getTypePtr();
    if (ty->isPointerType() || ty->isReferenceType()) {
      isPointer = true;
      QT = ty->getPointeeType();
    }
    while (QT.getTypePtr()->isArrayType()) {
      isPointer = true;
      QT = dyn_cast<ArrayType>(QT.getTypePtr())->getElementType();
    }

    j++;
    if (needComma)
      CLOS << ",\n";
    CLOS << "\t\t" << QT.getAsString();
    needComma = true;
    if (isPointer) {
      CLOS << " *" << KName;
    } else {
      CLOS << "  " << KName;
    }
  }
  CLOS << ") {\n";

  unsigned num_args = (unsigned)CGM.OpenMPSupport.getKernelVarSize();
  assert(num_args != 0 && "loop is not suitable to execute on GPUs");

  // Traverse the Body looking for all scalar
  // variables declared out of for scope and
  // generate value reference to pass to kernel
  // function
  Stmt *Body = S.getAssociatedStmt();
  if (auto *CS = dyn_cast_or_null<CapturedStmt>(Body)) {
    Body = CS->getCapturedStmt();
  }
  if (Body->getStmtClass() == Stmt::CompoundStmtClass) {
    auto *BS = cast<CompoundStmt>(Body);
    for (CompoundStmt::body_iterator I = BS->body_begin(), E = BS->body_end();
         I != E; ++I) {
      HandleStmts(*I, CLOS, num_args, false);
    }
  } else {
    HandleStmts(Body, CLOS, num_args, false);
  }

  CLOS << "\n#pragma scop\n";
  Body->printPretty(CLOS, nullptr,
                    PrintingPolicy(CGF.getContext().getLangOpts()), 4);
  CLOS << "\n#pragma endscop\n}\n";
  CLOS.close();

  int workSizes[8][3];
  int blockSizes[8][3];
  int kernelId, upperKernel = 0;
  int k = 0;
  std::vector<std::pair<int, std::string>> pName;

  if (!(tile || vectorize)) {
    std::remove(FileName.c_str());
  } else {
    // Change the temporary name to c name
    const std::string cName = FileName + ".c";
    rename(FileName.c_str(), cName.c_str());

    // Construct the pairs of <index, arg> that will
    // be passed to the kernels and sort it in
    // alphabetic order
    for (ArrayRef<llvm::Value *>::iterator I = MapClausePointerValues.begin(),
                                           E = MapClausePointerValues.end();
         I != E; ++I) {

      llvm::Value *PV = dyn_cast<llvm::User>(*I)->getOperand(0);
      pName.emplace_back(std::pair<int, std::string>(k, vectorMap[PV]));
      k++;
    }
    std::sort(pName.begin(), pName.end(), pairCompare);

    // Try to generate a (possible optimized) kernel
    // version using clang-pcg, a script that invoke
    // Polyhedral Codegen. Get the loop schedule
    // kind and chunk on pragmas:
    //   schedule(dynamic[,chunk]) set --tile-size=chunk
    //   schedule(static[,chunk]) also use no-reschedule
    //   schedule(auto) or none use --tile-size=16
    for (kernelId = 0; kernelId < 8; ++kernelId) {
      for (j = 0; j < 3; j++) {
        workSizes[kernelId][j] = 0;
        blockSizes[kernelId][j] = 0;
      }
      vectorNames[kernelId].clear();
      scalarNames[kernelId].clear();
    }

    // std::string ChunkSize = "--tile-size=" + tileSize + " ";
    bool hasScheduleStatic = false;
    for (ArrayRef<OMPClause *>::iterator I = S.clauses().begin(),
                                         E = S.clauses().end();
         I != E; ++I) {
      OpenMPClauseKind ckind = ((*I)->getClauseKind());
      if (ckind == OMPC_schedule) {
        auto *C = cast<OMPScheduleClause>(*I);
        OpenMPScheduleClauseKind ScheduleKind = C->getScheduleKind();
        if (ScheduleKind == OMPC_SCHEDULE_static ||
            ScheduleKind == OMPC_SCHEDULE_dynamic) {
          hasScheduleStatic = ScheduleKind == OMPC_SCHEDULE_static;
          Expr *CSExpr = C->getChunkSize();
          if (CSExpr) {
            llvm::APSInt Ch;
            if (CSExpr->EvaluateAsInt(Ch, CGM.getContext())) {
              ChunkSize = "--tile-size=" + Ch.toString(10) + " ";
            }
          }
        }
      }
    }

    if (!tile) {
      ChunkSize = "--no-reschedule --tile-size=1 "
                  "--no-shared-memory "
                  "--no-private-memory ";
    } else if (vectorize) {
      // Vector optimization use tile-size=4, the
      // preferred vector size for float. Also, turn
      // off the use of shared & private memories.
      ChunkSize = "--tile-size=4 "
                  "--no-shared-memory "
                  "--no-private-memory ";
    }

    std::string pcg;
    if (verbose) {
      pcg = "clang-pcg --verbose " + ChunkSize;
      if (hasScheduleStatic)
        pcg = pcg + "--no-reschedule ";
    } else {
      pcg = "clang-pcg " + ChunkSize;
      if (hasScheduleStatic)
        pcg = pcg + "--no-reschedule ";
    }

    const std::string polycg = pcg + cName;
    std::system(polycg.c_str());
    // verbose preserve temp files (for debug
    // purposes)
    if (!verbose) {
      const std::string rmCfile = "rm " + FileName + ".c";
      std::system(rmCfile.c_str());
      const std::string rmHfile = "rm " + FileName + "_host.c";
      std::system(rmHfile.c_str());
    }

    std::ifstream argFile(FileName);
    if (argFile.is_open()) {
      int kind, index;
      std::string arg_name;
      int last_KernelId = -1;
      while (argFile >> kernelId) {
        assert(kernelId < 8 && "Invalid kernel identifier");
        if (kernelId != last_KernelId) {
          last_KernelId = kernelId;
          argFile >> workSizes[kernelId][0] >> workSizes[kernelId][1] >>
              workSizes[kernelId][2];
          argFile >> kernelId;
          assert(kernelId == last_KernelId && "Invalid kernel structure");
          argFile >> blockSizes[kernelId][0] >> blockSizes[kernelId][1] >>
              blockSizes[kernelId][2];
          argFile >> kernelId;
          assert(kernelId == last_KernelId && "Invalid kernel structure");
        }
        argFile >> kind >> index >> arg_name;
        if (kind == 1) {
          vectorNames[kernelId].emplace_back(
              std::pair<int, std::string>(index, arg_name));
        } else if (kind == 2) {
          scalarNames[kernelId].emplace_back(
              std::pair<int, std::string>(index, arg_name));
        } else
          assert(false && "Invalid kernel structure");
      }
      upperKernel = kernelId;
      argFile.close();
    }

    if (!verbose)
      std::remove(FileName.c_str());
  }

  // Emit code to load the file that contain the kernels
  llvm::Value *Status = nullptr;
  llvm::Value *FileStr = CGF.Builder.CreateGlobalStringPtr(FileName);

  // CLgen control whether ppcg sucessfully generate the kernel
  // If ppcg returns workSizes = 0, meaning that optimization didn't work.
  bool CLgen = true;
  if (tile || vectorize)
    if (workSizes[0][0] != 0)
      CLgen = false;

  // Also, check if all scalars used to construct kernel was declared on host
  if (!CLgen) {
    for (kernelId = 0; kernelId < upperKernel; kernelId++) {
      for (std::vector<std::pair<int, std::string>>::iterator
               I = scalarNames[kernelId].begin(),
               E = scalarNames[kernelId].end();
           I != E; ++I) {
        if (scalarMap[(I)->second] == nullptr) {
          CLgen = true;
          break;
        }
      }
    }
  }

  // Look for CollapseNum
  bool hasCollapseClause = false;
  unsigned CollapseNum, loopNest;
  // If Collapse clause is not empty, get the
  // collapsedNum,
  for (ArrayRef<OMPClause *>::iterator I = S.clauses().begin(),
                                       E = S.clauses().end();
       I != E; ++I) {
    OpenMPClauseKind ckind = ((*I)->getClauseKind());
    if (ckind == OMPC_collapse) {
      hasCollapseClause = true;
      /* FIX-ME: CollapseNum =
       * getCollapsedNumberFromLoopDirective(&S); */
    }
  }

  // Look for number of loop nest.
  loopNest = GetNumNestedLoops(S);
  if (!hasCollapseClause)
    CollapseNum = loopNest;
  assert(loopNest <= 3 && "Invalid number of Loop nest.");
  assert(CollapseNum <= 3 && "Invalid number of Collapsed Loops.");

  // nCores is used only with CLgen, but must be declared outside it
  SmallVector<llvm::Value *, 3> nCores;

  // Generate kernel with vectorization ?
  if (vectorize) {
    const std::string vectorizer = "$LLVM_INCLUDE_PATH/vectorize/vectorize "
                                   "-silent " +
                                   clName;
    std::system(vectorizer.c_str());
    if (!verbose) {
      struct stat buffer;
      if (stat(AuxName.c_str(), &buffer) == 0) {
        std::remove(AuxName.c_str());
      }
    }
  }

  // Generate the spir-code ?
  if (CGM.getTriple().isSPIR()) {
    std::string tgtStr;
    tgtStr = CGM.getTriple().str();
    const std::string bcArg = "clang -cc1 -x cl -cl-std=CL1.2 "
                              "-fno-builtin "
                              "-emit-llvm-bc -triple " +
                              tgtStr +
                              " -include "
                              "$LLVM_INCLUDE_PATH/llvm/SpirTools/"
                              "opencl_spir.h "
                              "-ffp-contract=off -o " +
                              AuxName + " " + clName;
    std::system(bcArg.c_str());

    const std::string encodeStr =
        "spir-encoder " + AuxName + " " + FileName + ".bc";
    std::system(encodeStr.c_str());
    std::remove(AuxName.c_str());
  }

  if (!CLgen) {
    for (kernelId = 0; kernelId <= upperKernel; kernelId++) {
      llvm::Value *KernelStr = CGF.Builder.CreateGlobalStringPtr(
          FileName + std::to_string(kernelId));
      int wD;
      if (workSizes[kernelId][2] != 0)
        wD = 3;
      else if (workSizes[kernelId][1] != 0)
        wD = 2;
      else
        wD = 1;
      llvm::Value *workDim =
          CGF.Builder.CreateGlobalStringPtr(std::to_string(wD));
      llvm::Value *ws0 = CGF.Builder.CreateGlobalStringPtr(
          std::to_string(workSizes[kernelId][0]));
      llvm::Value *ws1 = CGF.Builder.CreateGlobalStringPtr(
          std::to_string(workSizes[kernelId][1]));
      llvm::Value *ws2 = CGF.Builder.CreateGlobalStringPtr(
          std::to_string(workSizes[kernelId][2]));
      llvm::Value *bl0 = CGF.Builder.CreateGlobalStringPtr(
          std::to_string(blockSizes[kernelId][0]));
      llvm::Value *bl1 = CGF.Builder.CreateGlobalStringPtr(
          std::to_string(blockSizes[kernelId][1]));
      llvm::Value *bl2 = CGF.Builder.CreateGlobalStringPtr(
          std::to_string(blockSizes[kernelId][2]));
    }
  }
}

/// \brief Emit code for the reduction directives for opencl/spir target.
void CGOpenMPRuntimeOCL::EmitOMPReductionAsCLKernel(CodeGenFunction &CGF,
                                                    const OMPLoopDirective &S) {
}

unsigned
CGOpenMPRuntimeOCL::GetNumNestedLoops(const OMPExecutableDirective &S) {
  return 0;
}

llvm::Value *
CGOpenMPRuntimeOCL::EmitHostParameters(ForStmt *FS, llvm::raw_fd_ostream &FOS,
                                       unsigned &num_args, bool Collapse,
                                       unsigned loopNest, unsigned lastLoop) {
  return nullptr;
}

llvm::Value *CGOpenMPRuntimeOCL::EmitSpirDeclRefLValue(const DeclRefExpr *D) {
  return nullptr;
}

void CGOpenMPRuntimeOCL::HandleStmts(Stmt *ST, llvm::raw_fd_ostream &CLOS,
                                     unsigned &num_args, bool CLgen) {}
