//===---- CGOpenMPRuntimeNVPTX.cpp - Interface to OpenMP NVPTX Runtimes ---===//
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

#include "CGOpenMPRuntimeSpark.h"
#include "CGCleanup.h"
#include "CGOpenMPRuntime.h"
#include "CodeGenFunction.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Frontend/FrontendDiagnostic.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"

using namespace clang;
using namespace CodeGen;

#define VERBOSE 1

namespace {
///
/// FIXME: This is stupid!
/// These class definitions are duplicated from CGOpenMPRuntime.cpp.  They
/// should instead be placed in the header file CGOpenMPRuntime.h and made
/// accessible to CGOpenMPRuntimeNVPTX.cpp.  Otherwise not only do we have
/// to duplicate code, but we have to ensure that both these definitions are
/// always the same.  This is a problem because a CGOpenMPRegionInfo object
/// from CGOpenMPRuntimeNVPTX.cpp is accessed in methods of CGOpenMPRuntime.cpp.
///
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
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind)
      : CGCapturedStmtInfo(CS, CR_OpenMP), RegionKind(RegionKind),
        CodeGen(CodeGen), Kind(Kind) {}

  CGOpenMPRegionInfo(const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind)
      : CGCapturedStmtInfo(CR_OpenMP), RegionKind(RegionKind), CodeGen(CodeGen),
        Kind(Kind) {}

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const = 0;

  /// \brief Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, const Stmt *S) override;

  CGOpenMPRegionKind getRegionKind() const { return RegionKind; }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }

  ~CGOpenMPRegionInfo() override = default;

protected:
  CGOpenMPRegionKind RegionKind;
  RegionCodeGenTy CodeGen;
  OpenMPDirectiveKind Kind;
};

void CGOpenMPRegionInfo::EmitBody(CodeGenFunction &CGF, const Stmt * /*S*/) {
  if (!CGF.HaveInsertPoint())
    return;
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
  {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    CodeGen(CGF);
  }
  CGF.EHStack.popTerminate();
}

/// \brief API for captured statement code generation in OpenMP constructs.
class CGOpenMPOutlinedRegionInfo final : public CGOpenMPRegionInfo {
public:
  CGOpenMPOutlinedRegionInfo(const CapturedStmt &CS, const VarDecl *ThreadIDVar,
                             const RegionCodeGenTy &CodeGen,
                             OpenMPDirectiveKind Kind)
      : CGOpenMPRegionInfo(CS, ParallelOutlinedRegion, CodeGen, Kind),
        ThreadIDVar(ThreadIDVar) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }

  /// \brief Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// \brief Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

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
} // namespace

CGOpenMPRuntimeSpark::CGOpenMPRuntimeSpark(CodeGenModule &CGM)
    : CGOpenMPRuntime(CGM) {
  llvm::errs() << "CGOpenMPRuntimeSpark\n";
  if (!CGM.getLangOpts().OpenMPIsDevice)
    llvm_unreachable("OpenMP Spark can only handle device code.");
}

void CGOpenMPRuntimeSpark::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen,
    unsigned CaptureLevel) {
  llvm::errs() << "CGOpenMPRuntimeSpark::emitTargetOutlinedFunction\n";
  assert(!ParentName.empty() && "Invalid target region parent name!");

  BuildJNITy();

  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen, 0);

  OutlinedFn->setCallingConv(llvm::CallingConv::C);
  OutlinedFn->addFnAttr(llvm::Attribute::NoUnwind);
  OutlinedFn->removeFnAttr(llvm::Attribute::OptimizeNone);

  EmitSparkJob();
}

llvm::Value *CGOpenMPRuntimeSpark::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    unsigned CaptureLevel, unsigned ImplicitParamStop) {

  llvm::errs() << "CGOpenMPRuntimeSpark::emitParallelOutlinedFunction\n";

  ShouldAccessJNIArgs = true;
  auto *MappingFn = GenerateMappingKernel(D);
  ShouldAccessJNIArgs = false;
  MappingFn->setLinkage(llvm::GlobalValue::WeakAnyLinkage);

  //  // Call to a parallel that is not combined with a teams or target
  //  // directive (non SPMD).
  //  // This could also be a nested 'parallel' in an SPMD region.
  //  const auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());
  //  if (D.hasClausesOfKind<OMPDependClause>() &&
  //      isOpenMPTargetExecutionDirective(D.getDirectiveKind()))
  //    CS = cast<CapturedStmt>(CS->getCapturedStmt());

  //  CodeGenFunction CGF(CGM, true);
  //  llvm::Function *OutlinedFun = nullptr;
  //  {
  //    // The outlined function takes as arguments the global_tid, bound_tid,
  //    // and a capture structure created from the captured variables.
  //    OutlinedFun = CGF.GenerateOpenMPCapturedStmtFunction(
  //        *CS, /*UseCapturedArgumentsOnly=*/false, CaptureLevel);
  //  }
  //  return OutlinedFun;

  const auto *CS = cast<CapturedStmt>(D.getAssociatedStmt());

  return emitFakeOpenMPFunction(*CS);
}

llvm::Function *CGOpenMPRuntimeSpark::emitFakeOpenMPFunction(
    const CapturedStmt &S, bool UseCapturedArgumentsOnly, unsigned CaptureLevel,
    unsigned ImplicitParamStop, bool NonAliasedMaps) {

  bool UIntPtrCastRequired = true;
  // Build the argument list.

  FunctionArgList Args;
  llvm::MapVector<const Decl *, std::pair<const VarDecl *, Address>> LocalAddrs;
  llvm::DenseMap<const Decl *, std::pair<const Expr *, llvm::Value *>> VLASizes;
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << "FakeFunction";

  CodeGenFunction CGF(CGM, true);

  const CapturedDecl *CD = S.getCapturedDecl();
  const RecordDecl *RD = S.getCapturedRecordDecl();
  assert(CD->hasBody() && "missing CapturedDecl body");

  // Build the argument list.
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &Ctx = CGM.getContext();

  CGF.GenerateOpenMPCapturedStmtParameters(S, UseCapturedArgumentsOnly,
                                           CaptureLevel, ImplicitParamStop,
                                           NonAliasedMaps, false, Args);

  FunctionArgList TargetArgs;
  if (UIntPtrCastRequired) {
    TargetArgs.append(Args.begin(), Args.end());
  } else {
    if (!UseCapturedArgumentsOnly) {
      TargetArgs.append(CD->param_begin(),
                        std::next(CD->param_begin(), ImplicitParamStop));
    }
    auto I = S.captures().begin();
    unsigned Cnt = 0;
    for (auto *FD : RD->fields()) {
      if (I->capturesVariable() || I->capturesVariableByCopy()) {
        VarDecl *CapVar = I->getCapturedVar();
        if (auto *C = dyn_cast<OMPCapturedExprDecl>(CapVar)) {
          // Check to see if the capture is to be a parameter in the
          // outlined function at this level.
          if (C->getCaptureLevel() < CaptureLevel) {
            ++I;
            continue;
          }
        }
      }
      TargetArgs.emplace_back(
          CGM.getOpenMPRuntime().translateParameter(FD, Args[Cnt]));
      ++I;
      ++Cnt;
    }
    TargetArgs.append(
        std::next(CD->param_begin(), CD->getContextParamPosition() + 1),
        CD->param_end());
  }

  // Create the function declaration.
  FunctionType::ExtInfo ExtInfo;
  const CGFunctionInfo &FuncInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(Ctx.VoidTy, TargetArgs);
  llvm::FunctionType *FuncLLVMTy = CGM.getTypes().GetFunctionType(FuncInfo);

  llvm::Function *F =
      llvm::Function::Create(FuncLLVMTy, llvm::GlobalValue::InternalLinkage,
                             Out.str(), &CGM.getModule());
  CGM.SetInternalFunctionAttributes(CD, F, FuncInfo);

  CGF.StartFunction(CD, Ctx.VoidTy, F, FuncInfo, TargetArgs, S.getLocStart(),
                    CD->getBody()->getLocStart());
  CGF.FinishFunction();
  return F;
}

void CGOpenMPRuntimeSpark::EmitSparkJob() {
  llvm::errs() << "EmitSparkJob\n";

  std::error_code EC;

  // char *tmpName = strdup("_kernel_spark_XXXXXX");
  llvm::raw_fd_ostream SPARK_FILE("_kernel_spark.scala", EC,
                                  llvm::sys::fs::F_Text);
  if (EC) {
    llvm::errs() << "Couldn't open kernel_spark file for dumping.\nError:"
                 << EC.message() << "\n";
    exit(1);
  }

  // Header
  SPARK_FILE << "package org.llvm.openmp\n\n"
             << "import java.nio.ByteBuffer\n\n";

  EmitSparkNativeKernel(SPARK_FILE);

  // Core
  SPARK_FILE << "object OmpKernel {"
             << "\n";

  SPARK_FILE << "  def main(args: Array[String]) {\n"
             << "    var i = 0 // used for loop index\n"
             << "    \n"
             << "    val info = new CloudInfo(args)\n"
             << "    val fs = new CloudFileSystem(info.fs, args(3), args(4))\n"
             << "    val at = AddressTable.create(fs)\n"
             << "    info.init(fs)\n"
             << "    \n"
             << "    import info.sqlContext.implicits._\n"
             << "    \n";

  EmitSparkInput(SPARK_FILE);

  for (auto it = SparkMappingFunctions.begin();
       it != SparkMappingFunctions.end(); it++) {
    llvm::errs() << "SparkMappingFunctions " << (*it)->Identifier << "\n";
    EmitSparkMapping(SPARK_FILE, **it, (it + 1) == SparkMappingFunctions.end());
  }

  EmitSparkOutput(SPARK_FILE);

  SPARK_FILE << "  }\n"
             << "\n"
             << "}\n";

  llvm::errs() << ">> Spark code generated...\n";
}

void CGOpenMPRuntimeSpark::EmitSparkNativeKernel(
    llvm::raw_fd_ostream &SPARK_FILE) {
  llvm::errs() << "EmitSparkNativeKernel\n";
  bool verbose = VERBOSE;

  int i;

  SPARK_FILE << "\n";
  SPARK_FILE << "import org.apache.spark.SparkFiles\n";
  SPARK_FILE << "class OmpKernel {\n";

  llvm::errs() << "Mapping Native\n";
  for (auto *info : SparkMappingFunctions) {

    llvm::errs() << "--- MappingFunction ID = " << info->Identifier << "\n";

    auto &OMPLoop = info->OMPDirective;

    llvm::errs() << "Native 1\n";
    llvm::errs() << "NbOutput = " << info->Outputs.size() << " + "
                 << info->InputsOutputs.size() << "\n";
    unsigned NbOutputs = info->Outputs.size() + info->InputsOutputs.size();

    llvm::errs() << "Native 2\n";
    SPARK_FILE << "  @native def mappingMethod" << info->Identifier << "(";
    i = 0;
    llvm::errs() << "Native 2.5\n";

    for (auto I : info->OMPDirective.counters()) {
      llvm::errs() << "Native 3\n";

      // Separator
      if (i != 0)
        SPARK_FILE << ", ";

      SPARK_FILE << "index" << i << ": Long, bound" << i << ": Long";
      i++;
    }
    llvm::errs() << "Native 3.5\n";

    i = 0;
    for (auto it = info->InVarUse.begin(); it != info->InVarUse.end();
         ++it, i++) {
      llvm::errs() << "Native 4\n";

      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    for (auto it = info->InOutVarUse.begin(); it != info->InOutVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    for (auto it = info->OutVarDef.begin(); it != info->OutVarDef.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    SPARK_FILE << ") : ";
    if (NbOutputs == 1)
      SPARK_FILE << "Array[Byte]";
    else {
      SPARK_FILE << "Tuple" << NbOutputs << "[Array[Byte]";
      for (unsigned i = 1; i < NbOutputs; i++)
        SPARK_FILE << ", Array[Byte]";
      SPARK_FILE << "]";
    }
    SPARK_FILE << "\n";

    llvm::errs() << "Mapping Native Loader\n";

    SPARK_FILE << "  def mapping" << info->Identifier << "(";
    i = 0;
    for (auto it = OMPLoop.counters().begin(); it != OMPLoop.counters().end();
         ++it, i++) {
      // Separator
      if (it != OMPLoop.counters().begin())
        SPARK_FILE << ", ";

      SPARK_FILE << "index" << i << ": Long, bound" << i << ": Long";
    }
    i = 0;
    for (auto it = info->InVarUse.begin(); it != info->InVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    for (auto it = info->InOutVarUse.begin(); it != info->InOutVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    for (auto it = info->OutVarDef.begin(); it != info->OutVarDef.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    SPARK_FILE << ") : ";
    if (NbOutputs == 1)
      SPARK_FILE << "Array[Byte]";
    else {
      SPARK_FILE << "Tuple" << NbOutputs << "[Array[Byte]";
      for (unsigned i = 1; i < NbOutputs; i++)
        SPARK_FILE << ", Array[Byte]";
      SPARK_FILE << "]";
    }
    SPARK_FILE << " = {\n";
    SPARK_FILE << "    NativeKernels.loadOnce()\n";
    SPARK_FILE << "    return mappingMethod" << info->Identifier << "(";
    i = 0;
    for (auto it = OMPLoop.counters().begin(); it != OMPLoop.counters().end();
         ++it, i++) {
      // Separator
      if (it != OMPLoop.counters().begin())
        SPARK_FILE << ", ";

      SPARK_FILE << "index" << i << ", bound" << i;
    }
    i = 0;
    for (auto it = info->InVarUse.begin(); it != info->InVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i;
    }
    for (auto it = info->InOutVarUse.begin(); it != info->InOutVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i;
    }
    for (auto it = info->OutVarDef.begin(); it != info->OutVarDef.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i;
    }
    SPARK_FILE << ")\n";
    SPARK_FILE << "  }\n\n";

    //    llvm::errs() << "Reduce Native\n";

    //    for (auto it = info.ReducedVar.begin(); it != info.ReducedVar.end();
    //    ++it) {
    //      SPARK_FILE << "  @native def reduceMethod" << (*it)->getName()
    //                 << info.Identifier
    //                 << "(n0 : Array[Byte], n1 : Array[Byte]) :
    //                 Array[Byte]\n\n";
    //    }
    //    for (auto it = info.ReducedVar.begin(); it != info.ReducedVar.end();
    //    ++it) {
    //      SPARK_FILE << "  def reduce" << (*it)->getName() << info.Identifier
    //                 << "(n0 : Array[Byte], n1 : Array[Byte]) : Array[Byte]";
    //      SPARK_FILE << " = {\n";
    //      SPARK_FILE << "    NativeKernels.loadOnce()\n";
    //      SPARK_FILE << "    return reduceMethod" << (*it)->getName()
    //                 << info.Identifier << "(n0, n1)\n";
    //      SPARK_FILE << "  }\n\n";
    //    }
  }
  SPARK_FILE << "}\n\n";

  llvm::errs() << "End Native Kernel\n";
}

class SparkExprPrinter : public ConstStmtVisitor<SparkExprPrinter> {

  llvm::raw_fd_ostream &SPARK_FILE;
  ASTContext &Context;
  CGOpenMPRuntimeSpark::OMPSparkMappingInfo &Info;
  std::string CntStr;

public:
  SparkExprPrinter(llvm::raw_fd_ostream &SPARK_FILE, ASTContext &Context,
                   CGOpenMPRuntimeSpark::OMPSparkMappingInfo &Info,
                   std::string CntStr)
      : SPARK_FILE(SPARK_FILE), Context(Context), Info(Info), CntStr(CntStr) {}

  void PrintExpr(const Expr *E) {
    if (E) {
      llvm::APSInt Value;
      bool isEvaluable = E->EvaluateAsInt(Value, Context);
      if (isEvaluable)
        SPARK_FILE << std::to_string(Value.getSExtValue());
      else
        Visit(E);
    } else
      SPARK_FILE << "<null expr>";
  }

  void VisitImplicitCastExpr(const ImplicitCastExpr *Node) {
    // No need to print anything, simply forward to the subexpression.
    PrintExpr(Node->getSubExpr());
  }

  void VisitParenExpr(const ParenExpr *Node) {
    SPARK_FILE << "(";
    PrintExpr(Node->getSubExpr());
    SPARK_FILE << ")";
  }

  void VisitBinaryOperator(const BinaryOperator *Node) {
    PrintExpr(Node->getLHS());
    SPARK_FILE << " " << BinaryOperator::getOpcodeStr(Node->getOpcode()) << " ";
    PrintExpr(Node->getRHS());
  }

  void VisitDeclRefExpr(const DeclRefExpr *Node) {
    const VarDecl *VD = dyn_cast<VarDecl>(Node->getDecl());
    if (Info.CounterInfo.find(VD) != Info.CounterInfo.end()) {
      SPARK_FILE << CntStr;
    } else {
      SPARK_FILE << "ByteBuffer.wrap(";
      SPARK_FILE << "__ompcloud_offload_" + VD->getName().str();
      // FIXME: How about long ?
      SPARK_FILE << ").order(java.nio.ByteOrder.LITTLE_ENDIAN).getInt";
    }
  }
};

std::string CGOpenMPRuntimeSpark::getSparkVarName(const ValueDecl *VD) {
  return "__ompcloud_offload_" + VD->getName().str();
}

void CGOpenMPRuntimeSpark::EmitSparkInput(llvm::raw_fd_ostream &SPARK_FILE) {
  llvm::errs() << "EmitSparkInput\n";

  bool verbose = VERBOSE;
  auto &IndexMap = OffloadingMapVarsIndex;
  auto &TypeMap = OffloadingMapVarsType;

  SPARK_FILE << "    // Read each input from the storage\n";
  for (auto it = IndexMap.begin(); it != IndexMap.end(); ++it) {
    const ValueDecl *VD = it->first;
    int OffloadId = IndexMap[VD];
    unsigned OffloadType = TypeMap[VD];
    bool NeedBcast = VD->getType()->isAnyPointerType();

    // Find the bit size of one element
    QualType VarType = VD->getType();

    while (VarType->isAnyPointerType()) {
      VarType = VarType->getPointeeType();
    }
    int64_t SizeInByte = CGM.getContext().getTypeSize(VarType) / 8;

    SPARK_FILE << "    val sizeOf_" << getSparkVarName(VD) << " = at.get("
               << OffloadId << ")\n";
    SPARK_FILE << "    val eltSizeOf_" << getSparkVarName(VD) << " = "
               << SizeInByte << "\n";

    if (OffloadType == OMPC_MAP_to ||
        OffloadType == (OMPC_MAP_to | OMPC_MAP_from)) {

      SPARK_FILE << "    var " << getSparkVarName(VD) << " = fs.read("
                 << OffloadId << ", sizeOf_" << getSparkVarName(VD) << ")\n";

    } else if (OffloadType == OMPC_MAP_from || OffloadType == OMPC_MAP_alloc) {
      SPARK_FILE << "    var " << getSparkVarName(VD)
                 << " = new Array[Byte](sizeOf_" << getSparkVarName(VD)
                 << ")\n";
    }

    if (verbose)
      SPARK_FILE << "    println(\"XXXX DEBUG XXXX SizeOf "
                 << getSparkVarName(VD) << "= \" + sizeOf_"
                 << getSparkVarName(VD) << ")\n";

    if (NeedBcast)
      SPARK_FILE << "    var " << getSparkVarName(VD)
                 << "_bcast = info.sc.broadcast(" << getSparkVarName(VD)
                 << ")\n";
  }

  SPARK_FILE << "    val _parallelism = info.getParallelism\n";

  SPARK_FILE << "\n";
}

void CGOpenMPRuntimeSpark::EmitSparkMapping(llvm::raw_fd_ostream &SPARK_FILE,
                                            OMPSparkMappingInfo &info,
                                            bool isLast) {
  llvm::errs() << "EmitSparkMapping\n";

  bool verbose = VERBOSE;
  auto &IndexMap = OffloadingMapVarsIndex;
  auto &TypeMap = OffloadingMapVarsType;
  auto &OMPLoop = info.OMPDirective;
  unsigned MappingId = info.Identifier;
  SparkExprPrinter MappingPrinter(SPARK_FILE, CGM.getContext(), info,
                                  "x.toInt");

  SPARK_FILE << "    // omp parallel for\n";

  SPARK_FILE << "    // 1 - Generate RDDs of index\n";
  int NbIndex = 0;

  llvm::errs() << "Print bound, index and blocksize variables\n";

  for (auto it = OMPLoop.counters().begin(); it != OMPLoop.counters().end();
       ++it) {
    if (DeclRefExpr *CntExpr = dyn_cast_or_null<DeclRefExpr>(*it))
      if (const VarDecl *CntDecl =
              dyn_cast_or_null<VarDecl>(CntExpr->getDecl())) {
        auto it_init = OMPLoop.counter_inits().begin();
        auto it_step = OMPLoop.counter_steps().begin();
        auto it_num = OMPLoop.counter_inits().begin();

        llvm::errs() << "TEST 1\n";

        CntDecl->dump();
        const Expr *Init = *it_init;
        const Expr *Num = *it_num;
        const Expr *Step = *it_step;

        llvm::errs() << "TEST 2\n";

        SPARK_FILE << "    val bound_" << MappingId << "_" << NbIndex << " = ";
        MappingPrinter.PrintExpr(Init);
        SPARK_FILE << " + ";
        MappingPrinter.PrintExpr(Step);
        SPARK_FILE << " * ";
        MappingPrinter.PrintExpr(Num);
        SPARK_FILE << ".toLong\n";

        SPARK_FILE << "    val blockSize_" << MappingId << "_" << NbIndex
                   << " = ((bound_" << MappingId << "_" << NbIndex
                   << ").toFloat/_parallelism).floor.toLong\n";

        llvm::errs() << "TEST 3\n";

        SPARK_FILE << "    val index_" << MappingId << "_" << NbIndex << " = (";
        MappingPrinter.PrintExpr(Init);
        SPARK_FILE << ".toLong to bound_" << MappingId << "_" << NbIndex;
        SPARK_FILE << " by blockSize_" << MappingId << "_" << NbIndex << ")";
        SPARK_FILE << " // Index " << CntDecl->getName() << "\n";

        llvm::errs() << "TEST 4\n";

        if (verbose) {
          SPARK_FILE << "    println(\"XXXX DEBUG XXXX blockSize = "
                        "\" + blockSize_"
                     << MappingId << "_" << NbIndex << ")\n";
          SPARK_FILE << "    println(\"XXXX DEBUG XXXX bound = \" + bound_"
                     << MappingId << "_" << NbIndex << ")\n";
        }
        NbIndex++;
        it_init++, it_step++, it_num++;
      }
  }

  llvm::errs() << "Print construction of input RDDs\n";

  // We need to explicitly create Tuple1 if there is no ranged input.
  int NumberOfRangedInput = 0;
  for (auto it = info.InVarUse.begin(); it != info.InVarUse.end(); ++it)
    if (const OMPArraySectionExpr *Range = info.RangedVar[it->first])
      NumberOfRangedInput++;
  for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end(); ++it)
    if (const OMPArraySectionExpr *Range = info.RangedVar[it->first])
      NumberOfRangedInput++;
  for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end(); ++it)
    if (const OMPArraySectionExpr *Range = info.RangedVar[it->first])
      NumberOfRangedInput++;

  SparkExprPrinter InputStartRangePrinter(SPARK_FILE, CGM.getContext(), info,
                                          "x.toInt");
  SparkExprPrinter InputEndRangePrinter(
      SPARK_FILE, CGM.getContext(), info,
      "x.toInt + blockSize_" + std::to_string(MappingId) + "_0.toInt");

  SPARK_FILE << "    val index_" << MappingId << " = index_" << MappingId
             << "_0";
  for (int i = 1; i < NbIndex; i++) {
    SPARK_FILE << ".cartesian(index_" << MappingId << "_" << i << ")";
  }
  SPARK_FILE << ".map{ x => ";

  if (NumberOfRangedInput == 0) {
    SPARK_FILE << "Tuple1(x)";
  } else {
    SPARK_FILE << "(x";
    for (auto it = info.InVarUse.begin(); it != info.InVarUse.end(); ++it) {
      const VarDecl *VD = it->first;
      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        // Separator
        SPARK_FILE << ", ";
        SPARK_FILE << getSparkVarName(VD);
        SPARK_FILE << ".slice((";
        InputStartRangePrinter.PrintExpr(Range->getLowerBound());
        SPARK_FILE << ") * eltSizeOf_" << getSparkVarName(VD) << ", Math.min((";
        InputEndRangePrinter.PrintExpr(Range->getLength());
        SPARK_FILE << ") * eltSizeOf_" << getSparkVarName(VD) << ", sizeOf_"
                   << getSparkVarName(VD) << "))";
      }
    }
    for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end();
         ++it) {
      const VarDecl *VD = it->first;
      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        // Separator
        SPARK_FILE << ", ";
        SPARK_FILE << getSparkVarName(VD);
        SPARK_FILE << ".slice((";
        InputStartRangePrinter.PrintExpr(Range->getLowerBound());
        SPARK_FILE << ") * eltSizeOf_" << getSparkVarName(VD) << ", Math.min((";
        InputEndRangePrinter.PrintExpr(Range->getLength());
        SPARK_FILE << ") * eltSizeOf_" << getSparkVarName(VD) << ", sizeOf_"
                   << getSparkVarName(VD) << "))";
      }
    }
    for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end(); ++it) {
      const VarDecl *VD = it->first;
      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        // Separator
        SPARK_FILE << ", ";
        SPARK_FILE << getSparkVarName(VD);
        SPARK_FILE << ".slice((";
        InputStartRangePrinter.PrintExpr(Range->getLowerBound());
        SPARK_FILE << ") * eltSizeOf_" << getSparkVarName(VD) << ", Math.min((";
        InputEndRangePrinter.PrintExpr(Range->getLength());
        SPARK_FILE << ") * eltSizeOf_" << getSparkVarName(VD) << ", sizeOf_"
                   << getSparkVarName(VD) << "))";
      }
    }
    SPARK_FILE << ")";
  }
  SPARK_FILE << "}.toDS()\n"; // FIXME: Inverse with more indexes

  llvm::errs() << "Print mapping operations \n";

  SPARK_FILE << "    // 2 - Perform Map operations\n";
  SPARK_FILE << "    val mapres_" << MappingId << " = index_" << MappingId
             << ".map{ x => (x._1, new OmpKernel().mapping" << MappingId << "(";

  // Assign each argument according to its type
  // x = (index, sliceOfInput1, sliceOfInput2, ...)
  int i = 1;
  NbIndex = 0;
  for (auto it = info.CounterUse.begin(); it != info.CounterUse.end(); ++it) {
    // Separator
    if (it != info.CounterUse.begin())
      SPARK_FILE << ", ";
    SPARK_FILE << "x._" << i << ", Math.min(x._" << i << "+blockSize_"
               << MappingId << "_" << NbIndex << "-1, bound_" << MappingId
               << "_" << NbIndex << "-1)";
    i++;
  }

  for (auto it = info.InVarUse.begin(); it != info.InVarUse.end(); ++it) {
    const VarDecl *VD = it->first;
    bool NeedBcast = VD->getType()->isAnyPointerType();
    // Separator
    SPARK_FILE << ", ";
    if (const OMPArraySectionExpr *Range = info.RangedVar[VD])
      SPARK_FILE << "x._" << i++;
    else if (NeedBcast)
      SPARK_FILE << getSparkVarName(VD) << "_bcast.value";
    else
      SPARK_FILE << getSparkVarName(VD) << ".clone";
  }
  for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end(); ++it) {
    const VarDecl *VD = it->first;
    bool NeedBcast = VD->getType()->isAnyPointerType();
    // Separator
    SPARK_FILE << ", ";
    if (const OMPArraySectionExpr *Range = info.RangedVar[VD])
      SPARK_FILE << "x._" << i++;
    else if (NeedBcast)
      // FIXME: Additional copy but avoid error when using multiple thread on
      // the same worker node
      SPARK_FILE << getSparkVarName(VD) << "_bcast.value.clone";
    else
      SPARK_FILE << getSparkVarName(VD) << ".clone";
  }
  for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end(); ++it) {
    const VarDecl *VD = it->first;
    bool NeedBcast = VD->getType()->isAnyPointerType();
    // Separator
    SPARK_FILE << ", ";
    if (const OMPArraySectionExpr *Range = info.RangedVar[VD])
      SPARK_FILE << "x._" << i++;
    else if (NeedBcast)
      // FIXME: Additional copy but avoid error when using multiple thread on
      // the same worker node
      SPARK_FILE << getSparkVarName(VD) << "_bcast.value.clone";
    else
      SPARK_FILE << getSparkVarName(VD) << ".clone";
  }

  SPARK_FILE << ")) }\n";

  unsigned NbOutputs = info.OutVarDef.size() + info.InOutVarUse.size();
  if (NbOutputs > 1) {
    SPARK_FILE << "    // cache not to perform the mapping for each output\n";
    SPARK_FILE << "    mapres_" << MappingId << ".cache\n";
  }

  llvm::errs() << "Print construction of the result\n";

  SPARK_FILE << "    // 3 - Merge back the results\n";

  i = 0;

  for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end(); ++it) {
    const VarDecl *VD = it->first;
    bool NeedBcast = VD->getType()->isAnyPointerType();
    const OMPArraySectionExpr *Range = info.RangedVar[VD];

    SPARK_FILE << "    ";
    if (Range)
      SPARK_FILE << "val " << getSparkVarName(VD) << "_tmp_" << MappingId;
    else
      SPARK_FILE << getSparkVarName(VD);
    SPARK_FILE << " = ";

    SPARK_FILE << "mapres_" << MappingId;

    if (NbOutputs == 1) {
      // 1 output -> return the result directly
    } else if (NbOutputs == 2 || NbOutputs == 3) {
      // 2 or 3 outputs -> extract each variable from the Tuple2 or Tuple3
      SPARK_FILE << ".map{ x => (x._1, x._2._" << i + 1 << ") }";
    } else {
      // More than 3 outputs -> extract each variable from the Collection
      SPARK_FILE << ".map{ x => (x._1, x._2(" << i << ")) }";
    }
    //    if (std::find(info.ReducedVar.begin(), info.ReducedVar.end(), VD) !=
    //        info.ReducedVar.end())
    //      SPARK_FILE << ".map{ x => x._2 }.reduce{(x, y) => new "
    //                    "OmpKernel().reduce"
    //                 << VD->getName() << MappingId << "(x, y)}";
    //    else
    if (Range)
      SPARK_FILE << ".collect()";
    else
      SPARK_FILE << ".map{ x => x._2 "
                    "}.repartition(info.getExecutorNumber.toInt).reduce{(x, y) "
                    "=> Util.bitor(x, y)}";
    SPARK_FILE << "\n";

    if (Range) {
      SparkExprPrinter RangePrinter(SPARK_FILE, CGM.getContext(), info,
                                    getSparkVarName(VD) + std::string("_tmp_") +
                                        std::to_string(MappingId) +
                                        std::string("(i)._1.toInt"));

      SPARK_FILE << "    " << getSparkVarName(VD)
                 << " = new Array[Byte](sizeOf_" << getSparkVarName(VD)
                 << ")\n";
      SPARK_FILE << "    i = 0\n";
      SPARK_FILE << "    while (i < " << getSparkVarName(VD) << "_tmp_"
                 << MappingId << ".length) {\n";
      SPARK_FILE << "      " << getSparkVarName(VD) << "_tmp_" << MappingId
                 << "(i)._2.copyToArray(" << getSparkVarName(VD) << ", (";
      RangePrinter.PrintExpr(Range->getLowerBound());
      SPARK_FILE << ") * eltSizeOf_" << getSparkVarName(VD) << ")\n"
                 << "      i += 1\n"
                 << "    }\n";
    }

    if (NeedBcast && !isLast)
      SPARK_FILE << "    " << getSparkVarName(VD) << "_bcast.destroy\n"
                 << "    " << getSparkVarName(VD)
                 << "_bcast = info.sc.broadcast(" << getSparkVarName(VD)
                 << ")\n";

    i++;
  }

  for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end(); ++it) {
    const VarDecl *VD = it->first;
    bool NeedBcast = VD->getType()->isAnyPointerType();
    const OMPArraySectionExpr *Range = info.RangedVar[VD];
    unsigned OffloadType = TypeMap[VD];

    if ((OffloadType == OMPC_MAP_alloc) && isLast)
      continue;

    SPARK_FILE << "    ";
    if (Range)
      SPARK_FILE << "val " << getSparkVarName(VD) << "_tmp_" << MappingId;
    else
      SPARK_FILE << getSparkVarName(VD);
    SPARK_FILE << " = ";

    SPARK_FILE << "mapres_" << MappingId;

    if (NbOutputs == 1) {
      // 1 output -> return the result directly
    } else if (NbOutputs == 2 || NbOutputs == 3) {
      // 2 or 3 outputs -> extract each variable from the Tuple2 or Tuple3
      SPARK_FILE << ".map{ x => (x._1, x._2._" << i + 1 << ") }";
    } else {
      // More than 3 outputs -> extract each variable from the Collection
      SPARK_FILE << ".map{ x => (x._1, x._2(" << i << ")) }";
    }
    //    if (std::find(info.ReducedVar.begin(), info.ReducedVar.end(), VD) !=
    //        info.ReducedVar.end())
    //      SPARK_FILE << ".map{ x => x._2 }.reduce{(x, y) => new "
    //                    "OmpKernel().reduce"
    //                 << VD->getName() << MappingId << "(x, y)}";
    //    else
    if (Range)
      SPARK_FILE << ".collect()";
    else
      SPARK_FILE << ".map{ x => x._2 "
                    "}.repartition(info.getExecutorNumber.toInt).reduce{(x, y) "
                    "=> Util.bitor(x, y)}";
    SPARK_FILE << "\n";

    if (Range) {
      SparkExprPrinter RangePrinter(SPARK_FILE, CGM.getContext(), info,
                                    getSparkVarName(VD) + std::string("_tmp_") +
                                        std::to_string(MappingId) +
                                        std::string("(i)._1.toInt"));

      SPARK_FILE << "    " << getSparkVarName(VD)
                 << " = new Array[Byte](sizeOf_" << getSparkVarName(VD)
                 << ")\n";
      SPARK_FILE << "    i = 0\n";
      SPARK_FILE << "    while (i < " << getSparkVarName(VD) << "_tmp_"
                 << MappingId << ".length) {\n";
      SPARK_FILE << "      " << getSparkVarName(VD) << "_tmp_" << MappingId
                 << "(i)._2.copyToArray(" << getSparkVarName(VD) << ", (";
      RangePrinter.PrintExpr(Range->getLowerBound());
      SPARK_FILE << ") * eltSizeOf_" << getSparkVarName(VD) << ")\n"
                 << "      i += 1\n"
                 << "    }\n";
    }

    if (NeedBcast && !isLast)
      SPARK_FILE << "    " << getSparkVarName(VD) << "_bcast.destroy\n"
                 << "    " << getSparkVarName(VD)
                 << "_bcast = info.sc.broadcast(" << getSparkVarName(VD)
                 << ")\n";

    i++;
  }
  SPARK_FILE << "\n";
}

void CGOpenMPRuntimeSpark::EmitSparkOutput(llvm::raw_fd_ostream &SPARK_FILE) {
  llvm::errs() << "EmitSparkOutput\n";

  auto &IndexMap = OffloadingMapVarsIndex;
  auto &TypeMap = OffloadingMapVarsType;

  SPARK_FILE << "    // Write the results back into the storage\n";

  for (auto it = IndexMap.begin(); it != IndexMap.end(); ++it) {
    const ValueDecl *VD = it->first;
    int OffloadId = IndexMap[VD];
    unsigned OffloadType = TypeMap[VD];

    if (OffloadType == OMPC_MAP_from ||
        OffloadType == (OMPC_MAP_to | OMPC_MAP_from)) {
      SPARK_FILE << "    fs.write(" << OffloadId << ", sizeOf_"
                 << getSparkVarName(VD) << ", " << getSparkVarName(VD) << ")\n";
    }
  }
}

Expr *CGOpenMPRuntimeSpark::ActOnIntegerConstant(SourceLocation Loc,
                                                 uint64_t Val) {
  unsigned IntSize = CGM.getContext().getTargetInfo().getIntWidth();
  return IntegerLiteral::Create(CGM.getContext(), llvm::APInt(IntSize, Val),
                                CGM.getContext().IntTy, Loc);
}

/// A StmtVisitor that propagates the raw counts through the AST and
/// records the count at statements where the value may change.
class FindKernelArguments : public RecursiveASTVisitor<FindKernelArguments> {

private:
  ArraySubscriptExpr *CurrArrayExpr;
  Expr *CurrArrayIndexExpr;

  llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 8>>
      MapVarToExpr;

  enum UseKind { Use, Def, UseDef };

  UseKind current_use;

public:
  CodeGenModule &CGM;
  CGOpenMPRuntimeSpark &SparkRuntime;
  bool verbose;

  llvm::SmallVector<VarDecl *, 8> LocalVars;

  CGOpenMPRuntimeSpark::OMPSparkMappingInfo *Info;

  FindKernelArguments(CodeGenModule &CGM, CGOpenMPRuntimeSpark &SparkRuntime,
                      CGOpenMPRuntimeSpark::OMPSparkMappingInfo *Info)
      : CGM(CGM), SparkRuntime(SparkRuntime), Info(Info) {
    verbose = VERBOSE;
    CurrArrayExpr = NULL;
    current_use = UseKind::UseDef;
  }

  void Explore(Stmt *S) {
    TraverseStmt(S);

    llvm::errs() << "Inputs =";
    for (auto In : Info->Inputs) {
      Info->InVarUse[In].append(MapVarToExpr[In].begin(),
                                MapVarToExpr[In].end());
      llvm::errs() << " " << In->getName();
    }
    llvm::errs() << "\n";
    llvm::errs() << "Outputs =";
    for (auto Out : Info->Outputs) {
      Info->OutVarDef[Out].append(MapVarToExpr[Out].begin(),
                                  MapVarToExpr[Out].end());
      llvm::errs() << " " << Out->getName();
    }
    llvm::errs() << "\n";
    llvm::errs() << "InputsOutputs =";
    for (auto InOut : Info->InputsOutputs) {
      Info->InOutVarUse[InOut].append(MapVarToExpr[InOut].begin(),
                                      MapVarToExpr[InOut].end());
      llvm::errs() << " " << InOut->getName();
    }
    llvm::errs() << "\n";
  }
  /* // FIXME: Decomment
    bool VisitOMPMapClause(const OMPMapClause *C) {
      if (verbose)
        llvm::errs() << "PASS THROUGH MAP CLAUSE\n";

      int MapType;
      ArrayRef<const Expr *> Vars;
      ArrayRef<const Expr *> BaseAddrs;
      ArrayRef<const Expr *> Addrs;
      ArrayRef<const Expr *> Sizes;

      Vars = C->getVars();
      BaseAddrs = C->getWholeStartAddresses();
      Addrs = C->getCopyingStartAddresses();
      Sizes = C->getCopyingSizesEndAddresses();

      switch (C->getClauseKind()) {
      default:
        llvm_unreachable("Unknown map clause type!");
        break;
      case OMPC_MAP_unknown:
      case OMPC_MAP_tofrom:
        MapType = OpenMPOffloadMappingFlags::OMP_MAP_TO |
                  OpenMPOffloadMappingFlags::OMP_MAP_FROM;
        break;
      case OMPC_MAP_to:
        MapType = OpenMPOffloadMappingFlags::OMP_MAP_TO;
        break;
      case OMPC_MAP_from:
        MapType = OpenMPOffloadMappingFlags::OMP_MAP_FROM;
        break;
      case OMPC_MAP_alloc: // FIXME: alloc == private ?
        MapType = OpenMPOffloadMappingFlags::OMP_MAP_PRIVATE;
        break;
      case OMPC_MAP_release:
      case OMPC_MAP_delete:
        llvm::errs() << "ERROR OmpCloud: euuh something not supported\n";
        exit(EXIT_FAILURE);
      }

      assert(BaseAddrs.size() == Vars.size() &&
             BaseAddrs.size() == Addrs.size() &&
             BaseAddrs.size() == Sizes.size() && "Vars addresses mismatch!");

      for (unsigned i = 0; i < BaseAddrs.size(); ++i) {
        const Expr *Var = Vars[i];
        const DeclRefExpr *RefExpr = 0;

        if (const DeclRefExpr *T1 = dyn_cast<DeclRefExpr>(Var))
          RefExpr = T1;
        else {
          const ArraySubscriptExpr *T2 = dyn_cast<ArraySubscriptExpr>(Var);
          while (T2) {
            if (const ImplicitCastExpr *T3 =
                    dyn_cast<ImplicitCastExpr>(T2->getBase())) {
              if (const DeclRefExpr *T4 =
                      dyn_cast<DeclRefExpr>(T3->getSubExpr())) {
                RefExpr = T4;
                break;
              } else {
                T2 = dyn_cast<ArraySubscriptExpr>(T3->getSubExpr());
              }
            }
          }

          const OMPArraySectionExpr *Range =
              dyn_cast<OMPArraySectionExpr>(T2->getIdx());
          const VarDecl *VD = dyn_cast<VarDecl>(RefExpr->getDecl());

          Info->RangedVar[VD] = Range;
        }

        assert(RefExpr && "Unexpected expression in the map clause");
      }

      return true;
    }
    */

  bool TraverseOMPTargetDataDirective(OMPTargetDataDirective *S) {
    WalkUpFromOMPTargetDataDirective(S);
    Stmt *Body = S->getAssociatedStmt();

    if (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(Body))
      Body = CS->getCapturedStmt();

    bool SkippedContainers = false;
    while (!SkippedContainers) {
      if (AttributedStmt *AS = dyn_cast_or_null<AttributedStmt>(Body))
        Body = AS->getSubStmt();
      else if (CompoundStmt *CS = dyn_cast_or_null<CompoundStmt>(Body)) {
        if (CS->size() != 1) {
          SkippedContainers = true;
        } else {
          Body = CS->body_back();
        }
      } else
        SkippedContainers = true;
    }

    TraverseStmt(Body);

    return true;
  }

  bool VisitOMPTargetDataDirective(OMPTargetDataDirective *S) {
    if (verbose)
      llvm::errs() << "PASS THROUGH TARGET DATA\n";

    for (ArrayRef<OMPClause *>::iterator I = S->clauses().begin(),
                                         E = S->clauses().end();
         I != E; ++I)
      if (const OMPMapClause *C = static_cast<OMPMapClause *>(*I))
        ; // VisitOMPMapClause(C); // FIXME: reactivate

    return true;
  }

  bool VisitVarDecl(VarDecl *VD) {
    LocalVars.push_back(VD);
    return true;
  }

  bool VisitDeclRefExpr(DeclRefExpr *D) {

    if (const VarDecl *VD = dyn_cast<VarDecl>(D->getDecl())) {
      if (verbose)
        llvm::errs() << ">>> Found RefExpr = " << VD->getName() << " --> ";

      if (Info->CounterInfo.find(VD) != Info->CounterInfo.end()) {
        Info->CounterUse[VD].push_back(D);
        if (verbose)
          llvm::errs() << "is cnt\n";
        return true;
      }

      if (std::find(LocalVars.begin(), LocalVars.end(), VD) !=
          LocalVars.end()) {
        if (verbose)
          llvm::errs() << "is local\n";
        return true;
      }

      int MapType = SparkRuntime.getMapType(VD);
      if (MapType == -1) {
        if (VD->hasGlobalStorage()) {
          if (verbose)
            llvm::errs() << "is global\n";
          return true;
        }

        // FIXME: That should be detected before
        if (verbose)
          llvm::errs() << "assume input (not in clause)";
        SparkRuntime.addOffloadingMapVariable(
            VD, OpenMPOffloadMappingFlags::OMP_MAP_TO);
        MapType = SparkRuntime.getMapType(VD);
      }

      bool currInput = std::find(Info->Inputs.begin(), Info->Inputs.end(),
                                 VD) != Info->Inputs.end();
      bool currOutput = std::find(Info->Outputs.begin(), Info->Outputs.end(),
                                  VD) != Info->Outputs.end();
      bool currInputOutput =
          std::find(Info->InputsOutputs.begin(), Info->InputsOutputs.end(),
                    VD) != Info->InputsOutputs.end();

      MapVarToExpr[VD].push_back(D);

      if (current_use == Use) {
        if (verbose)
          llvm::errs() << " is Use";
        if (currInputOutput) {
          ;
        } else if (currOutput) {
          Info->Outputs.erase(VD);
          Info->InputsOutputs.insert(VD);
        } else {
          Info->Inputs.insert(VD);
        }
      } else if (current_use == Def) {
        if (verbose)
          llvm::errs() << " is Def";
        if (currInputOutput) {
          ;
        } else if (currInput) {
          Info->Inputs.erase(VD);
          Info->InputsOutputs.insert(VD);
        } else {
          Info->Outputs.insert(VD);
        }
      } else if (current_use == UseDef) {
        if (verbose)
          llvm::errs() << " is UseDef";
        Info->Inputs.erase(VD);
        Info->Outputs.erase(VD);
        Info->InputsOutputs.insert(VD);
      } else {
        if (verbose)
          llvm::errs() << " is Nothing ???";
      }

      // When variables are not fully broadcasted to the workers (internal
      // data map), index expressions are marked for codegen modification
      // if(CurrArrayExpr)
      if (const OMPArraySectionExpr *Range = Info->RangedVar[VD]) {
        Info->RangedArrayAccess[VD].push_back(CurrArrayExpr);
        if (verbose)
          llvm::errs() << " and ranged";
      }

      if (verbose)
        llvm::errs() << "\n";

      if (VD->hasGlobalStorage() && verbose)
        llvm::errs() << "Warning OmpCloud: " << VD->getName()
                     << " is global and in the map clause\n";
    }

    return true;
  }

  // A workaround to allow a redefinition of Traverse...Operator.
  bool TraverseStmt(Stmt *S) {
    if (!S)
      return true;

    switch (S->getStmtClass()) {
    case Stmt::CompoundAssignOperatorClass: {
      CompoundAssignOperator *CAO = cast<CompoundAssignOperator>(S);
      return TraverseCompoundAssignOperator(CAO);
    }
    case Stmt::UnaryOperatorClass:
      return TraverseUnaryOperator(cast<UnaryOperator>(S));
    case Stmt::BinaryOperatorClass:
      return TraverseBinaryOperator(cast<BinaryOperator>(S));
    default:
      return RecursiveASTVisitor::TraverseStmt(S);
    }
  }

  bool TraverseVarDecl(VarDecl *VD) {
    UseKind backup = current_use; // backup the usage
    VisitVarDecl(VD);
    current_use = Use;
    TraverseStmt(VD->getInit());
    current_use = backup;
    return true;
  }

  bool TraverseArraySubscriptExpr(ArraySubscriptExpr *A) {
    UseKind backup = current_use; // backup the usage

    CurrArrayExpr = A;
    CurrArrayIndexExpr = A->getIdx();
    TraverseStmt(A->getBase());

    CurrArrayExpr = nullptr;
    CurrArrayIndexExpr = nullptr;
    current_use = Use;
    TraverseStmt(A->getIdx());

    current_use = backup; // write back the usage to the current usage
    return true;
  }

  bool TraverseBinaryOperator(BinaryOperator *B) {
    UseKind backup = current_use; // backup the usage
    if (B->isAssignmentOp()) {
      current_use = Use;
      TraverseStmt(B->getRHS());
      current_use = Def;
      TraverseStmt(B->getLHS());
    } else {
      TraverseStmt(B->getLHS());
      current_use = Use;
      TraverseStmt(B->getRHS());
    }
    current_use = backup; // write back the usage to the current usage
    return true;
  }

  bool TraverseCompoundAssignOperator(CompoundAssignOperator *B) {
    UseKind backup = current_use; // backup the usage
    current_use = Use;
    TraverseStmt(B->getRHS());
    current_use = UseDef;
    TraverseStmt(B->getLHS());
    current_use = backup; // write back the usage to the current usage
    return true;
  }

  bool TraverseCallExpr(CallExpr *C) {
    UseKind backup = current_use; // backup the usage
    for (CallExpr::arg_iterator I = C->arg_begin(), E = C->arg_end(); I != E;
         ++I) {
      if ((*I)->getType()->isPointerType() ||
          (*I)->getType()->isReferenceType())
        current_use = UseDef;
      else
        current_use = Use;
      TraverseStmt(*I);
      current_use = backup;
    }
    return true;
  }

  bool TraverseUnaryOperator(UnaryOperator *U) {
    UseKind backup = current_use; // backup the usage
    switch (U->getOpcode()) {
    case UO_PostInc:
    case UO_PostDec:
    case UO_PreInc:
    case UO_PreDec:
      current_use = UseDef;
      break;
    case UO_Plus:
    case UO_Minus:
    case UO_Not:
    case UO_LNot:
      current_use = Use;
      break;
    case UO_AddrOf:
    case UO_Deref:
      // use the current_use
      break;
    default:
      // DEBUG("Operator " << UnaryOperator::getOpcodeStr(U->getOpcode()) <<
      // " not supported in def-use analysis");
      break;
    }
    TraverseStmt(U->getSubExpr());
    current_use = backup; // write back the usage to the current usage
    return true;
  }
};

void CGOpenMPRuntimeSpark::GenerateReductionKernel(
    const OMPReductionClause &C, const OMPExecutableDirective &S) {

  llvm::errs() << "GenerateReductionKernel\n";
  bool verbose = VERBOSE;

  // Create the mapping function
  llvm::Module *mod = &(CGM.getModule());

  // FIXME: should be the right function
  auto &info = *(SparkMappingFunctions.back());

  // Get JNI type
  llvm::StructType *StructTy_JNINativeInterface =
      mod->getTypeByName("struct.JNINativeInterface_");
  llvm::PointerType *PointerTy_JNINativeInterface =
      llvm::PointerType::get(StructTy_JNINativeInterface, 0);
  llvm::PointerType *PointerTy_1 =
      llvm::PointerType::get(PointerTy_JNINativeInterface, 0);

  llvm::StructType *StructTy_jobject = mod->getTypeByName("struct._jobject");
  llvm::PointerType *PointerTy_jobject =
      llvm::PointerType::get(StructTy_jobject, 0);

  for (OMPReductionClause::varlist_const_iterator I = C.varlist_begin(),
                                                  E = C.varlist_end();
       I != E; ++I) {

    const VarDecl *VD = cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl());

    // Initialize arguments
    std::vector<llvm::Type *> FuncTy_args;

    // Add compulsary arguments
    FuncTy_args.push_back(PointerTy_1);
    FuncTy_args.push_back(PointerTy_jobject);

    FuncTy_args.push_back(PointerTy_jobject);
    FuncTy_args.push_back(PointerTy_jobject);

    llvm::FunctionType *FnTy = llvm::FunctionType::get(
        /*Result=*/PointerTy_jobject,
        /*Params=*/FuncTy_args,
        /*isVarArg=*/false);

    std::string RedFnName = "Java_org_llvm_openmp_OmpKernel_reduceMethod" +
                            VD->getNameAsString() +
                            std::to_string(info.Identifier);

    if (verbose)
      llvm::errs() << RedFnName << "\n";

    llvm::Function *RedFn = llvm::Function::Create(
        FnTy, llvm::GlobalValue::ExternalLinkage, RedFnName, mod);

    // Initialize a new CodeGenFunction used to generate the reduction
    CodeGenFunction CGF(CGM, true);

    auto &Bld = CGF.Builder;

    assert(!CGF.CurFn &&
           "Do not use a CodeGenFunction object for more than one function");

    CGF.CurFn = RedFn;
    CGF.EnsureInsertPoint();

    // Generate useful type and constant
    llvm::PointerType *PointerTy_Int8 =
        llvm::PointerType::get(Bld.getInt8Ty(), 0);
    llvm::PointerType *PointerTy_Int32 =
        llvm::PointerType::get(Bld.getInt32Ty(), 0);

    llvm::ConstantInt *const_int32_0 = llvm::ConstantInt::get(
        CGM.getLLVMContext(), llvm::APInt(32, llvm::StringRef("0"), 10));

    llvm::ConstantPointerNull *const_ptr_null =
        llvm::ConstantPointerNull::get(PointerTy_Int8);

    // Find the bit size
    QualType VarType = VD->getType();
    int32_t SizeInByte = CGM.getContext().getTypeSize(VarType) / 8;
    llvm::ConstantInt *const_int32_typeSizeIntByte =
        llvm::ConstantInt::get(Bld.getInt32Ty(), SizeInByte);

    // Allocate and load compulsry JNI arguments
    llvm::Function::arg_iterator args = RedFn->arg_begin();
    args->setName("env");
    llvm::AllocaInst *alloca_env = Bld.CreateAlloca(PointerTy_1);
    Bld.CreateAlignedStore(&*args, alloca_env, CGM.getPointerAlign());
    args++;
    args->setName("obj");
    llvm::AllocaInst *alloca_obj = Bld.CreateAlloca(PointerTy_jobject);
    Bld.CreateAlignedStore(&*args, alloca_obj, CGM.getPointerAlign());
    args++;

    // FIXME: check alignment
    llvm::LoadInst *ptr_env =
        Bld.CreateAlignedLoad(alloca_env, CGM.getPointerAlign());
    llvm::LoadInst *ptr_ptr_env =
        Bld.CreateAlignedLoad(ptr_env, CGM.getPointerAlign());

    llvm::Value *ptr_gep_getelement =
        Bld.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 184);
    llvm::LoadInst *ptr_fn_getelement =
        Bld.CreateAlignedLoad(ptr_gep_getelement, CGM.getPointerAlign());

    llvm::Value *ptr_gep_releaseelement =
        Bld.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 192);
    llvm::LoadInst *ptr_fn_releaseelement =
        Bld.CreateAlignedLoad(ptr_gep_releaseelement, CGM.getPointerAlign());

    llvm::Value *ptr_gep_newbytearray =
        Bld.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 176);
    llvm::LoadInst *ptr_fn_newbytearray =
        Bld.CreateAlignedLoad(ptr_gep_newbytearray, CGM.getPointerAlign());

    // Allocate, load and cast the first operand
    llvm::AllocaInst *alloca_arg1 = Bld.CreateAlloca(PointerTy_jobject);
    Bld.CreateAlignedStore(&*args, alloca_arg1, CGM.getPointerAlign());

    llvm::LoadInst *ptr_arg1 =
        Bld.CreateAlignedLoad(alloca_arg1, CGM.getPointerAlign());
    std::vector<llvm::Value *> ptr_275_params;
    ptr_275_params.push_back(ptr_env);
    ptr_275_params.push_back(ptr_arg1);
    ptr_275_params.push_back(const_ptr_null);
    llvm::CallInst *ptr_275 = Bld.CreateCall(ptr_fn_getelement, ptr_275_params);

    llvm::Value *ptr_265 = Bld.CreateBitCast(ptr_275, PointerTy_Int32);
    llvm::Value *ptr_265_3 =
        Bld.CreateAlignedLoad(ptr_265, CGM.getPointerAlign());
    llvm::Value *ptr_265_3_cast =
        Bld.CreateBitCast(ptr_265_3, Bld.getInt32Ty());
    args++;

    // Allocate, load and cast the second operand
    llvm::AllocaInst *alloca_arg2 = Bld.CreateAlloca(PointerTy_jobject);
    Bld.CreateAlignedStore(&*args, alloca_arg2, CGM.getPointerAlign());

    llvm::LoadInst *ptr_arg2 =
        Bld.CreateAlignedLoad(alloca_arg2, CGM.getPointerAlign());
    std::vector<llvm::Value *> ptr_275_1_params;
    ptr_275_1_params.push_back(ptr_env);
    ptr_275_1_params.push_back(ptr_arg2);
    ptr_275_1_params.push_back(const_ptr_null);
    llvm::CallInst *ptr_275_1 =
        Bld.CreateCall(ptr_fn_getelement, ptr_275_1_params);

    llvm::Value *ptr_265_1 = Bld.CreateBitCast(ptr_275_1, PointerTy_Int32);
    llvm::Value *ptr_265_2 =
        Bld.CreateAlignedLoad(ptr_265_1, CGM.getPointerAlign());
    llvm::Value *ptr_265_2_cast =
        Bld.CreateBitCast(ptr_265_2, Bld.getInt32Ty());

    // Compute the reduction
    llvm::Value *res = nullptr;

    /* FIXME: generate operation
    switch (C.getOperator()) {
    case OMPC_REDUCTION_or:
    case OMPC_REDUCTION_bitor: {
      res = Bld.CreateOr(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_bitxor: {
      res = Bld.CreateXor(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_sub: {
      res = Bld.CreateSub(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_add: {
      res = Bld.CreateAdd(ptr_265_3_cast, ptr_265_2_cast, "", false,
                                  true);
      break;
    }
    case OMPC_REDUCTION_and:
    case OMPC_REDUCTION_bitand: {
      res = Bld.CreateAnd(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_mult: {
      res = Bld.CreateMul(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_min: {
      break;
    }
    case OMPC_REDUCTION_max: {
      // TODO: What about min/max op ?
      break;
    }
    case OMPC_REDUCTION_custom:
      llvm_unreachable("Custom initialization cannot be NULLed.");
    case OMPC_REDUCTION_unknown:
    case NUM_OPENMP_REDUCTION_OPERATORS:
      llvm_unreachable("Unkonwn operator kind.");
    }
    */

    // Allocate and store the result
    // FIXME: what about other type
    llvm::AllocaInst *alloca_res = Bld.CreateAlloca(Bld.getInt32Ty());
    Bld.CreateAlignedStore(res, alloca_res, CGM.getIntAlign());

    // Protect arg 1

    {
      std::vector<llvm::Value *> void_272_params;
      void_272_params.push_back(ptr_env);
      void_272_params.push_back(ptr_arg1);
      void_272_params.push_back(ptr_275);
      void_272_params.push_back(const_int32_0);
      Bld.CreateCall(ptr_fn_releaseelement, void_272_params);
    }

    // Protect arg 2

    {
      std::vector<llvm::Value *> void_272_params;
      void_272_params.push_back(ptr_env);
      void_272_params.push_back(ptr_arg2);
      void_272_params.push_back(ptr_275_1);
      void_272_params.push_back(const_int32_0);
      Bld.CreateCall(ptr_fn_releaseelement, void_272_params);
    }

    // Cast back the result to bit array
    std::vector<llvm::Value *> ptr_277_params;
    ptr_277_params.push_back(ptr_env);
    ptr_277_params.push_back(const_int32_typeSizeIntByte);
    llvm::CallInst *ptr_277 =
        Bld.CreateCall(ptr_fn_newbytearray, ptr_277_params);

    llvm::Value *ptr_279 =
        Bld.CreateConstInBoundsGEP2_32(nullptr, ptr_ptr_env, 0, 208);
    llvm::LoadInst *ptr_280 =
        Bld.CreateAlignedLoad(ptr_279, CGM.getPointerAlign());
    llvm::Value *ptr_res_cast =
        Bld.CreateBitCast(alloca_res, PointerTy_Int8, "");
    std::vector<llvm::Value *> void_281_params;
    void_281_params.push_back(ptr_env);
    void_281_params.push_back(ptr_277);
    void_281_params.push_back(const_int32_0);
    void_281_params.push_back(const_int32_typeSizeIntByte);
    void_281_params.push_back(ptr_res_cast);
    Bld.CreateCall(ptr_280, void_281_params);

    Bld.CreateRet(ptr_277);
  }
}

llvm::Function *
CGOpenMPRuntimeSpark::GenerateMappingKernel(const OMPExecutableDirective &S) {

  llvm::errs() << "GenerateMappingKernel\n";

  bool verbose = VERBOSE;

  auto &DL = CGM.getDataLayout();
  auto &Ctx = CGM.getContext();

  const OMPParallelForDirective &ForDirective =
      cast<OMPParallelForDirective>(S);

  //  for (ArrayRef<OMPClause *>::const_iterator I = S.clauses().begin(),
  //                                             E = S.clauses().end();
  //       I != E; ++I)
  //    if (*I && (*I)->getClauseKind() == OMPC_reduction)
  //      GenerateReductionKernel(cast<OMPReductionClause>(*(*I)), S);

  auto &typeMap = OffloadingMapVarsType;
  auto &indexMap = OffloadingMapVarsIndex;

  // FIXME: what about several functions
  OMPSparkMappingInfo info(ForDirective);
  SparkMappingFunctions.push_back(&info);

  llvm::errs() << "--- MappingFunction ID = " << info.Identifier << "\n";

  if (verbose)
    llvm::errs() << "Offloaded variables \n";
  for (auto iter = typeMap.begin(); iter != typeMap.end(); ++iter) {
    if (verbose)
      llvm::errs() << iter->first->getName() << " - " << iter->second << " - "
                   << indexMap[iter->first] << "\n";
  }

  const Stmt *Body = S.getAssociatedStmt();
  Stmt *LoopStmt;
  const ForStmt *For;

  if (const CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(Body))
    Body = CS->getCapturedStmt();

  for (unsigned I = 0; I < ForDirective.getCollapsedNumber(); ++I) {
    bool SkippedContainers = false;
    while (!SkippedContainers) {
      if (const AttributedStmt *AS = dyn_cast_or_null<AttributedStmt>(Body))
        Body = AS->getSubStmt();
      else if (const CompoundStmt *CS = dyn_cast_or_null<CompoundStmt>(Body)) {
        if (CS->size() != 1) {
          SkippedContainers = true;
        } else {
          Body = CS->body_back();
        }
      } else
        SkippedContainers = true;
    }
    For = dyn_cast_or_null<ForStmt>(Body);

    LoopStmt = const_cast<Stmt *>(Body);

    // Detect info of the loop counter

    if (DeclRefExpr *CntExpr =
            dyn_cast_or_null<DeclRefExpr>(ForDirective.counters().front()))
      if (const VarDecl *CntDecl =
              dyn_cast_or_null<VarDecl>(CntExpr->getDecl()))
        if (verbose) {
          llvm::errs() << "Find counter " << CntDecl->getName() << "\n";
          auto &CntInfo = info.CounterInfo[CntDecl];
          CntDecl->dump();
          CntInfo.push_back(ForDirective.counter_inits().front());
          llvm::errs() << "== INIT =\n";
          llvm::errs() << ForDirective.counter_inits().size() << "\n";
          ForDirective.counter_inits().front()->dump();
          llvm::errs() << "== COND =\n";
          ForDirective.counter_num_iterations().front()->dump();
          llvm::errs() << "== INC =\n";
          ForDirective.counter_steps().front()->dump();
        }

    Body = For->getBody();
  }

  // For->dump();

  // Create the mapping function

  // Initialize a new CodeGenFunction used to generate the mapping

  // Detect input/output expression from the loop body
  FindKernelArguments Finder(CGM, *this, &info);
  Finder.Explore(LoopStmt);

  llvm::errs() << "-- SIZE1 = " << info.Outputs.size() << "\n";
  llvm::errs() << "-- SIZE2 = " << SparkMappingFunctions.back()->Outputs.size() << "\n";


  // Initialize arguments
  FunctionArgList ArgList;

  // Add compulsary arguments
  // ImplicitParamDecl JNIEnvParam(Ctx, /*DC=*/nullptr, S.getLocStart(),
  //                              &Ctx.Idents.get("JNIEnv"), JNIEnvQTy,
  //                              ImplicitParamDecl::Other);
  ArgList.push_back(
      ImplicitParamDecl::Create(Ctx, JNIEnvQTy, ImplicitParamDecl::Other));
  ArgList.push_back(
      ImplicitParamDecl::Create(Ctx, jobjectQTy, ImplicitParamDecl::Other));

  for (auto it = info.CounterInfo.begin(); it != info.CounterInfo.end(); ++it) {
    ArgList.push_back(
        ImplicitParamDecl::Create(Ctx, jlongQTy, ImplicitParamDecl::Other));
    ArgList.push_back(
        ImplicitParamDecl::Create(Ctx, jlongQTy, ImplicitParamDecl::Other));
  }

  for (auto it = info.InVarUse.begin(); it != info.InVarUse.end(); ++it) {
    ArgList.push_back(
        ImplicitParamDecl::Create(Ctx, jobjectQTy, ImplicitParamDecl::Other));
  }

  for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end(); ++it) {
    ArgList.push_back(
        ImplicitParamDecl::Create(Ctx, jobjectQTy, ImplicitParamDecl::Other));
  }

  for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end(); ++it) {
    ArgList.push_back(
        ImplicitParamDecl::Create(Ctx, jobjectQTy, ImplicitParamDecl::Other));
  }

  // FIXME: maybe output should be ptr to jobjectQTy
  auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(jobjectQTy, ArgList);
  auto *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string FnName = "Java_org_llvm_openmp_OmpKernel_mappingMethod" +
                       std::to_string(info.Identifier);
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::ExternalLinkage,
                                    FnName, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(/*D=*/nullptr, Fn, FnInfo);
  CodeGenFunction CGF(CGM);
  CGF.disableDebugInfo();
  CGF.StartFunction(GlobalDecl(), jobjectQTy, Fn, FnInfo, ArgList);

  auto &Bld = CGF.Builder;

  // Generate useful type and constant"
  llvm::PointerType *PointerTy_Int8 =
      llvm::PointerType::get(Bld.getInt8Ty(), 0);
  llvm::ConstantPointerNull *const_ptr_null =
      llvm::ConstantPointerNull::get(PointerTy_Int8);

  // Global variable
  llvm::Value *const_ptr_init =
      Bld.CreateGlobalStringPtr("<init>", ".str.init");
  llvm::Value *const_ptr_tuple2 =
      Bld.CreateGlobalStringPtr("scala/Tuple2", ".str.tuple2");
  llvm::Value *const_ptr_tuple3 =
      Bld.CreateGlobalStringPtr("scala/Tuple3", ".str.tuple3");
  llvm::Value *const_ptr_tuple2_args = Bld.CreateGlobalStringPtr(
      "(Ljava/lang/Object;Ljava/lang/Object;)V", ".str.tuple2.args");
  llvm::Value *const_ptr_tuple3_args = Bld.CreateGlobalStringPtr(
      "(Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;)V",
      ".str.tuple3.args");

  // Create the variables to save the slot, stack, frame and active threads.
  auto ArgsIt = ArgList.begin();
  auto EnvAddr =
      CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                            (*ArgsIt)->getType()->getAs<PointerType>());
  ++ArgsIt;
  auto ObjAddr =
      CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                            (*ArgsIt)->getType()->getAs<PointerType>());
  ++ArgsIt;

  auto *ptr_env = EnvAddr.getPointer();

  // Keep values that have to be used for releasing
  llvm::SmallVector<std::pair<llvm::Value *, llvm::Value *>, 8> InputsToRelease,
      ScalarInputsToRelease, InOutputsToRelease, OutputsToRelease;
  // Keep values that have to be return
  llvm::SmallVector<llvm::Value *, 8> OutputsToReturn;

  Address AddrCntArg = Address::invalid();
  llvm::Value *CntVal;
  Address AddrCntBoundArg = Address::invalid();
  llvm::Value *CntBoundVal;

  if (info.CounterInfo.size() > 1) {
    llvm::errs() << "ERROR OmpCloud: Do not support more than 1 iteration "
                    "index for now.";
    exit(EXIT_FAILURE);
  }

  llvm::errs() << "GetCnt\n";

  for (auto it = info.CounterUse.begin(); it != info.CounterUse.end(); ++it) {
    const VarDecl *VD = it->first;
    QualType CntQTy = VD->getType();
    llvm::Type *CntTy = CGM.getTypes().ConvertType(CntQTy);

    // Get current value of the loop counter
    // FIXME: Should we cast ??
    AddrCntArg = CGF.GetAddrOfLocalVar(*ArgsIt);
    CntVal = CGF.EmitLoadOfScalar(AddrCntArg, /*Volatile=*/false, Ctx.IntTy,
                                  SourceLocation());
    Address AddrCnt = CGF.CreateDefaultAlignTempAlloca(CntTy, VD->getName());
    CGF.EmitStoreOfScalar(Bld.CreateIntCast(CntVal, CntTy, false), AddrCnt,
                          /*Volatile=*/false, CntQTy);

    ArgsIt++;

    // Get current value of the loop bound
    // FIXME: Should we cast ??
    AddrCntBoundArg = CGF.GetAddrOfLocalVar(*ArgsIt);
    CntBoundVal = CGF.EmitLoadOfScalar(AddrCntBoundArg, /*Volatile=*/false,
                                       Ctx.IntTy, SourceLocation());

    addOpenMPKernelArgVar(VD, AddrCnt.getPointer());
    ArgsIt++;
  }

  llvm::errs() << "GetJObjectFromInput\n";

  // Allocate, load and cast input variables (i.e. the arguments)
  for (auto it = info.InVarUse.begin(); it != info.InVarUse.end(); ++it) {
    const VarDecl *VD = it->first;
    QualType varType = VD->getType();
    llvm::Type *TyObject_arg = CGM.getTypes().ConvertType(varType);

    Address JObjectAddr =
        CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                              (*ArgsIt)->getType()->getAs<PointerType>());
    llvm::Value *JObjectPtr = JObjectAddr.getPointer();

    llvm::Value *valuePtr;

    if (!varType->isAnyPointerType()) {
      llvm::Value *ptr_load_arg =
          EmitJNIGetByteArrayElements(CGF, JObjectPtr, const_ptr_null);

      ScalarInputsToRelease.push_back(std::make_pair(JObjectPtr, ptr_load_arg));
      valuePtr = Bld.CreateBitCast(ptr_load_arg, TyObject_arg->getPointerTo());

    } else {
      JObjectPtr->dump();
      llvm::Value *ptr_load_arg = EmitJNIGetPrimitiveArrayCritical(
          CGF, JObjectPtr,
          llvm::ConstantPointerNull::get(jbooleanTy->getPointerTo()));

      InputsToRelease.push_back(std::make_pair(JObjectPtr, ptr_load_arg));

      llvm::Value *ptr_casted_arg =
          Bld.CreateBitCast(ptr_load_arg, TyObject_arg);

      valuePtr = Bld.CreateAlloca(TyObject_arg);
      Bld.CreateAlignedStore(ptr_casted_arg, valuePtr, CGM.getPointerAlign());

      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        llvm::Value *LowerBound = CGF.EmitScalarExpr(Range->getLowerBound());
        for (auto it = info.RangedArrayAccess[VD].begin();
             it != info.RangedArrayAccess[VD].end(); ++it)
          addOpenMPKernelArgRange(*it, LowerBound);
      }
    }

    addOpenMPKernelArgVar(VD, valuePtr);
    ++ArgsIt;
  }

  llvm::errs() << "GetJObjectFromInputOutput\n";

  // Allocate, load and cast input/output variables (i.e. the arguments)
  for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end(); ++it) {
    const VarDecl *VD = it->first;

    Address JObjectAddr =
        CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                              (*ArgsIt)->getType()->getAs<PointerType>());
    llvm::Value *JObjectPtr = JObjectAddr.getPointer();
    llvm::Value *ArrayPtr = EmitJNIGetPrimitiveArrayCritical(
        CGF, JObjectPtr,
        llvm::ConstantPointerNull::get(jbooleanTy->getPointerTo()));

    InOutputsToRelease.push_back(std::make_pair(JObjectPtr, ArrayPtr));
    OutputsToReturn.push_back(JObjectPtr);

    QualType varType = VD->getType();
    llvm::Type *TyObject_arg = CGM.getTypes().ConvertType(varType);

    llvm::Value *ValPtr;

    if (!varType->isAnyPointerType()) {
      if (verbose)
        llvm::errs() << ">Test< " << VD->getName() << " is scalar\n";

      ValPtr = Bld.CreateBitCast(ArrayPtr, TyObject_arg->getPointerTo());

    } else {
      llvm::Value *CastArrayPtr = Bld.CreateBitCast(ArrayPtr, TyObject_arg);

      ValPtr = Bld.CreateAlloca(TyObject_arg);

      Bld.CreateAlignedStore(CastArrayPtr, ValPtr,
                             DL.getPrefTypeAlignment(TyObject_arg));

      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        llvm::Value *LowerBound = CGF.EmitScalarExpr(Range->getLowerBound());
        for (auto it = info.RangedArrayAccess[VD].begin();
             it != info.RangedArrayAccess[VD].end(); ++it)
          addOpenMPKernelArgRange(*it, LowerBound);
      }
    }

    addOpenMPKernelArgVar(VD, ValPtr);
    ++ArgsIt;
  }

  llvm::errs() << "GetJObjectFromOutput\n";

  // Allocate output variables
  for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end(); ++it) {
    const VarDecl *VD = it->first;

    Address JObjectAddr =
        CGF.EmitLoadOfPointer(CGF.GetAddrOfLocalVar(*ArgsIt),
                              (*ArgsIt)->getType()->getAs<PointerType>());
    llvm::Value *JObjectPtr = JObjectAddr.getPointer();
    llvm::Value *ArrayPtr =
        EmitJNIGetPrimitiveArrayCritical(CGF, JObjectPtr, const_ptr_null);

    OutputsToRelease.push_back(std::make_pair(JObjectPtr, ArrayPtr));
    OutputsToReturn.push_back(JObjectPtr);

    QualType varType = VD->getType();
    llvm::Type *TyObject_arg = CGM.getTypes().ConvertType(varType);

    llvm::Value *ValPtr;

    if (!varType->isAnyPointerType()) {
      if (verbose)
        llvm::errs() << ">Test< " << VD->getName() << " is scalar\n";

      ValPtr = Bld.CreateBitCast(ArrayPtr, TyObject_arg->getPointerTo());

    } else {
      llvm::Value *CastArrayPtr = Bld.CreateBitCast(ArrayPtr, TyObject_arg);

      ValPtr = Bld.CreateAlloca(TyObject_arg);
      Bld.CreateAlignedStore(CastArrayPtr, ValPtr,
                             DL.getPrefTypeAlignment(TyObject_arg));

      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        llvm::Value *LowerBound = CGF.EmitScalarExpr(Range->getLowerBound());
        for (auto it = info.RangedArrayAccess[VD].begin();
             it != info.RangedArrayAccess[VD].end(); ++it)
          addOpenMPKernelArgRange(*it, LowerBound);
      }
    }

    addOpenMPKernelArgVar(VD, ValPtr);
    ++ArgsIt;
  }

  {
    // FIXME: CGM.OpenMPSupport.startSparkRegion();
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.

    // Generate kernel code
    // FIXME: Change the condition
    llvm::errs() << "Start emitting Loop\n";

    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.
    CodeGenFunction::RunCleanupsScope BodyScope(CGF);

    // Generate kernel code
    if (const CompoundStmt *S = dyn_cast<CompoundStmt>(Body))
      CGF.EmitCompoundStmtWithoutScope(*S);
    else
      CGF.EmitStmt(Body);

    llvm::errs() << "End of Loop\n";

    // FIXME: CGM.OpenMPSupport.stopSparkRegion();
  }

  llvm::errs() << "Release bytearrays\n";
  // Release JNI arrays
  for (auto it = InputsToRelease.begin(); it != InputsToRelease.end(); ++it)
    EmitJNIReleasePrimitiveArrayCritical(CGF, it->first, it->second,
                                         CGF.Builder.getInt32(2));
  for (auto it = ScalarInputsToRelease.begin();
       it != ScalarInputsToRelease.end(); ++it)
    EmitJNIReleaseByteArrayElements(CGF, it->first, it->second,
                                    CGF.Builder.getInt32(2));
  for (auto it = OutputsToRelease.begin(); it != OutputsToRelease.end(); ++it)
    EmitJNIReleasePrimitiveArrayCritical(CGF, it->first, it->second,
                                         CGF.Builder.getInt32(0));
  for (auto it = InOutputsToRelease.begin(); it != InOutputsToRelease.end();
       ++it)
    EmitJNIReleasePrimitiveArrayCritical(CGF, it->first, it->second,
                                         CGF.Builder.getInt32(0));

  llvm::errs() << "Building outputs\n";

  int NbOutputs = info.OutVarDef.size() + info.InOutVarUse.size();

  // Returning the output
  llvm::Value *retValue = nullptr;

  if (NbOutputs == 1) {
    // Just return the ByteArray
    retValue = OutputsToReturn.front();
  } else if (NbOutputs > 1) {
    retValue = EmitJNICreateNewTuple(CGF, OutputsToReturn);
  } else {
    llvm::errs() << "WARNING OmpCloud: There is not output variable\n";
  }

  llvm::errs() << "Storing result\n";
  Bld.CreateStore(retValue, CGF.ReturnValue);

  llvm::errs() << "Finishing function\n";
  CGF.FinishFunction();

  llvm::errs() << "END of Mapping function CodeGen\n";

  return Fn;
}
