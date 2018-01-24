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

void CGOpenMPRuntimeSpark::EmitSparkJob() {
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
    EmitSparkMapping(SPARK_FILE, **it, (it + 1) == SparkMappingFunctions.end());
  }

  EmitSparkOutput(SPARK_FILE);

  SPARK_FILE << "  }\n"
             << "\n"
             << "}\n";
}

void CGOpenMPRuntimeSpark::EmitSparkNativeKernel(
    llvm::raw_fd_ostream &SPARK_FILE) {
  bool verbose = VERBOSE;

  int i;

  SPARK_FILE << "\n";
  SPARK_FILE << "import org.apache.spark.SparkFiles\n";
  SPARK_FILE << "class OmpKernel {\n";

  for (auto it = SparkMappingFunctions.begin();
       it != SparkMappingFunctions.end(); it++) {
    OMPSparkMappingInfo &info = **it;

    unsigned NbOutputs = info.OutVarDef.size() + info.InOutVarUse.size();

    SPARK_FILE << "  @native def mappingMethod" << info.Identifier << "(";
    i = 0;
    for (auto it = info.CounterUse.begin(); it != info.CounterUse.end();
         ++it, i++) {
      // Separator
      if (it != info.CounterUse.begin())
        SPARK_FILE << ", ";

      SPARK_FILE << "index" << i << ": Long, bound" << i << ": Long";
    }
    i = 0;
    for (auto it = info.InVarUse.begin(); it != info.InVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end();
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
    SPARK_FILE << "  def mapping" << info.Identifier << "(";
    i = 0;
    for (auto it = info.CounterUse.begin(); it != info.CounterUse.end();
         ++it, i++) {
      // Separator
      if (it != info.CounterUse.begin())
        SPARK_FILE << ", ";

      SPARK_FILE << "index" << i << ": Long, bound" << i << ": Long";
    }
    i = 0;
    for (auto it = info.InVarUse.begin(); it != info.InVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i << ": Array[Byte]";
    }
    for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end();
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
    SPARK_FILE << "    return mappingMethod" << info.Identifier << "(";
    i = 0;
    for (auto it = info.CounterUse.begin(); it != info.CounterUse.end();
         ++it, i++) {
      // Separator
      if (it != info.CounterUse.begin())
        SPARK_FILE << ", ";

      SPARK_FILE << "index" << i << ", bound" << i;
    }
    i = 0;
    for (auto it = info.InVarUse.begin(); it != info.InVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i;
    }
    for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i;
    }
    for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end();
         ++it, i++) {
      // Separator
      SPARK_FILE << ", ";
      SPARK_FILE << "n" << i;
    }
    SPARK_FILE << ")\n";
    SPARK_FILE << "  }\n\n";

    for (auto it = info.ReducedVar.begin(); it != info.ReducedVar.end(); ++it) {
      SPARK_FILE << "  @native def reduceMethod" << (*it)->getName()
                 << info.Identifier
                 << "(n0 : Array[Byte], n1 : Array[Byte]) : Array[Byte]\n\n";
    }
    for (auto it = info.ReducedVar.begin(); it != info.ReducedVar.end(); ++it) {
      SPARK_FILE << "  def reduce" << (*it)->getName() << info.Identifier
                 << "(n0 : Array[Byte], n1 : Array[Byte]) : Array[Byte]";
      SPARK_FILE << " = {\n";
      SPARK_FILE << "    NativeKernels.loadOnce()\n";
      SPARK_FILE << "    return reduceMethod" << (*it)->getName()
                 << info.Identifier << "(n0, n1)\n";
      SPARK_FILE << "  }\n\n";
    }
  }
  SPARK_FILE << "}\n\n";
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
  bool verbose = VERBOSE;
  auto &IndexMap = OffloadingMapVarsIndex;
  auto &TypeMap = OffloadingMapVarsType;
  unsigned MappingId = info.Identifier;
  SparkExprPrinter MappingPrinter(SPARK_FILE, CGM.getContext(), info,
                                  "x.toInt");

  SPARK_FILE << "    // omp parallel for\n";

  SPARK_FILE << "    // 1 - Generate RDDs of index\n";
  int NbIndex = 0;

  for (auto it = info.CounterInfo.begin(); it != info.CounterInfo.end(); ++it) {
    const VarDecl *VarCnt = it->first;
    const Expr *Init = it->second[0];
    const Expr *Check = it->second[1];
    const Expr *Step = it->second[2];
    const Expr *CheckOp = it->second[3];

    const BinaryOperator *BO = cast<BinaryOperator>(CheckOp);

    SPARK_FILE << "    val bound_" << MappingId << "_" << NbIndex << " = ";
    MappingPrinter.PrintExpr(Check);
    SPARK_FILE << ".toLong\n";
    SPARK_FILE << "    val blockSize_" << MappingId << "_" << NbIndex
               << " = ((bound_" << MappingId << "_" << NbIndex
               << ").toFloat/_parallelism).floor.toLong\n";

    SPARK_FILE << "    val index_" << MappingId << "_" << NbIndex << " = (";
    MappingPrinter.PrintExpr(Init);
    SPARK_FILE << ".toLong to bound_" << MappingId << "_" << NbIndex;
    if (BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT) {
      SPARK_FILE << "-1";
    }
    SPARK_FILE << " by blockSize_" << MappingId << "_" << NbIndex << ")";
    SPARK_FILE << " // Index " << VarCnt->getName() << "\n";

    if (verbose) {
      SPARK_FILE << "    println(\"XXXX DEBUG XXXX blockSize = "
                    "\" + blockSize_"
                 << MappingId << "_" << NbIndex << ")\n";
      SPARK_FILE << "    println(\"XXXX DEBUG XXXX bound = \" + bound_"
                 << MappingId << "_" << NbIndex << ")\n";
    }
    NbIndex++;
  }

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
    if (std::find(info.ReducedVar.begin(), info.ReducedVar.end(), VD) !=
        info.ReducedVar.end())
      SPARK_FILE << ".map{ x => x._2 }.reduce{(x, y) => new "
                    "OmpKernel().reduce"
                 << VD->getName() << MappingId << "(x, y)}";
    else if (Range)
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
    if (std::find(info.ReducedVar.begin(), info.ReducedVar.end(), VD) !=
        info.ReducedVar.end())
      SPARK_FILE << ".map{ x => x._2 }.reduce{(x, y) => new "
                    "OmpKernel().reduce"
                 << VD->getName() << MappingId << "(x, y)}";
    else if (Range)
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

namespace {
class ForInitChecker : public StmtVisitor<ForInitChecker, Decl *> {
  class ForInitVarChecker : public StmtVisitor<ForInitVarChecker, Decl *> {
  public:
    VarDecl *VisitDeclRefExpr(DeclRefExpr *E) {
      return dyn_cast_or_null<VarDecl>(E->getDecl());
    }
    Decl *VisitStmt(Stmt *S) { return 0; }
    ForInitVarChecker() {}
  } VarChecker;
  Expr *InitValue;

public:
  Decl *VisitBinaryOperator(BinaryOperator *BO) {
    if (BO->getOpcode() != BO_Assign)
      return 0;

    InitValue = BO->getRHS();
    return VarChecker.Visit(BO->getLHS());
  }
  Decl *VisitDeclStmt(DeclStmt *S) {
    if (S->isSingleDecl()) {
      VarDecl *Var = dyn_cast_or_null<VarDecl>(S->getSingleDecl());
      if (Var && Var->hasInit()) {
        if (CXXConstructExpr *Init =
                dyn_cast<CXXConstructExpr>(Var->getInit())) {
          if (Init->getNumArgs() != 1)
            return 0;
          InitValue = Init->getArg(0);
        } else {
          InitValue = Var->getInit();
        }
        return Var;
      }
    }
    return 0;
  }
  Decl *VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
    switch (E->getOperator()) {
    case OO_Equal:
      InitValue = E->getArg(1);
      return VarChecker.Visit(E->getArg(0));
    default:
      break;
    }
    return 0;
  }
  Decl *VisitStmt(Stmt *S) { return 0; }
  ForInitChecker() : VarChecker(), InitValue(0) {}
  Expr *getInitValue() { return InitValue; }
};

class ForVarChecker : public StmtVisitor<ForVarChecker, bool> {
  Decl *InitVar;

public:
  bool VisitDeclRefExpr(DeclRefExpr *E) { return E->getDecl() == InitVar; }
  bool VisitImplicitCastExpr(ImplicitCastExpr *E) {
    return Visit(E->getSubExpr());
  }
  bool VisitStmt(Stmt *S) { return false; }
  ForVarChecker(Decl *D) : InitVar(D) {}
};

class ForTestChecker : public StmtVisitor<ForTestChecker, bool> {
  ForVarChecker VarChecker;
  Expr *CheckValue;
  bool IsLessOp;
  bool IsStrictOp;

public:
  bool VisitBinaryOperator(BinaryOperator *BO) {
    if (!BO->isRelationalOp())
      return false;
    if (VarChecker.Visit(BO->getLHS())) {
      CheckValue = BO->getRHS();
      IsLessOp = BO->getOpcode() == BO_LT || BO->getOpcode() == BO_LE;
      IsStrictOp = BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT;
    } else if (VarChecker.Visit(BO->getRHS())) {
      CheckValue = BO->getLHS();
      IsLessOp = BO->getOpcode() == BO_GT || BO->getOpcode() == BO_GE;
      IsStrictOp = BO->getOpcode() == BO_LT || BO->getOpcode() == BO_GT;
    }
    return CheckValue != 0;
  }
  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
    switch (E->getOperator()) {
    case OO_Greater:
    case OO_GreaterEqual:
    case OO_Less:
    case OO_LessEqual:
      break;
    default:
      return false;
    }
    if (E->getNumArgs() != 2)
      return false;

    if (VarChecker.Visit(E->getArg(0))) {
      CheckValue = E->getArg(1);
      IsLessOp =
          E->getOperator() == OO_Less || E->getOperator() == OO_LessEqual;
      IsStrictOp = E->getOperator() == OO_Less;
    } else if (VarChecker.Visit(E->getArg(1))) {
      CheckValue = E->getArg(0);
      IsLessOp =
          E->getOperator() == OO_Greater || E->getOperator() == OO_GreaterEqual;
      IsStrictOp = E->getOperator() == OO_Greater;
    }

    return CheckValue != 0;
  }
  bool VisitStmt(Stmt *S) { return false; }
  ForTestChecker(Decl *D)
      : VarChecker(D), CheckValue(0), IsLessOp(false), IsStrictOp(false) {}
  Expr *getCheckValue() { return CheckValue; }
  bool isLessOp() const { return IsLessOp; }
  bool isStrictOp() const { return IsStrictOp; }
};

class ForIncrChecker : public StmtVisitor<ForIncrChecker, bool> {
  ForVarChecker VarChecker;

  Expr *ActOnIntegerConstant(SourceLocation Loc, uint64_t Val) {
    unsigned IntSize = Context.getTargetInfo().getIntWidth();
    return IntegerLiteral::Create(Context, llvm::APInt(IntSize, Val),
                                  Context.IntTy, Loc);
  }

  class ForIncrExprChecker : public StmtVisitor<ForIncrExprChecker, bool> {
    ForVarChecker VarChecker;
    Expr *StepValue;
    bool IsIncrement;

  public:
    bool VisitBinaryOperator(BinaryOperator *BO) {
      if (!BO->isAdditiveOp())
        return false;
      if (BO->getOpcode() == BO_Add) {
        IsIncrement = true;
        if (VarChecker.Visit(BO->getLHS()))
          StepValue = BO->getRHS();
        else if (VarChecker.Visit(BO->getRHS()))
          StepValue = BO->getLHS();
        return StepValue != 0;
      }
      // BO_Sub
      if (VarChecker.Visit(BO->getLHS()))
        StepValue = BO->getRHS();
      return StepValue != 0;
    }
    bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
      switch (E->getOperator()) {
      case OO_Plus:
        IsIncrement = true;
        if (VarChecker.Visit(E->getArg(0)))
          StepValue = E->getArg(1);
        else if (VarChecker.Visit(E->getArg(1)))
          StepValue = E->getArg(0);
        return StepValue != 0;
      case OO_Minus:
        if (VarChecker.Visit(E->getArg(0)))
          StepValue = E->getArg(1);
        return StepValue != 0;
      default:
        return false;
      }
    }
    bool VisitStmt(Stmt *S) { return false; }
    ForIncrExprChecker(ForVarChecker &C)
        : VarChecker(C), StepValue(0), IsIncrement(false) {}
    Expr *getStepValue() { return StepValue; }
    bool isIncrement() const { return IsIncrement; }
  } ExprChecker;
  Expr *StepValue;
  ASTContext &Context;
  bool IsLessOp, IsCompatibleWithTest;

public:
  bool VisitUnaryOperator(UnaryOperator *UO) {
    if (!UO->isIncrementDecrementOp())
      return false;
    if (VarChecker.Visit(UO->getSubExpr())) {
      IsCompatibleWithTest = (IsLessOp && UO->isIncrementOp()) ||
                             (!IsLessOp && UO->isDecrementOp());
      if (!IsCompatibleWithTest && IsLessOp)
        StepValue = ActOnIntegerConstant(SourceLocation(), -1);
      else
        StepValue = ActOnIntegerConstant(SourceLocation(), 1);
    }
    return StepValue != 0;
  }
  bool VisitBinaryOperator(BinaryOperator *BO) {
    IsCompatibleWithTest = (IsLessOp && BO->getOpcode() == BO_AddAssign) ||
                           (!IsLessOp && BO->getOpcode() == BO_SubAssign);
    switch (BO->getOpcode()) {
    case BO_AddAssign:
    case BO_SubAssign:
      if (VarChecker.Visit(BO->getLHS())) {
        StepValue = BO->getRHS();
        IsCompatibleWithTest = (IsLessOp && BO->getOpcode() == BO_AddAssign) ||
                               (!IsLessOp && BO->getOpcode() == BO_SubAssign);
      }
      return StepValue != 0;
    case BO_Assign:
      if (VarChecker.Visit(BO->getLHS()) && ExprChecker.Visit(BO->getRHS())) {
        StepValue = ExprChecker.getStepValue();
        IsCompatibleWithTest = IsLessOp == ExprChecker.isIncrement();
      }
      return StepValue != 0;
    default:
      break;
    }
    return false;
  }
  bool VisitCXXOperatorCallExpr(CXXOperatorCallExpr *E) {
    switch (E->getOperator()) {
    case OO_PlusPlus:
    case OO_MinusMinus:
      if (VarChecker.Visit(E->getArg(0))) {
        IsCompatibleWithTest = (IsLessOp && E->getOperator() == OO_PlusPlus) ||
                               (!IsLessOp && E->getOperator() == OO_MinusMinus);
        if (!IsCompatibleWithTest && IsLessOp)
          StepValue = ActOnIntegerConstant(SourceLocation(), -1);
        else
          StepValue = ActOnIntegerConstant(SourceLocation(), 1);
      }
      return StepValue != 0;
    case OO_PlusEqual:
    case OO_MinusEqual:
      if (VarChecker.Visit(E->getArg(0))) {
        StepValue = E->getArg(1);
        IsCompatibleWithTest = (IsLessOp && E->getOperator() == OO_PlusEqual) ||
                               (!IsLessOp && E->getOperator() == OO_MinusEqual);
      }
      return StepValue != 0;
    case OO_Equal:
      if (VarChecker.Visit(E->getArg(0)) && ExprChecker.Visit(E->getArg(1))) {
        StepValue = ExprChecker.getStepValue();
        IsCompatibleWithTest = IsLessOp == ExprChecker.isIncrement();
      }
      return StepValue != 0;
    default:
      break;
    }
    return false;
  }
  bool VisitStmt(Stmt *S) { return false; }
  ForIncrChecker(Decl *D, ASTContext &Context, bool LessOp)
      : VarChecker(D), ExprChecker(VarChecker), StepValue(0), Context(Context),
        IsLessOp(LessOp), IsCompatibleWithTest(false) {}
  Expr *getStepValue() { return StepValue; }
  bool isCompatibleWithTest() const { return IsCompatibleWithTest; }
};
} // namespace

bool CGOpenMPRuntimeSpark::isNotSupportedLoopForm(
    Stmt *S, OpenMPDirectiveKind Kind, Expr *&InitVal, Expr *&StepVal,
    Expr *&CheckVal, VarDecl *&VarCnt, Expr *&CheckOp,
    BinaryOperatorKind &OpKind) {
  // assert(S && "non-null statement must be specified");
  // OpenMP [2.9.5, Canonical Loop Form]
  //  for (init-expr; test-expr; incr-expr) structured-block
  OpKind = BO_Assign;
  ForStmt *For = dyn_cast_or_null<ForStmt>(S);
  if (!For) {
    //    Diag(S->getLocStart(), diag::err_omp_not_for)
    //        << getOpenMPDirectiveName(Kind);
    return true;
  }
  Stmt *Body = For->getBody();
  if (!Body) {
    //    Diag(S->getLocStart(), diag::err_omp_directive_nonblock)
    //        << getOpenMPDirectiveName(Kind);
    return true;
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  init-expr One of the following:
  //  var = lb
  //  integer-type var = lb
  //  random-access-iterator-type var = lb
  //  pointer-type var = lb
  ForInitChecker InitChecker;
  Stmt *Init = For->getInit();
  VarDecl *Var;
  if (!Init || !(Var = dyn_cast_or_null<VarDecl>(InitChecker.Visit(Init)))) {
    //    Diag(Init ? Init->getLocStart() : For->getForLoc(),
    //         diag::err_omp_not_canonical_for)
    //        << 0;
    return true;
  }
  SourceLocation InitLoc = Init->getLocStart();

  // OpenMP [2.11.1.1, Data-sharing Attribute Rules for Variables Referenced
  // in a Construct, C/C++]
  // The loop iteration variable(s) in the associated for-loop(s) of a for or
  // parallel for construct may be listed in a private or lastprivate clause.
  bool HasErrors = false;

  // OpenMP [2.9.5, Canonical Loop Form]
  // Var One of the following
  // A variable of signed or unsigned integer type
  // For C++, a variable of a random access iterator type.
  // For C, a variable of a pointer type.
  QualType Type = Var->getType()
                      .getNonReferenceType()
                      .getCanonicalType()
                      .getUnqualifiedType();
  if (!Type->isIntegerType() && !Type->isPointerType() &&
      (!CGM.getLangOpts().CPlusPlus || !Type->isOverloadableType())) {
    //    Diag(Init->getLocStart(), diag::err_omp_for_variable)
    //        << getLangOpts().CPlusPlus;
    HasErrors = true;
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  test-expr One of the following:
  //  var relational-op b
  //  b relational-op var
  ForTestChecker TestChecker(Var);
  Stmt *Cond = For->getCond();
  CheckOp = cast<Expr>(Cond);
  bool TestCheckCorrect = false;
  if (!Cond || !(TestCheckCorrect = TestChecker.Visit(Cond))) {
    //    Diag(Cond ? Cond->getLocStart() : For->getForLoc(),
    //         diag::err_omp_not_canonical_for)
    //        << 1;
    HasErrors = true;
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  incr-expr One of the following:
  //  ++var
  //  var++
  //  --var
  //  var--
  //  var += incr
  //  var -= incr
  //  var = var + incr
  //  var = incr + var
  //  var = var - incr
  ForIncrChecker IncrChecker(Var, CGM.getContext(), TestChecker.isLessOp());
  Stmt *Incr = For->getInc();
  bool IncrCheckCorrect = false;
  if (!Incr || !(IncrCheckCorrect = IncrChecker.Visit(Incr))) {
    //    Diag(Incr ? Incr->getLocStart() : For->getForLoc(),
    //         diag::err_omp_not_canonical_for)
    //        << 2;
    HasErrors = true;
  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  lb and b Loop invariant expressions of a type compatible with the type
  //  of var.
  Expr *InitValue = InitChecker.getInitValue();
  //  QualType InitTy =
  //    InitValue ? InitValue->getType().getNonReferenceType().
  //                                  getCanonicalType().getUnqualifiedType() :
  //                QualType();
  //  if (InitValue &&
  //      Context.mergeTypes(Type, InitTy, false, true).isNull()) {
  //    Diag(InitValue->getExprLoc(), diag::err_omp_for_type_not_compatible)
  //      << InitValue->getType()
  //      << Var << Var->getType();
  //    HasErrors = true;
  //  }
  Expr *CheckValue = TestChecker.getCheckValue();
  //  QualType CheckTy =
  //    CheckValue ? CheckValue->getType().getNonReferenceType().
  //                                  getCanonicalType().getUnqualifiedType() :
  //                 QualType();
  //  if (CheckValue &&
  //      Context.mergeTypes(Type, CheckTy, false, true).isNull()) {
  //    Diag(CheckValue->getExprLoc(), diag::err_omp_for_type_not_compatible)
  //      << CheckValue->getType()
  //      << Var << Var->getType();
  //    HasErrors = true;
  //  }

  // OpenMP [2.9.5, Canonical Loop Form]
  //  incr A loop invariant integer expression.
  Expr *Step = IncrChecker.getStepValue();
  if (Step && !Step->getType()->isIntegralOrEnumerationType()) {
    //    Diag(Step->getExprLoc(), diag::err_omp_for_incr_not_integer);
    HasErrors = true;
  }

  // OpenMP [2.9.5, Canonical Loop Form, Restrictions]
  //  If test-expr is of form var relational-op b and relational-op is < or
  //  <= then incr-expr must cause var to increase on each iteration of the
  //  loop. If test-expr is of form var relational-op b and relational-op is
  //  > or >= then incr-expr must cause var to decrease on each iteration of the
  //  loop.
  //  If test-expr is of form b relational-op var and relational-op is < or
  //  <= then incr-expr must cause var to decrease on each iteration of the
  //  loop. If test-expr is of form b relational-op var and relational-op is
  //  > or >= then incr-expr must cause var to increase on each iteration of the
  //  loop.
  if (Incr && TestCheckCorrect && IncrCheckCorrect &&
      !IncrChecker.isCompatibleWithTest()) {
    // Additional type checking.
    llvm::APSInt Result;
    bool IsConst = Step->isIntegerConstantExpr(Result, CGM.getContext());
    bool IsConstNeg = IsConst && Result.isSigned() && Result.isNegative();
    bool IsSigned = Step->getType()->hasSignedIntegerRepresentation();
    if ((TestChecker.isLessOp() && IsConst && IsConstNeg) ||
        (!TestChecker.isLessOp() &&
         ((IsConst && !IsConstNeg) || (!IsConst && !IsSigned)))) {
      //      Diag(Incr->getLocStart(), diag::err_omp_for_incr_not_compatible)
      //          << Var << TestChecker.isLessOp();
      HasErrors = true;
    } else {
      // TODO: Negative increment
      // Step = CreateBuiltinUnaryOp(Step->getExprLoc(), UO_Minus, Step);
    }
  }
  if (HasErrors)
    return true;

  assert(Step && "Null expr in Step in OMP FOR");
  Step = Step->IgnoreParenImpCasts();
  CheckValue = CheckValue->IgnoreParenImpCasts();
  InitValue = InitValue->IgnoreParenImpCasts();

  //  if (TestChecker.isStrictOp()) {
  //    Diff = BuildBinOp(DSAStack->getCurScope(), InitLoc, BO_Sub, CheckValue,
  //                      ActOnIntegerConstant(SourceLocation(), 1));
  //  }

  InitVal = InitValue;
  CheckVal = CheckValue;
  StepVal = Step;
  VarCnt = Var;

  return false;
}

/// A StmtVisitor that propagates the raw counts through the AST and
/// records the count at statements where the value may change.
class FindKernelArguments : public RecursiveASTVisitor<FindKernelArguments> {

private:
  ArraySubscriptExpr *CurrArrayExpr;
  Expr *CurrArrayIndexExpr;

  llvm::DenseMap<const VarDecl *, llvm::SmallVector<const Expr *, 8>>
      MapVarToExpr;
  llvm::SmallSet<const VarDecl *, 8> Inputs;
  llvm::SmallSet<const VarDecl *, 8> Outputs;
  llvm::SmallSet<const VarDecl *, 8> InputsOutputs;

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
    for (auto In : Inputs) {
      Info->InVarUse[In].append(MapVarToExpr[In].begin(),
                                MapVarToExpr[In].end());
      llvm::errs() << " " << In->getName();
    }
    llvm::errs() << "\n";
    llvm::errs() << "Outputs =";
    for (auto Out : Outputs) {
      Info->OutVarDef[Out].append(MapVarToExpr[Out].begin(),
                                  MapVarToExpr[Out].end());
      llvm::errs() << " " << Out->getName();
    }
    llvm::errs() << "\n";
    llvm::errs() << "InputsOutputs =";
    for (auto InOut : InputsOutputs) {
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
        // VisitOMPMapClause(C); // FIXME: reactivate

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

      bool currInput =
          std::find(Inputs.begin(), Inputs.end(), VD) != Inputs.end();
      bool currOutput =
          std::find(Outputs.begin(), Outputs.end(), VD) != Outputs.end();
      bool currInputOutput =
          std::find(InputsOutputs.begin(), InputsOutputs.end(), VD) !=
          InputsOutputs.end();

      MapVarToExpr[VD].push_back(D);

      if (current_use == Use) {
        if (verbose)
          llvm::errs() << " is Use";
        if (currInputOutput) {
          ;
        } else if (currOutput) {
          Outputs.erase(VD);
          InputsOutputs.insert(VD);
        } else {
          Inputs.insert(VD);
        }
      } else if (current_use == Def) {
        if (verbose)
          llvm::errs() << " is Def";
        if (currInputOutput) {
          ;
        } else if (currInput) {
          Inputs.erase(VD);
          InputsOutputs.insert(VD);
        } else {
          Outputs.insert(VD);
        }
      } else if (current_use == UseDef) {
        if (verbose)
          llvm::errs() << " is UseDef";
        Inputs.erase(VD);
        Outputs.erase(VD);
        InputsOutputs.insert(VD);
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
    CGF.CurFn = RedFn;
    CGF.EnsureInsertPoint();

    // Generate useful type and constant
    llvm::PointerType *PointerTy_Int8 =
        llvm::PointerType::get(CGF.Builder.getInt8Ty(), 0);
    llvm::PointerType *PointerTy_Int32 =
        llvm::PointerType::get(CGF.Builder.getInt32Ty(), 0);

    llvm::ConstantInt *const_int32_0 = llvm::ConstantInt::get(
        CGM.getLLVMContext(), llvm::APInt(32, llvm::StringRef("0"), 10));

    llvm::ConstantPointerNull *const_ptr_null =
        llvm::ConstantPointerNull::get(PointerTy_Int8);

    // Find the bit size
    QualType VarType = VD->getType();
    int32_t SizeInByte = CGM.getContext().getTypeSize(VarType) / 8;
    llvm::ConstantInt *const_int32_typeSizeIntByte =
        llvm::ConstantInt::get(CGF.Builder.getInt32Ty(), SizeInByte);

    // Allocate and load compulsry JNI arguments
    llvm::Function::arg_iterator args = RedFn->arg_begin();
    args->setName("env");
    llvm::AllocaInst *alloca_env = CGF.Builder.CreateAlloca(PointerTy_1);
    CGF.Builder.CreateAlignedStore(&*args, alloca_env, CGM.getPointerAlign());
    args++;
    args->setName("obj");
    llvm::AllocaInst *alloca_obj = CGF.Builder.CreateAlloca(PointerTy_jobject);
    CGF.Builder.CreateAlignedStore(&*args, alloca_obj, CGM.getPointerAlign());
    args++;

    // FIXME: check alignment
    llvm::LoadInst *ptr_env =
        CGF.Builder.CreateAlignedLoad(alloca_env, CGM.getPointerAlign());
    llvm::LoadInst *ptr_ptr_env =
        CGF.Builder.CreateAlignedLoad(ptr_env, CGM.getPointerAlign());

    llvm::Value *ptr_gep_getelement =
        CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 184);
    llvm::LoadInst *ptr_fn_getelement = CGF.Builder.CreateAlignedLoad(
        ptr_gep_getelement, CGM.getPointerAlign());

    llvm::Value *ptr_gep_releaseelement =
        CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 192);
    llvm::LoadInst *ptr_fn_releaseelement = CGF.Builder.CreateAlignedLoad(
        ptr_gep_releaseelement, CGM.getPointerAlign());

    llvm::Value *ptr_gep_newbytearray =
        CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 176);
    llvm::LoadInst *ptr_fn_newbytearray = CGF.Builder.CreateAlignedLoad(
        ptr_gep_newbytearray, CGM.getPointerAlign());

    // Allocate, load and cast the first operand
    llvm::AllocaInst *alloca_arg1 = CGF.Builder.CreateAlloca(PointerTy_jobject);
    CGF.Builder.CreateAlignedStore(&*args, alloca_arg1, CGM.getPointerAlign());

    llvm::LoadInst *ptr_arg1 =
        CGF.Builder.CreateAlignedLoad(alloca_arg1, CGM.getPointerAlign());
    std::vector<llvm::Value *> ptr_275_params;
    ptr_275_params.push_back(ptr_env);
    ptr_275_params.push_back(ptr_arg1);
    ptr_275_params.push_back(const_ptr_null);
    llvm::CallInst *ptr_275 =
        CGF.Builder.CreateCall(ptr_fn_getelement, ptr_275_params);

    llvm::Value *ptr_265 = CGF.Builder.CreateBitCast(ptr_275, PointerTy_Int32);
    llvm::Value *ptr_265_3 =
        CGF.Builder.CreateAlignedLoad(ptr_265, CGM.getPointerAlign());
    llvm::Value *ptr_265_3_cast =
        CGF.Builder.CreateBitCast(ptr_265_3, CGF.Builder.getInt32Ty());
    args++;

    // Allocate, load and cast the second operand
    llvm::AllocaInst *alloca_arg2 = CGF.Builder.CreateAlloca(PointerTy_jobject);
    CGF.Builder.CreateAlignedStore(&*args, alloca_arg2, CGM.getPointerAlign());

    llvm::LoadInst *ptr_arg2 =
        CGF.Builder.CreateAlignedLoad(alloca_arg2, CGM.getPointerAlign());
    std::vector<llvm::Value *> ptr_275_1_params;
    ptr_275_1_params.push_back(ptr_env);
    ptr_275_1_params.push_back(ptr_arg2);
    ptr_275_1_params.push_back(const_ptr_null);
    llvm::CallInst *ptr_275_1 =
        CGF.Builder.CreateCall(ptr_fn_getelement, ptr_275_1_params);

    llvm::Value *ptr_265_1 =
        CGF.Builder.CreateBitCast(ptr_275_1, PointerTy_Int32);
    llvm::Value *ptr_265_2 =
        CGF.Builder.CreateAlignedLoad(ptr_265_1, CGM.getPointerAlign());
    llvm::Value *ptr_265_2_cast =
        CGF.Builder.CreateBitCast(ptr_265_2, CGF.Builder.getInt32Ty());

    // Compute the reduction
    llvm::Value *res = nullptr;

    /* FIXME: generate operation
    switch (C.getOperator()) {
    case OMPC_REDUCTION_or:
    case OMPC_REDUCTION_bitor: {
      res = CGF.Builder.CreateOr(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_bitxor: {
      res = CGF.Builder.CreateXor(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_sub: {
      res = CGF.Builder.CreateSub(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_add: {
      res = CGF.Builder.CreateAdd(ptr_265_3_cast, ptr_265_2_cast, "", false,
                                  true);
      break;
    }
    case OMPC_REDUCTION_and:
    case OMPC_REDUCTION_bitand: {
      res = CGF.Builder.CreateAnd(ptr_265_3_cast, ptr_265_2_cast);
      break;
    }
    case OMPC_REDUCTION_mult: {
      res = CGF.Builder.CreateMul(ptr_265_3_cast, ptr_265_2_cast);
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
    llvm::AllocaInst *alloca_res =
        CGF.Builder.CreateAlloca(CGF.Builder.getInt32Ty());
    CGF.Builder.CreateAlignedStore(res, alloca_res, CGM.getIntAlign());

    // Protect arg 1

    {
      std::vector<llvm::Value *> void_272_params;
      void_272_params.push_back(ptr_env);
      void_272_params.push_back(ptr_arg1);
      void_272_params.push_back(ptr_275);
      void_272_params.push_back(const_int32_0);
      CGF.Builder.CreateCall(ptr_fn_releaseelement, void_272_params);
    }

    // Protect arg 2

    {
      std::vector<llvm::Value *> void_272_params;
      void_272_params.push_back(ptr_env);
      void_272_params.push_back(ptr_arg2);
      void_272_params.push_back(ptr_275_1);
      void_272_params.push_back(const_int32_0);
      CGF.Builder.CreateCall(ptr_fn_releaseelement, void_272_params);
    }

    // Cast back the result to bit array
    std::vector<llvm::Value *> ptr_277_params;
    ptr_277_params.push_back(ptr_env);
    ptr_277_params.push_back(const_int32_typeSizeIntByte);
    llvm::CallInst *ptr_277 =
        CGF.Builder.CreateCall(ptr_fn_newbytearray, ptr_277_params);

    llvm::Value *ptr_279 =
        CGF.Builder.CreateConstInBoundsGEP2_32(nullptr, ptr_ptr_env, 0, 208);
    llvm::LoadInst *ptr_280 =
        CGF.Builder.CreateAlignedLoad(ptr_279, CGM.getPointerAlign());
    llvm::Value *ptr_res_cast =
        CGF.Builder.CreateBitCast(alloca_res, PointerTy_Int8, "");
    std::vector<llvm::Value *> void_281_params;
    void_281_params.push_back(ptr_env);
    void_281_params.push_back(ptr_277);
    void_281_params.push_back(const_int32_0);
    void_281_params.push_back(const_int32_typeSizeIntByte);
    void_281_params.push_back(ptr_res_cast);
    CGF.Builder.CreateCall(ptr_280, void_281_params);

    CGF.Builder.CreateRet(ptr_277);
  }
}

void CGOpenMPRuntimeSpark::GenerateMappingKernel(
    const OMPExecutableDirective &S) {
  bool verbose = VERBOSE;

  auto& DL = CGM.getDataLayout();

  const OMPParallelForDirective &ForDirective =
      cast<OMPParallelForDirective>(S);

  DefineJNITypes();

  for (ArrayRef<OMPClause *>::const_iterator I = S.clauses().begin(),
                                             E = S.clauses().end();
       I != E; ++I)
    if (*I && (*I)->getClauseKind() == OMPC_reduction)
      GenerateReductionKernel(cast<OMPReductionClause>(*(*I)), S);

  auto &typeMap = OffloadingMapVarsType;
  auto &indexMap = OffloadingMapVarsIndex;

  // FIXME: what about several functions
  auto &info = *(SparkMappingFunctions.back());

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

    Expr *Step;
    Expr *Check;
    Expr *Init;
    VarDecl *VarCnt;
    Expr *CheckOp;
    BinaryOperatorKind OpKind;

    isNotSupportedLoopForm(LoopStmt, S.getDirectiveKind(), Init, Step, Check,
                           VarCnt, CheckOp, OpKind);

    if (verbose)
      llvm::errs() << "Find counter " << VarCnt->getName() << "\n";

    auto &CntInfo = info.CounterInfo[VarCnt];
    CntInfo.push_back(Init);
    CntInfo.push_back(Check);
    CntInfo.push_back(Step);
    CntInfo.push_back(CheckOp);

    Body = For->getBody();
  }

  // Create the mapping function

  // Initialize a new CodeGenFunction used to generate the mapping
  CodeGenFunction CGF(CGM, true);

  // Detect input/output expression from the loop body
  FindKernelArguments Finder(CGM, *this, &info);
  Finder.Explore(LoopStmt);

  // Get JNI type
  llvm::StructType *StructTy_JNINativeInterface =
      CGM.getModule().getTypeByName("struct.JNINativeInterface_");
  llvm::PointerType *PointerTy_JNINativeInterface =
      llvm::PointerType::get(StructTy_JNINativeInterface, 0);
  llvm::PointerType *PointerTy_1 =
      llvm::PointerType::get(PointerTy_JNINativeInterface, 0);

  llvm::StructType *StructTy_jobject =
      CGM.getModule().getTypeByName("struct._jobject");
  llvm::PointerType *PointerTy_jobject =
      llvm::PointerType::get(StructTy_jobject, 0);

  llvm::IntegerType *IntTy_jlong = CGF.Builder.getInt64Ty();
  llvm::IntegerType *IntTy_jint = CGF.Builder.getInt32Ty();

  // Initialize arguments
  std::vector<llvm::Type *> FuncTy_args;

  // Add compulsary arguments
  FuncTy_args.push_back(PointerTy_1);
  FuncTy_args.push_back(PointerTy_jobject);

  for (auto it = info.CounterInfo.begin(); it != info.CounterInfo.end(); ++it) {
    FuncTy_args.push_back(IntTy_jlong);
    FuncTy_args.push_back(IntTy_jlong);
  }

  for (auto it = info.InVarUse.begin(); it != info.InVarUse.end(); ++it) {
    FuncTy_args.push_back(PointerTy_jobject);
  }

  for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end(); ++it) {
    FuncTy_args.push_back(PointerTy_jobject);
  }

  for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end(); ++it) {
    FuncTy_args.push_back(PointerTy_jobject);
  }

  llvm::FunctionType *FnTy = llvm::FunctionType::get(
      /*Result=*/PointerTy_jobject,
      /*Params=*/FuncTy_args,
      /*isVarArg=*/false);

  std::string FnName = "Java_org_llvm_openmp_OmpKernel_mappingMethod" +
                       std::to_string(info.Identifier);

  llvm::Function *MapFn = llvm::Function::Create(
      FnTy, llvm::GlobalValue::ExternalLinkage, FnName, &CGM.getModule());

  CGF.CurFn = MapFn;

  llvm::BasicBlock *EntryBB = CGF.createBasicBlock("entry", MapFn);

  // Create a marker to make it easy to insert allocas into the entryblock
  // later.  Don't create this with the builder, because we don't want it
  // folded.
  llvm::Value *Undef = llvm::UndefValue::get(CGF.Builder.getInt32Ty());
  CGF.AllocaInsertPt = new llvm::BitCastInst(Undef, CGF.Builder.getInt32Ty(),
                                             "allocapt", EntryBB);

  CGF.Builder.SetInsertPoint(EntryBB);

  // Generate useful type and constant
  llvm::PointerType *PointerTy_Int8 =
      llvm::PointerType::get(CGF.Builder.getInt8Ty(), 0);

  llvm::ConstantInt *const_int32_0 = llvm::ConstantInt::get(
      CGM.getLLVMContext(), llvm::APInt(32, llvm::StringRef("0"), 10));
  llvm::ConstantInt *const_int32_2 = llvm::ConstantInt::get(
      CGM.getLLVMContext(), llvm::APInt(32, llvm::StringRef("2"), 10));

  llvm::ConstantPointerNull *const_ptr_null =
      llvm::ConstantPointerNull::get(PointerTy_Int8);

  // Global variable
  llvm::Value *const_ptr_init =
      CGF.Builder.CreateGlobalStringPtr("<init>", ".str.init");
  llvm::Value *const_ptr_tuple2 =
      CGF.Builder.CreateGlobalStringPtr("scala/Tuple2", ".str.tuple2");
  llvm::Value *const_ptr_tuple3 =
      CGF.Builder.CreateGlobalStringPtr("scala/Tuple3", ".str.tuple3");
  llvm::Value *const_ptr_tuple2_args = CGF.Builder.CreateGlobalStringPtr(
      "(Ljava/lang/Object;Ljava/lang/Object;)V", ".str.tuple2.args");
  llvm::Value *const_ptr_tuple3_args = CGF.Builder.CreateGlobalStringPtr(
      "(Ljava/lang/Object;Ljava/lang/Object;Ljava/lang/Object;)V",
      ".str.tuple3.args");

  // Allocate and load compulsory JNI arguments
  llvm::Function::arg_iterator args = MapFn->arg_begin();
  args->setName("env");
  llvm::AllocaInst *alloca_env = CGF.Builder.CreateAlloca(PointerTy_1);
  CGF.Builder.CreateAlignedStore(&*args, alloca_env, CGM.getPointerAlign());
  args++;
  args->setName("obj");
  llvm::AllocaInst *alloca_obj = CGF.Builder.CreateAlloca(PointerTy_jobject);
  CGF.Builder.CreateAlignedStore(&*args, alloca_obj, CGM.getPointerAlign());
  args++;

  llvm::LoadInst *ptr_env =
      CGF.Builder.CreateAlignedLoad(alloca_env, CGM.getPointerAlign());
  llvm::LoadInst *ptr_ptr_env =
      CGF.Builder.CreateAlignedLoad(ptr_env, CGM.getPointerAlign());

  // Load pointer to JNI functions
  llvm::Value *ptr_gep_getelement =
      CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 184);
  llvm::LoadInst *ptr_fn_getelement =
      CGF.Builder.CreateAlignedLoad(ptr_gep_getelement, CGM.getPointerAlign());

  llvm::Value *ptr_gep_getcritical =
      CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 222);
  llvm::LoadInst *ptr_fn_getcritical =
      CGF.Builder.CreateAlignedLoad(ptr_gep_getcritical, CGM.getPointerAlign());

  llvm::Value *ptr_gep_releaseelement =
      CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 192);
  llvm::LoadInst *ptr_fn_releaseelement = CGF.Builder.CreateAlignedLoad(
      ptr_gep_releaseelement, CGM.getPointerAlign());

  llvm::Value *ptr_gep_releasecritical =
      CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 223);
  llvm::LoadInst *ptr_fn_releasecritical = CGF.Builder.CreateAlignedLoad(
      ptr_gep_releasecritical, CGM.getPointerAlign());

  llvm::Value *ptr_gep_newbytearray =
      CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 176);
  llvm::LoadInst *ptr_fn_newbytearray = CGF.Builder.CreateAlignedLoad(
      ptr_gep_newbytearray, CGM.getPointerAlign());

  llvm::Value *ptr_gep_findclass =
      CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 6);
  llvm::LoadInst *ptr_fn_findclass =
      CGF.Builder.CreateAlignedLoad(ptr_gep_findclass, CGM.getPointerAlign());

  llvm::Value *ptr_gep_getmethodid =
      CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 33);
  llvm::LoadInst *ptr_fn_getmethodid =
      CGF.Builder.CreateAlignedLoad(ptr_gep_getmethodid, CGM.getPointerAlign());

  llvm::Value *ptr_gep_newobject =
      CGF.Builder.CreateConstGEP2_32(nullptr, ptr_ptr_env, 0, 28);
  llvm::LoadInst *ptr_fn_newobject =
      CGF.Builder.CreateAlignedLoad(ptr_gep_newobject, CGM.getPointerAlign());

  // Keep values that have to be used for releasing.
  llvm::SmallVector<std::pair<llvm::Value *, llvm::Value *>, 8> InputsToRelease;
  llvm::SmallVector<std::pair<llvm::Value *, llvm::Value *>, 8>
      ScalarInputsToRelease;
  llvm::SmallVector<std::pair<llvm::Value *, llvm::Value *>, 8>
      InOutputsToRelease;
  llvm::SmallVector<std::pair<llvm::Value *, llvm::Value *>, 8>
      OutputsToRelease;

  llvm::Value *alloca_cnt;
  llvm::Value *alloca_cnt_bound;

  if (info.CounterInfo.size() > 1) {
    llvm::errs() << "ERROR OmpCloud: Do not support more than 1 iteration "
                    "index for now.";
    exit(EXIT_FAILURE);
  }

  for (auto it = info.CounterUse.begin(); it != info.CounterUse.end(); ++it) {
    const VarDecl *VD = it->first;

    // FIXME: What about long ??
    // Store current value of the loop counter
    alloca_cnt = CGF.Builder.CreateAlloca(IntTy_jint);
    llvm::Value *cast_cnt =
        CGF.Builder.CreateTruncOrBitCast(&*args, IntTy_jint);
    CGF.Builder.CreateAlignedStore(cast_cnt, alloca_cnt, CGM.getIntAlign());

    args++;

    // Store the bound of the inner tiled loop according to the current
    // iteration
    alloca_cnt_bound = CGF.Builder.CreateAlloca(IntTy_jint);
    llvm::Value *cast_cnt_bound =
        CGF.Builder.CreateTruncOrBitCast(&*args, IntTy_jint);
    CGF.Builder.CreateAlignedStore(cast_cnt_bound, alloca_cnt_bound,
                                   CGM.getIntAlign());

    args++;

    addOpenMPKernelArgVar(VD, alloca_cnt);
  }

  // Allocate, load and cast input variables (i.e. the arguments)
  for (auto it = info.InVarUse.begin(); it != info.InVarUse.end(); ++it) {
    const VarDecl *VD = it->first;
    args->setName(VD->getName());

    QualType varType = VD->getType();
    llvm::Value *valuePtr;

    if (!varType->isAnyPointerType()) {
      // GetByteArrayElements
      std::vector<llvm::Value *> ptr_load_arg_params;
      ptr_load_arg_params.push_back(ptr_env);
      ptr_load_arg_params.push_back(&*args);
      ptr_load_arg_params.push_back(const_ptr_null);
      llvm::CallInst *ptr_load_arg =
          CGF.Builder.CreateCall(ptr_fn_getelement, ptr_load_arg_params);

      ScalarInputsToRelease.push_back(std::make_pair(&*args, ptr_load_arg));

      llvm::Type *TyObject_arg = CGM.getTypes().ConvertType(varType);

      llvm::PointerType *PointerTy_arg =
          llvm::PointerType::get(TyObject_arg, 0);
      valuePtr = CGF.Builder.CreateBitCast(ptr_load_arg, PointerTy_arg);

    } else {
      // GetPrimitiveArrayCritical
      std::vector<llvm::Value *> ptr_load_arg_params;
      ptr_load_arg_params.push_back(ptr_env);
      ptr_load_arg_params.push_back(&*args);
      ptr_load_arg_params.push_back(const_ptr_null);
      llvm::CallInst *ptr_load_arg =
          CGF.Builder.CreateCall(ptr_fn_getcritical, ptr_load_arg_params);

      InputsToRelease.push_back(std::make_pair(&*args, ptr_load_arg));

      llvm::Type *TyObject_arg = CGM.getTypes().ConvertType(varType);

      llvm::Value *ptr_casted_arg =
          CGF.Builder.CreateBitCast(ptr_load_arg, TyObject_arg);

      valuePtr = CGF.Builder.CreateAlloca(TyObject_arg);
      CGF.Builder.CreateAlignedStore(ptr_casted_arg, valuePtr,
                                     CGM.getPointerAlign());

      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        llvm::Value *LowerBound = CGF.EmitScalarExpr(Range->getLowerBound());
        for (auto it = info.RangedArrayAccess[VD].begin();
             it != info.RangedArrayAccess[VD].end(); ++it)
          addOpenMPKernelArgRange(*it, LowerBound);
      }
    }

    addOpenMPKernelArgVar(VD, valuePtr);

    args++;
  }

  // Allocate, load and cast input/output variables (i.e. the arguments)
  for (auto it = info.InOutVarUse.begin(); it != info.InOutVarUse.end(); ++it) {
    const VarDecl *VD = it->first;

    // GetPrimitiveArrayCritical
    std::vector<llvm::Value *> ptr_load_arg_params;
    ptr_load_arg_params.push_back(ptr_env);
    ptr_load_arg_params.push_back(&*args);
    ptr_load_arg_params.push_back(const_ptr_null);
    llvm::CallInst *ptr_load_arg =
        CGF.Builder.CreateCall(ptr_fn_getcritical, ptr_load_arg_params);

    args->setName(VD->getName());

    InOutputsToRelease.push_back(std::make_pair(&*args, ptr_load_arg));

    QualType varType = VD->getType();
    llvm::Type *TyObject_arg = CGM.getTypes().ConvertType(varType);

    llvm::Value *valuePtr;

    if (!varType->isAnyPointerType()) {
      if (verbose)
        llvm::errs() << ">Test< " << VD->getName() << " is scalar\n";

      valuePtr = CGF.Builder.CreateBitCast(
          ptr_load_arg, llvm::PointerType::get(TyObject_arg, 0));

    } else {
      llvm::Value *ptr_casted_arg =
          CGF.Builder.CreateBitCast(ptr_load_arg, TyObject_arg);

      valuePtr = CGF.Builder.CreateAlloca(TyObject_arg);

      CGF.Builder.CreateAlignedStore(ptr_casted_arg, valuePtr, DL.getPrefTypeAlignment(TyObject_arg));

      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        llvm::Value *LowerBound = CGF.EmitScalarExpr(Range->getLowerBound());
        for (auto it = info.RangedArrayAccess[VD].begin();
             it != info.RangedArrayAccess[VD].end(); ++it)
          addOpenMPKernelArgRange(*it, LowerBound);
      }
    }

    addOpenMPKernelArgVar(VD, valuePtr);

    args++;
  }

  // Allocate output variables
  for (auto it = info.OutVarDef.begin(); it != info.OutVarDef.end(); ++it) {
    const VarDecl *VD = it->first;

    // GetPrimitiveArrayCritical
    std::vector<llvm::Value *> ptr_load_arg_params;
    ptr_load_arg_params.push_back(ptr_env);
    ptr_load_arg_params.push_back(&*args);
    ptr_load_arg_params.push_back(const_ptr_null);
    llvm::CallInst *ptr_load_arg =
        CGF.Builder.CreateCall(ptr_fn_getcritical, ptr_load_arg_params);

    args->setName(VD->getName());

    OutputsToRelease.push_back(std::make_pair(&*args, ptr_load_arg));

    QualType varType = VD->getType();
    llvm::Type *TyObject_arg = CGM.getTypes().ConvertType(varType);

    llvm::Value *valuePtr;

    if (!varType->isAnyPointerType()) {
      if (verbose)
        llvm::errs() << ">Test< " << VD->getName() << " is scalar\n";

      llvm::PointerType *PointerTy_arg =
          llvm::PointerType::get(TyObject_arg, 0);
      valuePtr = CGF.Builder.CreateBitCast(ptr_load_arg, PointerTy_arg);

    } else {
      llvm::Value *ptr_casted_arg =
          CGF.Builder.CreateBitCast(ptr_load_arg, TyObject_arg);

      valuePtr = CGF.Builder.CreateAlloca(TyObject_arg);
      CGF.Builder.CreateAlignedStore(ptr_casted_arg, valuePtr, DL.getPrefTypeAlignment(TyObject_arg));

      if (const OMPArraySectionExpr *Range = info.RangedVar[VD]) {
        llvm::Value *LowerBound = CGF.EmitScalarExpr(Range->getLowerBound());
        for (auto it = info.RangedArrayAccess[VD].begin();
             it != info.RangedArrayAccess[VD].end(); ++it)
          addOpenMPKernelArgRange(*it, LowerBound);
      }
    }

    addOpenMPKernelArgVar(VD, valuePtr);

    args++;
  }


  //JumpDest LoopExit = CGF.getJumpDestInCurrentScope("for.end");

  // Evaluate the first part before the loop.
  if (For->getInit())
  CGF.EmitStmt(For->getInit());

  // Start the loop with a block that tests the condition.
  // If there's an increment, the continue scope will be overwritten
  // later.
  llvm::BasicBlock *CondBlock = CGF.createBasicBlock("for.cond");
  llvm::BasicBlock *Continue = CGF.createBasicBlock("for.cont");
  CGF.EmitBlock(Continue);

  llvm::BasicBlock *ExitBlock = CGF.createBasicBlock("for.end");
  llvm::BasicBlock *ForBody = CGF.createBasicBlock("for.body");

  // If the for loop doesn't have an increment we can just use the
  // condition as the continue block.  Otherwise we'll need to create
  // a block for it (in the current scope, i.e. in the scope of the
  // condition), and that we will become our continue block.
  if (For->getInc())
    Continue = CGF.createBasicBlock("for.inc");

  if (For->getCond()) {
    // If the for statement has a condition scope, emit the local variable
    // declaration.
    if (For->getConditionVariable()) {
      CGF.EmitAutoVarDecl(*For->getConditionVariable());
    }

    // C99 6.8.5p2/p4: The first substatement is executed if the expression
    // compares unequal to 0.  The condition must be a scalar type.

    llvm::Value *Cond = CGF.Builder.CreateICmpULE(
        CGF.Builder.CreateAlignedLoad(alloca_cnt, CGM.getIntAlign()),
        CGF.Builder.CreateAlignedLoad(alloca_cnt_bound, CGM.getIntAlign()));

    CGF.Builder.CreateCondBr(Cond, ForBody, ExitBlock);

    CGF.EmitBlock(ExitBlock);

    CGF.EmitBlock(ForBody);
  }

  {
    // FIXME: CGM.OpenMPSupport.startSparkRegion();
    // Create a separate cleanup scope for the body, in case it is not
    // a compound statement.

    // Generate kernel code
    CGF.EmitStmt(Body);

    // FIXME: CGM.OpenMPSupport.stopSparkRegion();
  }

  // If there is an increment, emit it next.
  if (For->getInc()) {
    CGF.EmitBlock(Continue);
    CGF.EmitStmt(For->getInc());
  }

  CGF.EmitBranch(CondBlock);

  // Emit the fall-through block.
  CGF.EmitBlock(ExitBlock, true);

  for (auto it = InputsToRelease.begin(); it != InputsToRelease.end(); ++it) {
    // ReleaseCritical
    std::vector<llvm::Value *> params_release;
    params_release.push_back(ptr_env);
    params_release.push_back(it->first);
    params_release.push_back(it->second);
    params_release.push_back(const_int32_2);
    CGF.Builder.CreateCall(ptr_fn_releasecritical, params_release);
  }

  for (auto it = ScalarInputsToRelease.begin();
       it != ScalarInputsToRelease.end(); ++it) {
    // ReleaseByteArrayElements
    std::vector<llvm::Value *> params_release;
    params_release.push_back(ptr_env);
    params_release.push_back(it->first);
    params_release.push_back(it->second);
    params_release.push_back(const_int32_2);
    CGF.Builder.CreateCall(ptr_fn_releaseelement, params_release);
  }

  llvm::SmallVector<llvm::Value *, 8> OutputsToReturn;

  for (auto it = OutputsToRelease.begin(); it != OutputsToRelease.end(); ++it) {
    // ReleaseByteArrayElements
    std::vector<llvm::Value *> params_release;
    params_release.push_back(ptr_env);
    params_release.push_back(it->first);
    params_release.push_back(it->second);
    params_release.push_back(const_int32_0);
    CGF.Builder.CreateCall(ptr_fn_releasecritical, params_release);

    OutputsToReturn.push_back(it->first);
  }

  for (auto it = InOutputsToRelease.begin(); it != InOutputsToRelease.end();
       ++it) {
    // ReleaseCritical
    std::vector<llvm::Value *> params_release;
    params_release.push_back(ptr_env);
    params_release.push_back(it->first);
    params_release.push_back(it->second);
    params_release.push_back(const_int32_0);
    CGF.Builder.CreateCall(ptr_fn_releasecritical, params_release);

    OutputsToReturn.push_back(it->first);
  }

  unsigned NbOutputs = info.OutVarDef.size() + info.InOutVarUse.size();

  if (NbOutputs == 1) {
    // Just return the value
    CGF.Builder.CreateRet(OutputsToReturn.front());
  } else if (NbOutputs == 2) {
    // Construct and return a Tuple2
    std::vector<llvm::Value *> params_findclass;
    params_findclass.push_back(ptr_env);
    params_findclass.push_back(const_ptr_tuple2);
    llvm::CallInst *ptr_class_tuple2 =
        CGF.Builder.CreateCall(ptr_fn_findclass, params_findclass);

    std::vector<llvm::Value *> params_getmethodid;
    params_getmethodid.push_back(ptr_env);
    params_getmethodid.push_back(ptr_class_tuple2);
    params_getmethodid.push_back(const_ptr_init);
    params_getmethodid.push_back(const_ptr_tuple2_args);
    llvm::CallInst *ptr_new_tuple2 =
        CGF.Builder.CreateCall(ptr_fn_getmethodid, params_getmethodid);

    std::vector<llvm::Value *> params_newobject;
    params_newobject.push_back(ptr_env);
    params_newobject.push_back(ptr_class_tuple2);
    params_newobject.push_back(ptr_new_tuple2);
    params_newobject.push_back(OutputsToReturn[0]);
    params_newobject.push_back(OutputsToReturn[1]);
    llvm::CallInst *ptr_tuple2 =
        CGF.Builder.CreateCall(ptr_fn_newobject, params_newobject);

    CGF.Builder.CreateRet(ptr_tuple2);
  } else if (NbOutputs == 3) {
    // Construct and return a Tuple3
    std::vector<llvm::Value *> params_findclass;
    params_findclass.push_back(ptr_env);
    params_findclass.push_back(const_ptr_tuple3);
    llvm::CallInst *ptr_class_tuple3 =
        CGF.Builder.CreateCall(ptr_fn_findclass, params_findclass);

    std::vector<llvm::Value *> params_getmethodid;
    params_getmethodid.push_back(ptr_env);
    params_getmethodid.push_back(ptr_class_tuple3);
    params_getmethodid.push_back(const_ptr_init);
    params_getmethodid.push_back(const_ptr_tuple3_args);
    llvm::CallInst *ptr_new_tuple3 =
        CGF.Builder.CreateCall(ptr_fn_getmethodid, params_getmethodid);

    std::vector<llvm::Value *> params_newobject;
    params_newobject.push_back(ptr_env);
    params_newobject.push_back(ptr_class_tuple3);
    params_newobject.push_back(ptr_new_tuple3);
    params_newobject.push_back(OutputsToReturn[0]);
    params_newobject.push_back(OutputsToReturn[1]);
    params_newobject.push_back(OutputsToReturn[2]);
    llvm::CallInst *ptr_tuple3 =
        CGF.Builder.CreateCall(ptr_fn_newobject, params_newobject);

    CGF.Builder.CreateRet(ptr_tuple3);
  } else {
    // TODO: Construct and return Tuples in generic way
    llvm::errs() << "ERROR OmpCloud: Need support for more than 3 outputs\n";
    exit(EXIT_FAILURE);
  }
}
