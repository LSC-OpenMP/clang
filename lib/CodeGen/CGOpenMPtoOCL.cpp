//===-- CGOpenMPtoOCL.cpp - Emit Opencl Kernel and LLVM Code from OpenMP
//Statements --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "TargetInfo.h"
#include "clang/AST/DeclOpenMP.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/StmtOpenMP.h"
using namespace clang;
using namespace CodeGen;

namespace {
//
// Some assets used by OpenMPtoOCL
//
std::vector<std::pair<int, std::string>> vectorNames[8];
std::vector<std::pair<int, std::string>> scalarNames[8];

std::map<llvm::Value *, std::string> vectorMap;
std::map<std::string, llvm::Value *> scalarMap;

llvm::SmallVector<QualType, 16> deftypes;

static bool dumpedDefType(const QualType *T) {
  for (ArrayRef<QualType>::iterator I = deftypes.begin(), E = deftypes.end();
       I != E; ++I) {
    if ((*I).getAsString() == T->getAsString())
      return true;
  }
  deftypes.push_back(*T);
  return false;
}

static bool pairCompare(const std::pair<int, std::string> &p1,
                        const std::pair<int, std::string> &p2) {
  return p1.second < p2.second;
}

struct Required {
  Required(std::string val) : val_(val) {}

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

} // namespace

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

void CodeGenModule::OpenMPSupportStackTy::getMapData(ArrayRef<llvm::Value*> &MapPointers,
                                                     ArrayRef<llvm::Value*> &MapSizes,
                                                     ArrayRef<QualType> &MapQualTypes,
                                                     ArrayRef<unsigned> &MapTypes){
    MapPointers = OpenMPStack.back().MapPointers;
    MapSizes = OpenMPStack.back().MapSizes;
    MapQualTypes = OpenMPStack.back().MapQualTypes;
    MapTypes = OpenMPStack.back().MapTypes;
}

void CodeGenModule::OpenMPSupportStackTy::addMapData(llvm::Value *MapPointer,
                                                     llvm::Value *MapSize,
                                                     QualType MapQualType,
                                                     unsigned MapType){
    OpenMPStack.back().MapPointers.push_back(MapPointer);
    OpenMPStack.back().MapSizes.push_back(MapSize);
    OpenMPStack.back().MapQualTypes.push_back(MapQualType);
    OpenMPStack.back().MapTypes.push_back(MapType);
}

void CodeGenModule::OpenMPSupportStackTy::getMapPos(ArrayRef<llvm::Value*> &MapPointers,
                                                    ArrayRef<llvm::Value*> &MapSizes,
                                                    ArrayRef<QualType> &MapQualTypes,
                                                    ArrayRef<unsigned> &MapTypes,
                                                    ArrayRef<unsigned> &MapPositions,
                                                    ArrayRef<unsigned> &MapScopes) {
    MapPointers = OpenMPStack.back().MapPointers;
    MapSizes = OpenMPStack.back().MapSizes;
    MapQualTypes = OpenMPStack.back().MapQualTypes;
    MapTypes = OpenMPStack.back().MapTypes;
    MapPositions = OpenMPStack.back().MapPositions;
    MapScopes = OpenMPStack.back().MapScopes;
}

void CodeGenModule::OpenMPSupportStackTy::addMapPos(llvm::Value *MapPointer,
                                                    llvm::Value *MapSize,
                                                    QualType MapQualType,
                                                    unsigned MapType,
                                                    unsigned MapPosition,
                                                    unsigned MapScope) {
    OpenMPStack.back().MapPointers.push_back(MapPointer);
    OpenMPStack.back().MapSizes.push_back(MapSize);
    OpenMPStack.back().MapQualTypes.push_back(MapQualType);
    OpenMPStack.back().MapTypes.push_back(MapType);
    OpenMPStack.back().MapPositions.push_back(MapPosition);
    OpenMPStack.back().MapScopes.push_back(MapScope);
}

bool CodeGenModule::OpenMPSupportStackTy::isKernelVar(llvm::Value *KernelVar) {

    for (SmallVector<llvm::Value*,16>::iterator I  = OpenMPStack.back().KernelVars.begin(),
                 E  = OpenMPStack.back().KernelVars.end();
         I != E; ++I) {
        llvm::Value *LV = (*I);
        unsigned oper = dyn_cast<llvm::User>(LV)->getNumOperands();
        while (!isa<llvm::AllocaInst>(LV) && oper>0) {
            LV = cast<llvm::User>(LV)->getOperand(0);
            oper = dyn_cast<llvm::User>(LV)->getNumOperands();
        }
        if (LV == KernelVar) return true;
    }
    return false;
}

bool CodeGenModule::OpenMPSupportStackTy::isScopVar(llvm::Value *ScopVar) {

    for (SmallVector<llvm::Value*,16>::iterator I  = OpenMPStack.back().ScopVars.begin(),
                 E  = OpenMPStack.back().ScopVars.end();
         I != E; ++I) {
        llvm::Value *LV = (*I);
        unsigned oper = dyn_cast<llvm::User>(LV)->getNumOperands();
        while (!isa<llvm::AllocaInst>(LV) && oper>0) {
            LV = cast<llvm::User>(LV)->getOperand(0);
            oper = dyn_cast<llvm::User>(LV)->getNumOperands();
        }
        if (LV == ScopVar) return true;
    }
    return false;
}

void CodeGenModule::OpenMPSupportStackTy::InheritMapPos() {
    int i;
    llvm::SmallVector<OMPStackElemTy, 16>::iterator end = OpenMPStack.end();
    end--;
    for (llvm::SmallVector<OMPStackElemTy, 16>::iterator IS = OpenMPStack.begin(),
                 ES = end;
         IS != ES; ++IS) {
        i = 0;
        for (SmallVector<llvm::Value*,16>::iterator IM  = IS->MapPointers.begin(),
                     EM  = IS->MapPointers.end();
             IM != EM; ++IM) {
            OpenMPStack.back().MapPointers.push_back(IS->MapPointers[i]);
            OpenMPStack.back().MapSizes.push_back(IS->MapSizes[i]);
            OpenMPStack.back().MapQualTypes.push_back(IS->MapQualTypes[i]);
            OpenMPStack.back().MapTypes.push_back(IS->MapTypes[i]);
            OpenMPStack.back().MapPositions.push_back(IS->MapPositions[i]);
            OpenMPStack.back().MapScopes.push_back(IS->MapScopes[i]);
            i++;
        }
    }
}

// Only for test purposes
void CodeGenModule::OpenMPSupportStackTy::PrintMapped(OMPStackElemTy *elem) {

    for (SmallVector<llvm::Value*,16>::iterator I  = elem->MapPointers.begin(),
                 E  = elem->MapPointers.end();
         I != E; ++I) {
        llvm::Value *LV = (*I);
        llvm::errs() << "  Val : " << *LV << "\n";
    }
}

// Only for test purposes
void CodeGenModule::OpenMPSupportStackTy::PrintAllStack() {

    int i = 0;
    llvm::SmallVector<OMPStackElemTy, 16>::iterator end = OpenMPStack.end();

    for (llvm::SmallVector<OMPStackElemTy, 16>::iterator I  = OpenMPStack.begin(),
                 E = end;
         I != E; ++I) {
        llvm::errs() << "Item " << i++ << ":\n";
        PrintMapped(I);
    }
}

bool CodeGenModule::OpenMPSupportStackTy::inLocalScope(llvm::Value *LocalVar) {

    for (SmallVector<llvm::Value*,16>::iterator I  = OpenMPStack.back().LocalVars.begin(),
                 E  = OpenMPStack.back().LocalVars.end();
         I != E; ++I) {

        llvm::Value *LV = (*I);
        unsigned nop = dyn_cast<llvm::User>(LV)->getNumOperands();
        while (!isa<llvm::AllocaInst>(LV) && nop>0) {
            LV = cast<llvm::User>(LV)->getOperand(0);
            nop = dyn_cast<llvm::User>(LV)->getNumOperands();
        }
        if (LV == LocalVar) return true;
    }
    return false;
}


///
/// Emit Decl Ref LValues to construct Spir Functions
///
llvm::Value *CodeGenFunction::EmitSpirDeclRefLValue(const DeclRefExpr *D) {

    const NamedDecl *ND = D->getDecl();
    if (!isa<VarDecl>(ND)) return nullptr;

    const VarDecl *VD = dyn_cast<VarDecl>(ND);
    CharUnits Alignment = getContext().getDeclAlign(ND);
    QualType T = D->getType();

    // check if global Named registers accessed via intrinsics only
    if (VD->getStorageClass() == SC_Register &&
        VD->hasAttr<AsmLabelAttr>() && !VD->isLocalVarDecl()) {
        return (EmitGlobalNamedRegister(VD, CGM, Alignment)).getAddress();
    }

    // DeclRefExpr for a reference initialized by a constant expression
    const Expr *Init = VD->getAnyInitializer(VD);
    if (Init && !isa<ParmVarDecl>(VD) && VD->getType()->isReferenceType()) {
        llvm::Constant *Val =
                CGM.EmitConstantValue(*VD->evaluateValue(), VD->getType(), this);
        assert(Val && "failed to emit reference constant expression");
        return (MakeAddrLValue(Val, T, Alignment)).getAddress();
    }

    if (ND->hasAttr<WeakRefAttr>()) {
        const auto *VD = cast<ValueDecl>(ND);
        llvm::Constant *Aliasee = CGM.GetWeakRefReference(VD);
        return (MakeAddrLValue(Aliasee, T, Alignment)).getAddress();
    }

    if (const auto *VD = dyn_cast<VarDecl>(ND)) {
        // Check if this is a global variable.
        if (VD->hasLinkage() || VD->isStaticDataMember()) {
            return (EmitGlobalVarDeclLValue(*this, D, VD)).getAddress();
        }
        else {
            bool isBlockVariable = VD->hasAttr<BlocksAttr>();
            llvm::Value *V = LocalDeclMap.lookup(VD);
            if (!V && VD->isStaticLocal())
                V = CGM.getStaticLocalDeclAddress(VD);
            if (!V) return nullptr;
            else {
                LValue LV;
                if (isBlockVariable) V = BuildBlockByrefAddress(V, VD);
                if (VD->getType()->isReferenceType()) {
                    llvm::LoadInst *LI = Builder.CreateLoad(V);
                    LI->setAlignment(Alignment.getQuantity());
                    V = LI;
                    LV = MakeNaturalAlignAddrLValue(V, T);
                } else {
                    LV = MakeAddrLValue(V, T, Alignment);
                }
                bool isLocalStorage = VD->hasLocalStorage();
                bool NonGCable = isLocalStorage &&
                                 !VD->getType()->isReferenceType() &&
                                 !isBlockVariable;
                if (NonGCable) {
                    LV.getQuals().removeObjCGCAttr();
                    LV.setNonGC(true);
                }
                bool isImpreciseLifetime = (isLocalStorage && !VD->hasAttr<ObjCPreciseLifetimeAttr>());
                if (isImpreciseLifetime) LV.setARCPreciseLifetime(ARCImpreciseLifetime);
                setObjCGCLValueClass(getContext(), D, LV);
                return LV.getAddress();
            }
            return nullptr;
        }
    }
    return nullptr;
}

///
/// Recursively transverse the body of the for loop looking for uses or assigns.
///
void CodeGenFunction::HandleStmts(Stmt *ST,
                                  llvm::raw_fd_ostream &FOS,
                                  unsigned long &num_args,
                                  bool CLgen) {
  llvm::Value *Status = nullptr;
  if (isa<DeclRefExpr>(ST)) {
    auto *D = dyn_cast<DeclRefExpr>(ST);
    llvm::Value *BodyVar = EmitSpirDeclRefLValue(D);

    if (BodyVar) {
      const NamedDecl *ND = D->getDecl();
      if (!CGM.OpenMPSupport.inLocalScope(BodyVar)) {
        if (CLgen) {
          if (!CGM.OpenMPSupport.isKernelVar(BodyVar)) {
            CGM.OpenMPSupport.addKernelVar(BodyVar);
            llvm::Value *BVRef = Builder.CreateBitCast(BodyVar, CGM.VoidPtrTy);
            llvm::Value *CArg[] = {
                Builder.getInt32(num_args++),
                Builder.getInt32(
                    (dyn_cast<llvm::AllocaInst>(BodyVar)->getAllocatedType())
                        ->getPrimitiveSizeInBits() /
                    8),
                BVRef};
            auto &RT = CGM.getAClangRuntime();
            Status = EmitRuntimeCall(RT.cl_set_kernel_hostArg(), CArg);
            FOS << ",\n";
            FOS << D->getType().getAsString() << " " << ND->getDeclName();
          }
        } else if (!CGM.OpenMPSupport.isScopVar(BodyVar)) {
          CGM.OpenMPSupport.addScopVar(BodyVar);
          scalarMap[ND->getName().str()] = BodyVar;
          FOS << "\t" << D->getType().getAsString() << " " << ND->getDeclName()
              << ";\n";
        }
      }
    }
  }

  // Get the children of the current node in the AST and call the function
  // recursively
  for (Stmt::child_iterator I = ST->child_begin(), E = ST->child_end(); I != E;
       ++I) {
    if (*I != NULL)
      HandleStmts(*I, FOS, num_args, CLgen);
  }
}

///
/// Emit host arg values that will be passed to kernel function
///
llvm::Value *CodeGenFunction::EmitHostParameters(ForStmt *FS,
                                                 llvm::raw_fd_ostream &FOS,
                                                 unsigned long &num_args,
                                                 bool Collapse,
                                                 unsigned loopNest,
                                                 unsigned lastLoop) {

  DiagnosticsEngine &Diags = CGM.getDiags();
  bool compareEqual = false;
  bool isLesser = false;
  bool isIncrement = false;
  llvm::Value *A = nullptr;
  llvm::Value *B = nullptr;
  llvm::Value *C = nullptr;
  llvm::Value *Status = nullptr;
  llvm::Value *IVal = nullptr;
  Expr *init = nullptr;
  std::string initType;

  if (isa<DeclStmt>(FS->getInit())) {
    Diags.Report(FS->getLocStart(), 7)
        << "for statement in Canonical-form only";
    return nullptr;
  } else {
    const BinaryOperator *INIT = dyn_cast<BinaryOperator>(FS->getInit());
    IVal = EmitLValue(dyn_cast<Expr>(INIT)).getAddress();
    init = INIT->getRHS();
    initType = INIT->getType().getAsString();
    CGM.OpenMPSupport.addLocalVar(IVal);
    A = EmitAnyExprToTemp(init).getScalarVal();
  }

  // Check the comparator (<, <=, > or >=)
  BinaryOperator *COND = dyn_cast<BinaryOperator>(FS->getCond());
  switch (COND->getOpcode()) {
  case BO_LT:
    isLesser = true;
    compareEqual = false;
    break;
  case BO_GT:
    isLesser = false;
    compareEqual = false;
    break;
  case BO_LE:
    isLesser = true;
    compareEqual = true;
    break;
  case BO_GE:
    isLesser = false;
    compareEqual = true;
    break;
  default:
    break;
  }

  // Check the increment type (i=i(+/-)incr, i(+/-)=incr, i(++/--))
  Expr *inc = FS->getInc();
  if (isa<CompoundAssignOperator>(inc)) { // i(+/-)=incr
    auto *BO = dyn_cast<BinaryOperator>(inc);
    Expr *incr = BO->getRHS();
    C = EmitAnyExprToTemp(incr).getScalarVal();
    if (BO->getOpcode() == BO_AddAssign)
      isIncrement = true;
    else if (BO->getOpcode() == BO_SubAssign)
      isIncrement = false;
  } else if (isa<BinaryOperator>(inc)) { // i=i(+/-)incr
    Stmt::child_iterator ci = inc->child_begin();
    ci++;
    auto *BO = dyn_cast<BinaryOperator>(*ci);
    Expr *incr = BO->getRHS();
    C = EmitAnyExprToTemp(incr).getScalarVal();
    if (BO->getOpcode() == BO_Add)
      isIncrement = true;
    else if (BO->getOpcode() == BO_Sub)
      isIncrement = false;
  } else if (isa<UnaryOperator>(inc)) { // i(++/--)
    const UnaryOperator *BO = dyn_cast<UnaryOperator>(inc);
    C = Builder.getInt32(1);
    if (BO->isIncrementOp())
      isIncrement = true;
    else if (BO->isDecrementOp())
      isIncrement = false;
  }

  Expr *cond = nullptr;
  if (isIncrement && isLesser)
    cond = COND->getRHS();
  else if (isIncrement && !isLesser)
    cond = COND->getLHS();
  else if (!isIncrement && isLesser)
    cond = COND->getLHS();
  else // !isIncrement && !isLesser
    cond = COND->getRHS();

  B = EmitAnyExprToTemp(cond).getScalarVal();

  llvm::Value *MIN;
  if (isIncrement) {
    MIN = A;
  } else {
    if (compareEqual)
      MIN = B;
    else
      MIN = Builder.CreateAdd(B, Builder.getInt32(1));
  }

  std::string IName = getVarNameAsString(IVal);
  llvm::AllocaInst *AL = Builder.CreateAlloca(B->getType(), NULL);
  AL->setUsedWithInAlloca(true);

  llvm::Value *T;
  if (compareEqual)
    T = Builder.getInt32(0);
  else
    T = Builder.getInt32(1);

  auto &RT = CGM.getAClangRuntime();

  llvm::Value *KArg[] = {A, B, C, T};
  llvm::Value *nCores =
      EmitRuntimeCall(RT.Get_num_cores(), KArg);
  Builder.CreateStore(nCores, AL);

  // Create hostArg to represent _UB_n (i.e., nCores)
  llvm::Value *CVRef = Builder.CreateBitCast(AL, CGM.VoidPtrTy);
  llvm::Value *CArg[] = {
      Builder.getInt32(num_args++),
      Builder.getInt32((AL->getAllocatedType())->getPrimitiveSizeInBits() / 8),
      CVRef};
  Status =
      EmitRuntimeCall(RT.cl_set_kernel_hostArg(), CArg);

  if (Collapse) {
    FOS << initType;
    FOS << " _UB_" << loopNest << ", ";
    FOS << initType;
    FOS << " _MIN_" << loopNest << ", ";

    llvm::AllocaInst *AL2 = Builder.CreateAlloca(B->getType(), NULL);
    AL2->setUsedWithInAlloca(true);
    Builder.CreateStore(MIN, AL2);
    llvm::Value *CVRef2 = Builder.CreateBitCast(AL2, CGM.VoidPtrTy);

    // Create hostArg to represent _MIN_n
    llvm::Value *CArg2[] = {
        Builder.getInt32(num_args++),
        Builder.getInt32((AL2->getAllocatedType())->getPrimitiveSizeInBits() /
                         8),
        CVRef2};
    Status =
        EmitRuntimeCall(RT.cl_set_kernel_hostArg(), CArg2);

    FOS << initType;
    FOS << " _INC_" << loopNest;
    if (loopNest != lastLoop)
      FOS << ",\n";

    llvm::AllocaInst *AL3 = Builder.CreateAlloca(C->getType(), NULL);
    AL2->setUsedWithInAlloca(true);
    Builder.CreateStore(C, AL3);
    llvm::Value *CVRef3 = Builder.CreateBitCast(AL3, CGM.VoidPtrTy);

    // Create hostArg to represent _INC_n
    llvm::Value *CArg3[] = {
        Builder.getInt32(num_args++),
        Builder.getInt32((AL3->getAllocatedType())->getPrimitiveSizeInBits() /
                         8),
        CVRef3};
    Status =
        EmitRuntimeCall(RT.cl_set_kernel_hostArg(), CArg3);
  } else {
    if (isa<BinaryOperator>(FS->getInit())) {
      auto *lInit = dyn_cast<BinaryOperator>(FS->getInit());
      auto *leftExpr = dyn_cast<DeclRefExpr>(lInit->getLHS());
      if (leftExpr) {
        NamedDecl *ND = leftExpr->getDecl();
        FOS << initType << " " << ND->getNameAsString();
      } else
        Diags.Report(FS->getLocStart(), 7)
            << "for statement in Canonical-form only";
    } else
      Diags.Report(FS->getLocStart(), 7)
          << "for statement in Canonical-form only";
  }
  return nCores;
}

///
/// Get the number of loop nest
///
unsigned CodeGenFunction::GetNumNestedLoops(const OMPExecutableDirective &S) {
  unsigned nLoops = 0;
  bool SkippedContainers = false;
  const ForStmt *For;
  const Stmt *Body = S.getAssociatedStmt();
  if (const auto *CS = dyn_cast_or_null<CapturedStmt>(Body)) {
    Body = CS->getCapturedStmt();
  }
  while (!SkippedContainers) {
    if ((For = dyn_cast<ForStmt>(Body))) {
      Body = For->getBody();
      nLoops++;
    } else if (const auto *AS = dyn_cast_or_null<AttributedStmt>(Body)) {
      Body = AS->getSubStmt();
    } else if (const auto *CS = dyn_cast_or_null<CompoundStmt>(Body)) {
      if (CS->size() != 1) {
        SkippedContainers = true;
      } else {
        Body = CS->body_back();
      }
    } else
      SkippedContainers = true;
  }
  return nLoops;
}

void CodeGenFunction::EmitOMPLoopAsCLKernel(const OMPLoopDirective &S) {

  /* marcio */
  llvm::errs() << "EmitOMPLoopAsCLKernel\n";
  /* oicram */

  bool verbose = true;  /* change to false on release */
  bool tile = true;
  std::string tileSize = std::to_string(16); // default tile size
  if (auto *C = S.getSingleClause<OMPScheduleClause>()) {
    if (const auto *Ch = C->getChunkSize()) {
      // We only support chunk expression that folds to a constant
      llvm::APSInt Result;
      if (ConstantFoldsToSimpleInteger(Ch, Result)) {
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
  const std::string AuxName = FileName + ".tmp";

  std::string Error;
  llvm::raw_fd_ostream AXOS(AuxName.c_str(), Error, llvm::sys::fs::F_Text);

  // Add the basic c header files.
  CLOS << "#include <stdlib.h>\n";
  CLOS << "#include <stdint.h>\n";
  CLOS << "#include <math.h>\n\n";

  // use of type 'double' requires cl_khr_fp64 extension to be enabled
  AXOS << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

  std::string includeContents = CGM.OpenMPSupport.getIncludeStr();
  if (includeContents != "") {
    AXOS << includeContents << "\n";
  }

  ArrayRef<llvm::Value *> MapClausePointerValues;
  ArrayRef<llvm::Value *> MapClauseSizeValues;
  ArrayRef<QualType> MapClauseQualTypes;
  ArrayRef<unsigned> MapClauseTypeValues;
  ArrayRef<unsigned> MapClausePositionValues;
  ArrayRef<unsigned> MapClauseScopeValues;

  CGM.OpenMPSupport.getMapPos(MapClausePointerValues, MapClauseSizeValues,
                              MapClauseQualTypes, MapClauseTypeValues,
                              MapClausePositionValues, MapClauseScopeValues);

  // Dump necessary typedefs in scope file (and also in aux file)
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
          // Need to check if RecordDecl was already dumped?
          RD->print(CLOS);
          CLOS << ";\n";
          RD->print(AXOS);
          AXOS << ";\n";
        }

        if (B.isCanonical() && B.getAsString().compare(defty) != 0) {
          CLOS << "typedef " << B.getAsString() << " " << defty << ";\n";
          AXOS << "typedef " << B.getAsString() << " " << defty << ";\n";
        }
      }
    }
  }

  CGM.OpenMPSupport.clearScopVars();
  CGM.OpenMPSupport.clearKernelVars();
  CGM.OpenMPSupport.clearLocalVars();
  scalarMap.clear();

  CLOS << "void foo (\n";
  AXOS << "\n__kernel void " << FileName << " (\n";

  int j = 0;
  bool needComma = false;
  for (ArrayRef<llvm::Value *>::iterator I = MapClausePointerValues.begin(),
                                         E = MapClausePointerValues.end();
       I != E; ++I) {

    llvm::Value *KV = dyn_cast<llvm::User>(*I)->getOperand(0);
    QualType QT = MapClauseQualTypes[j];
    std::string KName = vectorMap[KV];

    CGM.OpenMPSupport.addScopVar(KV);
    CGM.OpenMPSupport.addScopType(QT);
    CGM.OpenMPSupport.addKernelVar(KV);
    CGM.OpenMPSupport.addKernelType(QT);

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

    AXOS << "__global ";

    j++;

    AXOS << QT.getAsString();
    if (needComma)
      CLOS << ",\n";
    CLOS << "\t\t" << QT.getAsString();
    needComma = true;
    if (isPointer) {
      AXOS << " *" << KName << ",\n";
      CLOS << " *" << KName;
    } else {
      AXOS << "  " << KName << ",\n";
      CLOS << "  " << KName;
    }
  }
  CLOS << ") {\n";

  unsigned long num_args = CGM.OpenMPSupport.getKernelVarSize();
  assert (num_args != 0 && "loop is not suitable to execute on GPUs");

  // Traverse the Body looking for all scalar variables declared out of
  // for scope and generate value reference to pass to kernel function
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
  Body->printPretty(CLOS, nullptr, PrintingPolicy(getContext().getLangOpts()),
                    4);
  CLOS << "\n#pragma endscop\n}\n";
  CLOS.close();

  int workSizes[8][3];
  int blockSizes[8][3];
  int kernelId, upperKernel = 0;
  int k = 0;
  std::vector<std::pair<int, std::string>> pName;

  if (!(tile || vectorize )) {
    std::remove(FileName.c_str());
  } else {
    // Change the temporary name to c name
    const std::string cName = FileName + ".c";
    rename(FileName.c_str(), cName.c_str());

    // Construct the pairs of <index, arg> that will be passed to
    // the kernels and sort it in alphabetic order
    for (ArrayRef<llvm::Value *>::iterator I = MapClausePointerValues.begin(),
                                           E = MapClausePointerValues.end();
         I != E; ++I) {

      llvm::Value *PV = dyn_cast<llvm::User>(*I)->getOperand(0);
      pName.push_back(std::pair<int, std::string>(k, vectorMap[PV]));
      k++;
    }
    std::sort(pName.begin(), pName.end(), pairCompare);

    // Try to generate a (possible optimized) kernel version using
    // clang-pcg, a script that invoke Polyhedral Codegen.
    // Get the loop schedule kind and chunk on pragmas:
    //       schedule(dynamic[,chunk]) set --tile-size=chunk
    //       schedule(static[,chunk]) also use no-reschedule
    //       schedule(auto) or none use --tile-size=16
    for (kernelId = 0; kernelId < 8; ++kernelId) {
      for (j = 0; j < 3; j++) {
        workSizes[kernelId][j] = 0;
        blockSizes[kernelId][j] = 0;
      }
      vectorNames[kernelId].clear();
      scalarNames[kernelId].clear();
    }

    std::string ChunkSize = "--tile-size=" + tileSize + " ";
    bool hasScheduleStatic = false;
    for (ArrayRef<OMPClause *>::iterator I = S.clauses().begin(),
                                         E = S.clauses().end();
         I != E; ++I) {
      OpenMPClauseKind ckind = ((*I)->getClauseKind());
      if (ckind == OMPC_schedule) {
        OMPScheduleClause *C = cast<OMPScheduleClause>(*I);
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
      ChunkSize = "--no-reschedule --tile-size=1 --no-shared-memory "
                  "--no-private-memory ";
    } else if (vectorize) {
      // Vector optimization use tile-size=4, the preferred vector size for
      // float. Also, turn off the use of shared & private memories.
      ChunkSize = "--tile-size=4 --no-shared-memory --no-private-memory ";
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
    // verbose preserve temp files (for debug purposes)
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
          vectorNames[kernelId].push_back(
              std::pair<int, std::string>(index, arg_name));
        } else if (kind == 2) {
          scalarNames[kernelId].push_back(
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

  auto &RT = CGM.getAClangRuntime();

  // Emit code to load the file that contain the kernels
  llvm::Value *Status = nullptr;
  llvm::Value *FileStr = Builder.CreateGlobalStringPtr(FileName);
  Status =
      EmitRuntimeCall(RT.cl_create_program(), FileStr);

  // CLgen control whether we need to generate the default kernel code.
  // The polyhedral optimization returns workSizes = 0, meaning that
  // the optimization does not worked. In this case generate naive kernel.
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
        if (scalarMap[(I)->second] == NULL) {
          CLgen = true;
          break;
        }
      }
    }
  }

  if (CLgen) {
    Status =
        EmitRuntimeCall(RT.cl_create_kernel(), FileStr);
    // Get the number of cl_mem args that will be passed first to
    // kernel_function
    int num_args = CGM.OpenMPSupport.getKernelVarSize();
    llvm::Value *Args[] = {Builder.getInt32(num_args)};
    Status =
        EmitRuntimeCall(RT.cl_set_kernel_args(), Args);
  }

  // Look for CollapseNum
  bool hasCollapseClause = false;
  unsigned CollapseNum, loopNest;
  // If Collapse clause is not empty, get the collapsedNum,
  for (ArrayRef<OMPClause *>::iterator I = S.clauses().begin(),
                                       E = S.clauses().end();
       I != E; ++I) {
    OpenMPClauseKind ckind = ((*I)->getClauseKind());
    if (ckind == OMPC_collapse) {
      hasCollapseClause = true;
      CollapseNum = getCollapsedNumberFromLoopDirective(&S);
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

  // Initialize Body to traverse it again, now for AXOS.
  Body = S.getAssociatedStmt();
  if (CapturedStmt *CS = dyn_cast_or_null<CapturedStmt>(Body)) {
    Body = CS->getCapturedStmt();
  }

  if (CLgen) {
    ForStmt *For;
    unsigned nLoops = CollapseNum;
    int loop = 0;
    while (nLoops > 0) {
      For = dyn_cast<ForStmt>(Body);
      if (For) {
        nCores.push_back(EmitHostParameters(For, AXOS, num_args, true, loop,
                                            CollapseNum - 1));
        Body = For->getBody();
        --nLoops;
        loop++;
      } else if (AttributedStmt *AS = dyn_cast<AttributedStmt>(Body)) {
        Body = AS->getSubStmt();
      } else if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Body)) {
        if (CS->size() == 1) {
          Body = CS->body_back();
        } else {
          assert(0 && "Unexpected compound stmt in the loop nest");
        }
      } else {
        assert(0 && "Unexpected stmt in the loop nest");
      }
    }

    assert(Body && "Failed to extract the loop body");

    if (loopNest > CollapseNum) {
      Stmt *Aux = Body;
      while (loopNest > CollapseNum) {
        For = dyn_cast<ForStmt>(Aux);
        int loop = loopNest - 1;
        if (For) {
          AXOS << ",\n";
          EmitHostParameters(For, AXOS, num_args, false, loop, CollapseNum - 1);
          Aux = For->getBody();
          --loopNest;
          loop--;
        } else if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Aux)) {
          if (CS->size() == 1) {
            Aux = CS->body_back();
          } else {
            assert(0 && "Unexpected compound stmt in the loop nest");
          }
        }
      }
    }

    // Traverse again the Body looking for scalar variables declared out of
    // "for" scope and generate value reference to pass to kernel function
    if (Body->getStmtClass() == Stmt::CompoundStmtClass) {
      CompoundStmt *BS = cast<CompoundStmt>(Body);
      for (CompoundStmt::body_iterator I = BS->body_begin(), E = BS->body_end();
           I != E; ++I) {
        HandleStmts(*I, AXOS, num_args, true);
      }
    } else {
      HandleStmts(Body, AXOS, num_args, true);
    }

    AXOS << ") {\n   ";

    for (unsigned i = 0; i < CollapseNum; ++i)
      AXOS << "int _ID_" << i << " = get_global_id(" << i << ");\n   ";

    SmallVector<llvm::Value *, 16> LocalVars;
    CGM.OpenMPSupport.getLocalVars(LocalVars);
    for (unsigned i = 0; i < CollapseNum; ++i) {
      std::string IName = getVarNameAsString(LocalVars[i]);
      AXOS << "int " << IName << " = _INC_" << i;
      AXOS << " * _ID_" << i << " + _MIN_" << i << ";\n   ";
    }

    if (CollapseNum == 1) {
      AXOS << "  if ( _ID_0 < _UB_0 )\n";
    } else if (CollapseNum == 2) {
      AXOS << "  if ( _ID_0 < _UB_0 && _ID_1 < _UB_1 )\n";
    } else {
      AXOS << "  if ( _ID_0 < _UB_0 && _ID_1 < _UB_1 && _ID_2 < _UB_2 )\n";
    }

    if (isa<CompoundStmt>(Body)) {
      Body->printPretty(AXOS, nullptr,
                        PrintingPolicy(getContext().getLangOpts()));
      AXOS << "\n}\n";
    } else {
      AXOS << " {\n";
      Body->printPretty(AXOS, nullptr,
                        PrintingPolicy(getContext().getLangOpts()), 8);
      AXOS << ";\n }\n}\n";
    }

    // Close the kernel file
    AXOS.close();

    // Change the auxiliary name to OpenCL kernel name
    std::rename(AuxName.c_str(), clName.c_str());

  } else {
    // AXOS was not used. Then remove the AuxName associated with it.
    AXOS.close();
    std::remove(AuxName.c_str());
    // Also insert the include contents into the clName, if any.
    std::ofstream outputFile(AuxName);
    std::ifstream inputFile(clName);
    outputFile << includeContents << inputFile.rdbuf();
    inputFile.close();
    outputFile.close();
    std::remove(clName.c_str());
    std::rename(AuxName.c_str(), clName.c_str());
  }

  // Generate kernel with vectorization ?
  if (vectorize) {
    const std::string vectorizer =
        "$LLVM_INCLUDE_PATH/vectorize/vectorize -silent " + clName;
    std::system(vectorizer.c_str());
    if (!verbose) {
      struct stat buffer;
      if (stat(AuxName.c_str(), &buffer) == 0) {
        std::remove(AuxName.c_str());
      }
    }
  }

  // Generate the spir-code ?
  llvm::Triple Tgt = CGM.getLangOpts().OMPtoGPUTriple;
  if (Tgt.getArch() == llvm::Triple::spir ||
      Tgt.getArch() == llvm::Triple::spir64 ||
      Tgt.getArch() == llvm::Triple::spirv) {

    std::string tgtStr;
    if (Tgt.getArch() == llvm::Triple::spirv) {
      // First Generate code for spir64
      tgtStr = "spir64-unknown-unknown";
    } else {
      tgtStr = Tgt.getTriple();
    }

    const std::string bcArg = "clang-3.5 -cc1 -x cl -cl-std=CL1.2 -fno-builtin "
                              "-emit-llvm-bc -triple " +
                              tgtStr +
                              " -include "
                              "$LLVM_INCLUDE_PATH/llvm/SpirTools/opencl_spir.h "
                              "-ffp-contract=off -o " +
                              AuxName + " " + clName;
    std::system(bcArg.c_str());

    const std::string encodeStr =
        "spir-encoder " + AuxName + " " + FileName + ".bc";
    std::system(encodeStr.c_str());
    std::remove(AuxName.c_str());

    if (Tgt.getArch() == llvm::Triple::spirv) {
      // Now convert to spir-v format
      const std::string spirvStr = "llvm-spirv " + FileName + ".bc";
      std::system(spirvStr.c_str());
      if (!verbose) {
        const std::string rmbc = "rm " + FileName + ".bc";
        std::system(rmbc.c_str());
      }
    }
  }

  if (!CLgen) {
    for (kernelId = 0; kernelId <= upperKernel; kernelId++) {
      llvm::Value *KernelStr =
          Builder.CreateGlobalStringPtr(FileName + std::to_string(kernelId));
      Status = EmitRuntimeCall(RT.cl_create_kernel(),
                               KernelStr);

      // Set kernel args according pos & index of buffer, only if required
      k = 0;
      for (std::vector<std::pair<int, std::string>>::iterator I = pName.begin(),
                                                              E = pName.end();
           I != E; ++I) {
        std::vector<std::pair<int, std::string>>::iterator it =
            std::find_if(vectorNames[kernelId].begin(),
                         vectorNames[kernelId].end(), Required((I)->second));
        if (it == vectorNames[kernelId].end()) {
          // the array is not required
        } else {
          llvm::Value *Args[] = {Builder.getInt32(k),
                                 Builder.getInt32((I)->first)};
          Status = EmitRuntimeCall(RT.cl_set_kernel_arg(),
                                   Args);
          k++;
        }
      }

      for (std::vector<std::pair<int, std::string>>::iterator
               I = scalarNames[kernelId].begin(),
               E = scalarNames[kernelId].end();
           I != E; ++I) {
        llvm::Value *BV = scalarMap[(I)->second];
        llvm::Value *BVRef = Builder.CreateBitCast(BV, CGM.VoidPtrTy);
        llvm::Value *CArg[] = {
            Builder.getInt32((I)->first),
            Builder.getInt32(
                (dyn_cast<llvm::AllocaInst>(BV)->getAllocatedType())
                    ->getPrimitiveSizeInBits() /
                8),
            BVRef};
        Status = EmitRuntimeCall(RT.cl_set_kernel_hostArg(), CArg);
      }

      int workDim;
      if (workSizes[kernelId][2] != 0)
        workDim = 3;
      else if (workSizes[kernelId][1] != 0)
        workDim = 2;
      else
        workDim = 1;

      llvm::Value *GroupSize[] = {Builder.getInt32(workSizes[kernelId][0]),
                                  Builder.getInt32(workSizes[kernelId][1]),
                                  Builder.getInt32(workSizes[kernelId][2]),
                                  Builder.getInt32(blockSizes[kernelId][0]),
                                  Builder.getInt32(blockSizes[kernelId][1]),
                                  Builder.getInt32(blockSizes[kernelId][2]),
                                  Builder.getInt32(workDim)};

      Status = EmitRuntimeCall(RT.cl_execute_tiled_kernel(), GroupSize);
    }
  } else {
    if (CollapseNum == 1) {
      nCores.push_back(Builder.getInt32(0));
      nCores.push_back(Builder.getInt32(0));
    } else if (CollapseNum == 2) {
      nCores.push_back(Builder.getInt32(0));
    }
    llvm::Value *WGSize[] = {
        Builder.CreateIntCast(nCores[0], CGM.Int64Ty, false),
        Builder.CreateIntCast(nCores[1], CGM.Int64Ty, false),
        Builder.CreateIntCast(nCores[2], CGM.Int64Ty, false),
        Builder.getInt32(CollapseNum)};
    Status =
        EmitRuntimeCall(RT.cl_execute_kernel(), WGSize);
  }
}

/// Generate an instructions for '#pragma omp parallel for [simd] reduction'
/// directive
void CodeGenFunction::EmitOMPReductionAsCLKernel(const OMPLoopDirective &S) {

  for (const auto *C : S.getClausesOfKind<OMPReductionClause>()) {
    OMPVarListClause<OMPReductionClause> *list =
        cast<OMPVarListClause<OMPReductionClause>>(
            cast<OMPReductionClause>(*I));
    for (auto l = list->varlist_begin(); l != list->varlist_end(); l++) {

      DeclRefExpr *reductionVar = cast<DeclRefExpr>(*l);
      llvm::Value *rv = EmitLValue(*l).getAddress();

      QualType qt = getContext().IntTy;
      llvm::Type *TT1 = ConvertType(qt);
      llvm::Value *T1 = CreateTempAlloca(TT1, "nthreads");

      llvm::Type *BB1 = ConvertType(qt);
      llvm::Value *B1 = CreateTempAlloca(BB1, "nblocks");

      ArrayRef<llvm::Value *> MapClausePointerValues;
      ArrayRef<llvm::Value *> MapClauseSizeValues;
      ArrayRef<QualType> MapClauseQualTypes;
      ArrayRef<unsigned> MapClauseTypeValues;
      ArrayRef<unsigned> MapClausePositionValues;
      ArrayRef<unsigned> MapClauseScopeValues;

      CGM.OpenMPSupport.getMapPos(
          MapClausePointerValues, MapClauseSizeValues, MapClauseQualTypes,
          MapClauseTypeValues, MapClausePositionValues, MapClauseScopeValues);

      int idxInput = 0, idxAux = 1, idxOut = 2, templateId = 1;
      QualType Q = MapClauseQualTypes[idxInput];
      const Type *ty = Q.getTypePtr();
      if (ty->isPointerType() || ty->isReferenceType()) {
        Q = ty->getPointeeType();
      }
      while (Q.getTypePtr()->isArrayType()) {
        Q = dyn_cast<ArrayType>(Q.getTypePtr())->getElementType();
      }
      if (!dumpedDefType(&Q)) {
        std::string defty = Q.getAsString();
        Q = ty->getCanonicalTypeInternal().getTypePtr()->getPointeeType();
        while (Q.getTypePtr()->isArrayType()) {
          Q = dyn_cast<ArrayType>(Q.getTypePtr())->getElementType();
        }
      }

      auto &RT = CGM.getAClangRuntime();

      /* get the number of blocks and threads */
      llvm::Value *Status = nullptr;
      llvm::Type *tR = ConvertType(Q);
      int typeSize = GetTypeSizeInBits(tR);
      llvm::Value *Bytes = Builder.getInt32(typeSize / 8);
      llvm::Value *KArg[] = {T1, B1, MapClauseSizeValues[idxInput], Bytes};
      llvm::Value *ThreadBytes = nullptr;
      ThreadBytes = EmitRuntimeCall(RT.cl_get_threads_blocks_reduction(), KArg);

      /* Offload the Auxiliary array */
      llvm::Value *Size[] = {
          Builder.CreateIntCast(ThreadBytes, CGM.Int64Ty, false)};
      Status =
          EmitRuntimeCall(RT.cl_create_read_write(), Size);

      /* Offload of answer variable*/
      llvm::Value *Size2[] = {Builder.CreateIntCast(Bytes, CGM.Int64Ty, false)};
      Status = EmitRuntimeCall(RT.cl_create_read_write(),
                               Size2);

      /*Fetch the scan variable type and its operator */
      const std::string reductionVarType =
          reductionVar->getType().getAsString();
      OpenMPReductionClauseOperator op =
          cast<OMPReductionClause>(*I)->getOperator();
      const std::string operatorName =
          cast<OMPReductionClause>(*I)->getOpName().getAsString();

      /* Create the unique filename that refers to kernel file */
      llvm::raw_fd_ostream CLOS(CGM.OpenMPSupport.createTempFile(), true);
      const std::string FileNameReduction = CGM.OpenMPSupport.getTempName();
      const std::string clNameReduction = FileNameReduction;

      /* use of type 'double' requires cl_khr_fp64 extension to be enabled */
      CLOS << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n\n";

      /* Dump the IncludeStr, if any */
      std::string includeContents = CGM.OpenMPSupport.getIncludeStr();
      if (includeContents != "") {
        CLOS << includeContents;
        templateId = 2;
      }

      std::string initializer;
      switch (op) {
      case OMPC_REDUCTION_add:
        initializer = "0";
        break;
      case OMPC_REDUCTION_mult:
        initializer = "1";
        break;
      default:
        initializer = "";
      }
      if (initializer == "") {
        // custom initializer is already inserted in include file
        templateId = 1; /* signal user-defined operation */
      } else {
        CLOS << "\n#define _initializer " << initializer;
      }

      CLOS << "\n#define _dataType_ "
           << reductionVarType.substr(0, reductionVarType.find_last_of(' '))
           << "\n";
      CLOS.close();

      /* Generate code to compile the kernel file */
      llvm::Value *FileStrReduction =
          Builder.CreateGlobalStringPtr(clNameReduction);
      Status = EmitRuntimeCall(RT.cl_create_program(), FileStrReduction);

      /* Generate code to build the first Kernel*/
      std::string KernelName = "kernel_0";
      llvm::Value *FunctionKernel_0 = Builder.CreateGlobalStringPtr(KernelName);
      Status = EmitRuntimeCall(RT.cl_create_kernel(), FunctionKernel_0);

      /* Generate code for calling the 1st kernel */
      llvm::Value *Args[] = {Builder.getInt32(0), Builder.getInt32(idxInput)};
      Status =
          EmitRuntimeCall(RT.cl_set_kernel_arg(), Args);
      llvm::Value *Args2[] = {Builder.getInt32(1), Builder.getInt32(idxAux)};
      Status =
          EmitRuntimeCall(RT.cl_set_kernel_arg(), Args2);
      llvm::Value *BVReduction = Builder.CreateBitCast(T1, CGM.VoidPtrTy);
      llvm::Value *CArgReduction[] = {
          Builder.getInt32(2),
          Builder.getInt32((dyn_cast<llvm::AllocaInst>(T1)->getAllocatedType())
                               ->getPrimitiveSizeInBits() /
                           8),
          BVReduction};
      Status = EmitRuntimeCall(RT.cl_set_kernel_hostArg(), CArgReduction);

      llvm::Value *LB = Builder.CreateLoad(B1);
      llvm::Value *LST = Builder.CreateLoad(T1);
      llvm::Value *GroupSize[] = {
          Builder.CreateIntCast(LB, CGM.Int32Ty, false),
          Builder.getInt32(0),
          Builder.getInt32(0),
          Builder.CreateIntCast(LST, CGM.Int32Ty, false),
          Builder.getInt32(0),
          Builder.getInt32(0),
          Builder.getInt32(1)};
      Status = EmitRuntimeCall(RT.cl_execute_tiled_kernel(), GroupSize);

      /* Generate code for calling the 2nd kernel */
      KernelName = "kernel_1";
      llvm::Value *FunctionKernel_1 = Builder.CreateGlobalStringPtr(KernelName);
      Status = EmitRuntimeCall(RT.cl_create_kernel(), FunctionKernel_1);
      llvm::Value *Args3[] = {Builder.getInt32(0), Builder.getInt32(idxAux)};
      Status =
          EmitRuntimeCall(RT.cl_set_kernel_arg(), Args3);
      llvm::Value *Args4[] = {Builder.getInt32(1), Builder.getInt32(idxOut)};
      Status =
          EmitRuntimeCall(RT.cl_set_kernel_arg(), Args4);
      llvm::Value *BVReduction2 = Builder.CreateBitCast(B1, CGM.VoidPtrTy);
      llvm::Value *CArgReduction2[] = {
          Builder.getInt32(2),
          Builder.getInt32((dyn_cast<llvm::AllocaInst>(B1)->getAllocatedType())
                               ->getPrimitiveSizeInBits() /
                           8),
          BVReduction2};
      Status = EmitRuntimeCall(RT.cl_set_kernel_hostArg(), CArgReduction2);

      llvm::Value *LSB = Builder.CreateLoad(B1);
      llvm::Value *GroupSize2[] = {
          Builder.getInt32(1), Builder.getInt32(0),
          Builder.getInt32(0), Builder.CreateIntCast(LSB, CGM.Int32Ty, false),
          Builder.getInt32(0), Builder.getInt32(0),
          Builder.getInt32(1)};
      Status = EmitRuntimeCall(RT.cl_execute_tiled_kernel(), GroupSize2);

      llvm::Value *Res[] = {Builder.CreateIntCast(Bytes, CGM.Int64Ty, false),
                            Builder.getInt32(idxOut),
                            Builder.CreateBitCast(rv, CGM.VoidPtrTy)};
      Status = EmitRuntimeCall(RT.cl_read_buffer(), Res);

      /* release the Aux buffer */
      llvm::Value *Aux[] = {Builder.getInt32(idxAux)};
      Status =
          EmitRuntimeCall(RT.cl_release_buffer(), Aux);
      llvm::Value *Aux2[] = {Builder.getInt32(idxOut)};
      Status =
          EmitRuntimeCall(RT.cl_release_buffer(), Aux2);

      /* Build the kernel file */
      const std::string generator =
          "$LLVM_INCLUDE_PATH/scan/reductiongenerator " + FileNameReduction +
          " " + std::to_string(templateId) + " '" + operatorName + "' ";
      std::system(generator.c_str());
    }
  }
  return;
}
