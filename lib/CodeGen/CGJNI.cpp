//===--------- CGJNI.cpp - Emit LLVM Code for declarations ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Decl nodes as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntimeSpark.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypeBuilder.h"

using namespace clang;
using namespace CodeGen;

void CGOpenMPRuntimeSpark::BuildJNITy() {
  if (jintQTy.isNull()) {
    ASTContext &C = CGM.getContext();
    jsizeQTy =
        C.buildImplicitTypedef(C.getSizeType(), "jsize")->getUnderlyingType();
    jbyteQTy =
        C.buildImplicitTypedef(C.SignedCharTy, "jbyte")->getUnderlyingType();
    jbooleanQTy =
        C.buildImplicitTypedef(C.UnsignedCharTy, "jsize")->getUnderlyingType();
    jintQTy = C.buildImplicitTypedef(C.IntTy, "jint")->getUnderlyingType();
    jlongQTy = C.buildImplicitTypedef(C.LongTy, "jlong")->getUnderlyingType();

    _jobjectQTy = C.getRecordType(C.buildImplicitRecord("_jobject"));
    jobjectQTy =
        C.buildImplicitTypedef(C.getPointerType(_jobjectQTy), "jobject")
            ->getUnderlyingType();
    jarrayQTy =
        C.buildImplicitTypedef(jobjectQTy, "jarray")->getUnderlyingType();
    jbyteArrayQTy =
        C.buildImplicitTypedef(jobjectQTy, "jbyteArray")->getUnderlyingType();
    jclassQTy =
        C.buildImplicitTypedef(jobjectQTy, "jclass")->getUnderlyingType();

    // struct _jmethodID;
    // typedef struct _jmethodID *jmethodID;
    _jmethodIDQTy = C.getRecordType(C.buildImplicitRecord("_jmethodID"));
    jmethodIDQTy =
        C.buildImplicitTypedef(C.getPointerType(_jmethodIDQTy), "jmethodIDQTy")
            ->getUnderlyingType();

    // struct JNINativeInterface_;
    // typedef const struct JNINativeInterface_ *JNIEnv;
    _JNINativeInterfaceQTy =
        C.getRecordType(C.buildImplicitRecord("JNINativeInterface_"));
    JNIEnvQTy = C.buildImplicitTypedef(C.getPointerType(_JNINativeInterfaceQTy),
                                       "JNIEnv")
                    ->getUnderlyingType();

    CodeGenTypes &CGT = CGM.getTypes();

    jintTy = CGT.ConvertType(jintQTy);
    jsizeTy = CGT.ConvertType(jsizeQTy);
    jbooleanTy = CGT.ConvertType(jbooleanQTy);
    jlongTy = CGT.ConvertType(jlongQTy);
    jobjectTy = CGT.ConvertType(jobjectQTy);
    jarrayTy = CGT.ConvertType(jarrayQTy);
    jbyteArrayTy = CGT.ConvertType(jbyteArrayQTy);
    jclassTy = CGT.ConvertType(jclassQTy);
    jmethodIDTy = CGT.ConvertType(jmethodIDQTy);
    JNIEnvTy = CGT.ConvertType(JNIEnvQTy);
  }
}

enum OpenMPRTLFunctionJNI {
  /// \brief Call to jbyteArray NewByteArray(jsize len)
  OMPRTL_JNI__NewByteArray,
  /// \brief Call to void ReleaseByteArrayElements(jbyteArray array, jbyte
  /// *elems, jint mode)
  OMPRTL_JNI__ReleaseByteArrayElements,
  /// \brief Call to jbyte *GetByteArrayElements(jbyteArray array, jboolean
  /// *isCopy)
  OMPRTL_JNI__GetByteArrayElements,
  /// \brief Call to void ReleasePrimitiveArrayCritical(jarray array, void
  /// *carray, jint mode)
  OMPRTL_JNI__ReleasePrimitiveArrayCritical,
  /// \brief Call to void *GetPrimitiveArrayCritical(jarray array, jboolean
  /// *isCopy)
  OMPRTL_JNI__GetPrimitiveArrayCritical,
  /// \brief Call to jclass FindClass(const char *name);
  OMPRTL_JNI__FindClass,
  /// \brief Call to jmethodID GetMethodID(jclass clazz, const char *name, const
  /// char *sig)
  OMPRTL_JNI__GetMethodID,
  /// \brief Call to jobject NewObject(jclass clazz, jmethodID methodID, ...);
  OMPRTL_JNI__NewObject,
};

/// \brief Returns specified OpenMP runtime function for the current OpenMP
/// implementation.  Specialized for the NVPTX device.
/// \param Function OpenMP runtime function.
/// \return Specified function.
llvm::Constant *
CGOpenMPRuntimeSpark::createJNIRuntimeFunction(unsigned Function) {
  ASTContext &Ctx = CGM.getContext();
  CodeGenTypes &CGT = CGM.getTypes();
  llvm::Constant *RTLFn = nullptr;

  switch (static_cast<OpenMPRTLFunctionJNI>(Function)) {
  case OMPRTL_JNI__NewByteArray: {
    // Build jbyteArray NewByteArray(jsize len);
    llvm::Type *TypeParams[] = {jsizeTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(jbyteArrayTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "NewByteArray");
    break;
  }
  case OMPRTL_JNI__ReleaseByteArrayElements: {
    // Build void ReleaseByteArrayElements(jbyteArray array, jbyte *elems, jint
    // mode);
    llvm::Type *TypeParams[] = {jbyteArrayTy, jbyteTy, jintTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "ReleaseByteArrayElements");
    break;
  }
  case OMPRTL_JNI__GetByteArrayElements: {
    // Build jbyte *GetByteArrayElements(jbyteArray array, jboolean *isCopy);
    llvm::Type *TypeParams[] = {jbyteArrayTy, jbooleanTy->getPointerTo()};
    llvm::FunctionType *FnTy = llvm::FunctionType::get(
        jbyteTy->getPointerTo(), TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "GetByteArrayElements");
    break;
  }
  case OMPRTL_JNI__ReleasePrimitiveArrayCritical: {
    // Build void ReleasePrimitiveArrayCritical(jarray array, void *carray, jint
    // mode);
    llvm::Type *TypeParams[] = {jarrayTy, CGM.VoidPtrTy, jintTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "ReleasePrimitiveArrayCritical");
    break;
  }
  case OMPRTL_JNI__GetPrimitiveArrayCritical: {
    // Build void *GetPrimitiveArrayCritical(jarray array, jboolean *isCopy);
    llvm::Type *TypeParams[] = {jarrayTy, jbooleanTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "GetPrimitiveArrayCritical");
    break;
  }
  case OMPRTL_JNI__FindClass: {
    // Build jclass FindClass(const char *name);
    llvm::Type *TypeParams[] = {CGT.ConvertType(Ctx.getUIntPtrType())};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(jclassTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "FindClass");
    break;
  }
  case OMPRTL_JNI__GetMethodID: {
    // Build jmethodID GetMethodID(jclass clazz, const char *name, const char
    // *sig);
    llvm::Type *TypeParams[] = {jclassTy, CGT.ConvertType(Ctx.getUIntPtrType()),
                                CGT.ConvertType(Ctx.getUIntPtrType())};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(jmethodIDTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "GetMethodID");
    break;
  }
  case OMPRTL_JNI__NewObject: {
    // Build jobject NewObject(jclass clazz, jmethodID methodID, ...);
    llvm::Type *TypeParams[] = {jclassTy, jmethodIDTy,
                                CGT.ConvertType(Ctx.getBuiltinVaListType())};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(jobjectTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "GetMethodID");
    break;
  }
  }

  return RTLFn;
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNINewByteArray(CodeGenFunction &CGF,
                                                       llvm::Value *Size) {
  // Build call NewByteArray(jsize len)
  llvm::Value *Args[] = {
      CGF.Builder.CreateIntCast(Size, CGF.Int32Ty, /*isSigned*/ true)};
  return CGF.EmitRuntimeCall(createJNIRuntimeFunction(OMPRTL_JNI__NewByteArray),
                             Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNIReleaseByteArrayElements(
    CodeGenFunction &CGF, llvm::Value *Array, llvm::Value *Elems,
    llvm::Value *Mode) {
  // Build call
  // ReleaseByteArrayElements(jbyteArray array, jbyte *elems, jint mode)
  llvm::Value *Args[] = {Array, Elems, Mode};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__ReleaseByteArrayElements), Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNIGetByteArrayElements(
    CodeGenFunction &CGF, llvm::Value *Array, llvm::Value *IsCopy) {
  // Build call GetByteArrayElements(jbyteArray array, jboolean *isCopy)
  llvm::Value *Args[] = {Array, IsCopy};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__GetByteArrayElements), Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNIReleasePrimitiveArrayCritical(
    CodeGenFunction &CGF, llvm::Value *Array, llvm::Value *Carray,
    llvm::Value *Mode) {
  // Build call
  // ReleasePrimitiveArrayCritical(jarray array, void *carray, jint mode)
  llvm::Value *Args[] = {Array, Carray, Mode};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__ReleasePrimitiveArrayCritical),
      Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNIGetPrimitiveArrayCritical(
    CodeGenFunction &CGF, llvm::Value *Array, llvm::Value *IsCopy) {
  // Build call GetPrimitiveArrayCritical(jarray array, jboolean *isCopy)
  llvm::Value *Args[] = {Array, IsCopy};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__GetPrimitiveArrayCritical), Args);
}

llvm::Value *
CGOpenMPRuntimeSpark::EmitJNICreateNewTuple(CodeGenFunction &CGF,
                                            ArrayRef<llvm::Value *> Elements) {
  // Build call FindClass(JNIEnv *env, const char *name);
  // TODO: args
  llvm::Value *Name = nullptr;
  llvm::Value *ArgsClassObject[] = {Name};
  llvm::Value *ClassObject = CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__FindClass), ArgsClassObject);
  // Build call GetMethodID(jclass clazz, const char *name, const char *sig);
  // TODO: args
  llvm::Value *InitName = nullptr;
  llvm::Value *Sig = nullptr;
  llvm::Value *ArgsMethodId[] = {ClassObject, InitName, Sig};
  llvm::Value *MethodId = CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__GetMethodID), ArgsMethodId);
  // T
  SmallVector<llvm::Constant *, 16> ConstElements;
  for (auto E : Elements)
    ConstElements.push_back(cast<llvm::Constant>(E));
  // Build call NewObject(jclass clazz, jmethodID methodID, ...)
  auto *ElementArray = llvm::ConstantArray::get(
      llvm::ArrayType::get(jobjectTy, Elements.size()), ConstElements);
  llvm::Value *Args[] = {ClassObject, MethodId, ElementArray};
  return CGF.EmitRuntimeCall(createJNIRuntimeFunction(OMPRTL_JNI__NewObject),
                             Args);
}
