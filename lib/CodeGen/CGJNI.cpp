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
    jbyteTy = CGT.ConvertType(jbyteQTy);
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
  /// \brief Call to jbyteArray NewByteArray(JNIEnv *env, jsize len)
  OMPRTL_JNI__NewByteArray,
  /// \brief Call to void ReleaseByteArrayElements(JNIEnv *env, jbyteArray
  /// array, jbyte
  /// *elems, jint mode)
  OMPRTL_JNI__ReleaseByteArrayElements,
  /// \brief Call to jbyte *GetByteArrayElements(JNIEnv *env, jbyteArray array,
  /// jboolean
  /// *isCopy)
  OMPRTL_JNI__GetByteArrayElements,
  /// \brief Call to void ReleasePrimitiveArrayCritical(JNIEnv *env, jarray
  /// array, void
  /// *carray, jint mode)
  OMPRTL_JNI__ReleasePrimitiveArrayCritical,
  /// \brief Call to void *GetPrimitiveArrayCritical(JNIEnv *env, jarray array,
  /// jboolean
  /// *isCopy)
  OMPRTL_JNI__GetPrimitiveArrayCritical,
  /// \brief Call to jobject NewObject(JNIEnv *env, jint size, jbyteArray
  /// *arrays);
  OMPRTL_JNI__CreateNewTuple,
};

/// \brief Returns specified OpenMP runtime function for the current OpenMP
/// implementation.  Specialized for the NVPTX device.
/// \param Function OpenMP runtime function.
/// \return Specified function.
llvm::Constant *
CGOpenMPRuntimeSpark::createJNIRuntimeFunction(unsigned Function) {
  llvm::Constant *RTLFn = nullptr;

  switch (static_cast<OpenMPRTLFunctionJNI>(Function)) {
  case OMPRTL_JNI__NewByteArray: {
    // Build jbyteArray __jni_NewByteArray(JNIEnv *env, jsize len);
    llvm::Type *TypeParams[] = {JNIEnvTy, jsizeTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(jbyteArrayTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__jni_NewByteArray");
    break;
  }
  case OMPRTL_JNI__ReleaseByteArrayElements: {
    // Build void __jni_ReleaseByteArrayElements(JNIEnv *env, jbyteArray array,
    // jbyte *elems, jint mode);
    llvm::Type *TypeParams[] = {JNIEnvTy, jbyteArrayTy, jbyteTy->getPointerTo(),
                                jintTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__jni_ReleaseByteArrayElements");
    break;
  }
  case OMPRTL_JNI__GetByteArrayElements: {
    // Build jbyte *__jni_GetByteArrayElements(JNIEnv *env, jbyteArray array,
    // jboolean *isCopy);
    llvm::Type *TypeParams[] = {JNIEnvTy, jbyteArrayTy,
                                jbooleanTy->getPointerTo()};
    llvm::FunctionType *FnTy = llvm::FunctionType::get(
        jbyteTy->getPointerTo(), TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__jni_GetByteArrayElements");
    break;
  }
  case OMPRTL_JNI__ReleasePrimitiveArrayCritical: {
    // Build void __jni_ReleasePrimitiveArrayCritical(JNIEnv *env, jarray array,
    // void *carray, jint mode);
    llvm::Type *TypeParams[] = {JNIEnvTy, jarrayTy, CGM.VoidPtrTy, jintTy};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
    RTLFn =
        CGM.CreateRuntimeFunction(FnTy, "__jni_ReleasePrimitiveArrayCritical");
    break;
  }
  case OMPRTL_JNI__GetPrimitiveArrayCritical: {
    // Build void *__jni_GetPrimitiveArrayCritical(JNIEnv *env, jarray array,
    // jboolean *isCopy);
    llvm::Type *TypeParams[] = {JNIEnvTy, jarrayTy, jbooleanTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(CGM.VoidPtrTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__jni_GetPrimitiveArrayCritical");
    break;
  }
  case OMPRTL_JNI__CreateNewTuple: {
    // Build jobject __jni_CreateNewTuple(JNIEnv *env, jint size, jbyteArray
    // *arrays);
    llvm::Type *TypeParams[] = {JNIEnvTy, jintTy, jbyteArrayTy->getPointerTo()};
    llvm::FunctionType *FnTy =
        llvm::FunctionType::get(jobjectTy, TypeParams, /*isVarArg*/ false);
    RTLFn = CGM.CreateRuntimeFunction(FnTy, "__jni_CreateNewTuple");
    break;
  }
  }

  return RTLFn;
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNINewByteArray(CodeGenFunction &CGF,
                                                       llvm::Value *Env,
                                                       llvm::Value *Size) {
  // Build call __jni_NewByteArray(jsize len)
  llvm::Value *Args[] = {
      Env, CGF.Builder.CreateIntCast(Size, CGF.Int32Ty, /*isSigned*/ true)};
  return CGF.EmitRuntimeCall(createJNIRuntimeFunction(OMPRTL_JNI__NewByteArray),
                             Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNIReleaseByteArrayElements(
    CodeGenFunction &CGF, llvm::Value *Env, llvm::Value *Array,
    llvm::Value *Elems, llvm::Value *Mode) {
  // Build call __jni_ReleaseByteArrayElements(JNIEnv *env, jbyteArray array,
  // jbyte *elems, jint mode)
  llvm::Value *Args[] = {Env, Array, Elems, Mode};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__ReleaseByteArrayElements), Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNIGetByteArrayElements(
    CodeGenFunction &CGF, llvm::Value *Env, llvm::Value *Array,
    llvm::Value *IsCopy) {
  // Build call __jni_GetByteArrayElements(JNIEnv *env, jbyteArray array,
  // jboolean *isCopy)
  llvm::Value *Args[] = {Env, Array, IsCopy};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__GetByteArrayElements), Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNIReleasePrimitiveArrayCritical(
    CodeGenFunction &CGF, llvm::Value *Env, llvm::Value *Array,
    llvm::Value *Carray, llvm::Value *Mode) {
  // Build call __jni_ReleasePrimitiveArrayCritical(JNIEnv *env, jarray array,
  // void *carray, jint mode)
  llvm::Value *Args[] = {Env, Array, Carray, Mode};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__ReleasePrimitiveArrayCritical),
      Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNIGetPrimitiveArrayCritical(
    CodeGenFunction &CGF, llvm::Value *Env, llvm::Value *Array,
    llvm::Value *IsCopy) {
  // Build call __jni_GetPrimitiveArrayCritical(JNIEnv *env, jarray array,
  // jboolean *isCopy)
  llvm::Value *Args[] = {Env, Array, IsCopy};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__GetPrimitiveArrayCritical), Args);
}

llvm::Value *CGOpenMPRuntimeSpark::EmitJNICreateNewTuple(
    CodeGenFunction &CGF, llvm::Value *Env, ArrayRef<llvm::Value *> Elements) {
  auto *Size = CGF.Builder.getInt32(Elements.size());
  llvm::Value *Array = CGF.Builder.CreateAlloca(jbyteArrayTy, Size);
  for (unsigned i = 0; i < Elements.size(); i++) {
    llvm::Value *CurrIndex = CGF.Builder.getInt32(i);
    llvm::Value *CurrElement =
        CGF.Builder.CreateBitCast(Elements[i], jbyteArrayTy);
    llvm::Value *GepPtr = CGF.Builder.CreateGEP(Array, CurrIndex);
    CGF.Builder.CreateAlignedStore(CurrElement, GepPtr, CGF.getPointerAlign());
  }
  llvm::Value *Args[] = {Env, Size, Array};
  return CGF.EmitRuntimeCall(
      createJNIRuntimeFunction(OMPRTL_JNI__CreateNewTuple), Args);
}
