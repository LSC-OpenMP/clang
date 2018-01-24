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

void CGOpenMPRuntimeSpark::DefineJNITypes() {
  llvm::Module *mod = &CGM.getModule();

  // Type Definitions
  llvm::StructType *StructTy_struct_JNINativeInterface_ =
      mod->getTypeByName("struct.JNINativeInterface_");
  if (!StructTy_struct_JNINativeInterface_) {
    StructTy_struct_JNINativeInterface_ = llvm::StructType::create(
        mod->getContext(), "struct.JNINativeInterface_");
  }
  std::vector<llvm::Type *> StructTy_struct_JNINativeInterface__fields;
  llvm::PointerType *PointerTy_3 =
      llvm::PointerType::get(llvm::IntegerType::get(mod->getContext(), 8), 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_3);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_3);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_3);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_3);
  std::vector<llvm::Type *> FuncTy_5_args;
  llvm::PointerType *PointerTy_2 =
      llvm::PointerType::get(StructTy_struct_JNINativeInterface_, 0);

  llvm::PointerType *PointerTy_1 = llvm::PointerType::get(PointerTy_2, 0);

  FuncTy_5_args.push_back(PointerTy_1);
  llvm::FunctionType *FuncTy_5 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_5_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_4 = llvm::PointerType::get(FuncTy_5, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_4);
  std::vector<llvm::Type *> FuncTy_7_args;
  FuncTy_7_args.push_back(PointerTy_1);
  FuncTy_7_args.push_back(PointerTy_3);
  llvm::StructType *StructTy_struct__jobject =
      mod->getTypeByName("struct._jobject");
  if (!StructTy_struct__jobject) {
    StructTy_struct__jobject =
        llvm::StructType::create(mod->getContext(), "struct._jobject");
  }

  llvm::PointerType *PointerTy_8 =
      llvm::PointerType::get(StructTy_struct__jobject, 0);

  FuncTy_7_args.push_back(PointerTy_8);
  FuncTy_7_args.push_back(PointerTy_3);
  FuncTy_7_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_7 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_7_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_6 = llvm::PointerType::get(FuncTy_7, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_6);
  std::vector<llvm::Type *> FuncTy_10_args;
  FuncTy_10_args.push_back(PointerTy_1);
  FuncTy_10_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_10 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_10_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_9 = llvm::PointerType::get(FuncTy_10, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_9);
  std::vector<llvm::Type *> FuncTy_12_args;
  FuncTy_12_args.push_back(PointerTy_1);
  FuncTy_12_args.push_back(PointerTy_8);
  llvm::StructType *StructTy_struct__jmethodID =
      mod->getTypeByName("struct._jmethodID");
  if (!StructTy_struct__jmethodID) {
    StructTy_struct__jmethodID =
        llvm::StructType::create(mod->getContext(), "struct._jmethodID");
  }

  llvm::PointerType *PointerTy_13 =
      llvm::PointerType::get(StructTy_struct__jmethodID, 0);

  llvm::FunctionType *FuncTy_12 = llvm::FunctionType::get(
      /*Result=*/PointerTy_13,
      /*Params=*/FuncTy_12_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_11 = llvm::PointerType::get(FuncTy_12, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_11);
  std::vector<llvm::Type *> FuncTy_15_args;
  FuncTy_15_args.push_back(PointerTy_1);
  FuncTy_15_args.push_back(PointerTy_8);
  llvm::StructType *StructTy_struct__jfieldID =
      mod->getTypeByName("struct._jfieldID");
  if (!StructTy_struct__jfieldID) {
    StructTy_struct__jfieldID =
        llvm::StructType::create(mod->getContext(), "struct._jfieldID");
  }

  llvm::PointerType *PointerTy_16 =
      llvm::PointerType::get(StructTy_struct__jfieldID, 0);

  llvm::FunctionType *FuncTy_15 = llvm::FunctionType::get(
      /*Result=*/PointerTy_16,
      /*Params=*/FuncTy_15_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_14 = llvm::PointerType::get(FuncTy_15, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_14);
  std::vector<llvm::Type *> FuncTy_18_args;
  FuncTy_18_args.push_back(PointerTy_1);
  FuncTy_18_args.push_back(PointerTy_8);
  FuncTy_18_args.push_back(PointerTy_13);
  FuncTy_18_args.push_back(llvm::IntegerType::get(mod->getContext(), 8));
  llvm::FunctionType *FuncTy_18 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_18_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_17 = llvm::PointerType::get(FuncTy_18, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_17);
  std::vector<llvm::Type *> FuncTy_20_args;
  FuncTy_20_args.push_back(PointerTy_1);
  FuncTy_20_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_20 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_20_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_19 = llvm::PointerType::get(FuncTy_20, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_19);
  std::vector<llvm::Type *> FuncTy_22_args;
  FuncTy_22_args.push_back(PointerTy_1);
  FuncTy_22_args.push_back(PointerTy_8);
  FuncTy_22_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_22 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_22_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_21 = llvm::PointerType::get(FuncTy_22, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_21);
  std::vector<llvm::Type *> FuncTy_24_args;
  FuncTy_24_args.push_back(PointerTy_1);
  FuncTy_24_args.push_back(PointerTy_8);
  FuncTy_24_args.push_back(PointerTy_16);
  FuncTy_24_args.push_back(llvm::IntegerType::get(mod->getContext(), 8));
  llvm::FunctionType *FuncTy_24 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_24_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_23 = llvm::PointerType::get(FuncTy_24, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_23);
  std::vector<llvm::Type *> StructTy_26_fields;
  llvm::StructType *StructTy_26 = llvm::StructType::get(
      mod->getContext(), StructTy_26_fields, /*isPacked=*/false);

  llvm::PointerType *PointerTy_25 = llvm::PointerType::get(StructTy_26, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_25);
  std::vector<llvm::Type *> FuncTy_28_args;
  FuncTy_28_args.push_back(PointerTy_1);
  FuncTy_28_args.push_back(PointerTy_8);
  FuncTy_28_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_28 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_28_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_27 = llvm::PointerType::get(FuncTy_28, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_27);
  std::vector<llvm::Type *> FuncTy_30_args;
  FuncTy_30_args.push_back(PointerTy_1);
  llvm::FunctionType *FuncTy_30 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_30_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_29 = llvm::PointerType::get(FuncTy_30, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_29);
  std::vector<llvm::Type *> FuncTy_32_args;
  FuncTy_32_args.push_back(PointerTy_1);
  llvm::FunctionType *FuncTy_32 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_32_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_31 = llvm::PointerType::get(FuncTy_32, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_31);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_31);
  std::vector<llvm::Type *> FuncTy_34_args;
  FuncTy_34_args.push_back(PointerTy_1);
  FuncTy_34_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_34 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_34_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_33 = llvm::PointerType::get(FuncTy_34, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_33);
  std::vector<llvm::Type *> FuncTy_36_args;
  FuncTy_36_args.push_back(PointerTy_1);
  FuncTy_36_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_36 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_36_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_35 = llvm::PointerType::get(FuncTy_36, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_35);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_19);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_19);
  std::vector<llvm::Type *> FuncTy_38_args;
  FuncTy_38_args.push_back(PointerTy_1);
  FuncTy_38_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_38 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_38_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_37 = llvm::PointerType::get(FuncTy_38, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_37);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_37);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_21);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_19);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_35);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_19);
  std::vector<llvm::Type *> FuncTy_40_args;
  FuncTy_40_args.push_back(PointerTy_1);
  FuncTy_40_args.push_back(PointerTy_8);
  FuncTy_40_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_40 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_40_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_39 = llvm::PointerType::get(FuncTy_40, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_39);
  std::vector<llvm::Type *> FuncTy_42_args;
  FuncTy_42_args.push_back(PointerTy_1);
  FuncTy_42_args.push_back(PointerTy_8);
  FuncTy_42_args.push_back(PointerTy_13);
  llvm::StructType *StructTy_struct___va_list_tag =
      mod->getTypeByName("struct.__va_list_tag");
  if (!StructTy_struct___va_list_tag) {
    StructTy_struct___va_list_tag =
        llvm::StructType::create(mod->getContext(), "struct.__va_list_tag");
  }
  std::vector<llvm::Type *> StructTy_struct___va_list_tag_fields;
  StructTy_struct___va_list_tag_fields.push_back(
      llvm::IntegerType::get(mod->getContext(), 32));
  StructTy_struct___va_list_tag_fields.push_back(
      llvm::IntegerType::get(mod->getContext(), 32));
  StructTy_struct___va_list_tag_fields.push_back(PointerTy_3);
  StructTy_struct___va_list_tag_fields.push_back(PointerTy_3);
  if (StructTy_struct___va_list_tag->isOpaque()) {
    StructTy_struct___va_list_tag->setBody(StructTy_struct___va_list_tag_fields,
                                           /*isPacked=*/false);
  }

  llvm::PointerType *PointerTy_43 =
      llvm::PointerType::get(StructTy_struct___va_list_tag, 0);

  FuncTy_42_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_42 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_42_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_41 = llvm::PointerType::get(FuncTy_42, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_41);
  std::vector<llvm::Type *> FuncTy_45_args;
  FuncTy_45_args.push_back(PointerTy_1);
  FuncTy_45_args.push_back(PointerTy_8);
  FuncTy_45_args.push_back(PointerTy_13);
  llvm::StructType *StructTy_union_jvalue = mod->getTypeByName("union.jvalue");
  if (!StructTy_union_jvalue) {
    StructTy_union_jvalue =
        llvm::StructType::create(mod->getContext(), "union.jvalue");
  }
  std::vector<llvm::Type *> StructTy_union_jvalue_fields;
  StructTy_union_jvalue_fields.push_back(
      llvm::IntegerType::get(mod->getContext(), 64));
  if (StructTy_union_jvalue->isOpaque()) {
    StructTy_union_jvalue->setBody(StructTy_union_jvalue_fields,
                                   /*isPacked=*/false);
  }

  llvm::PointerType *PointerTy_46 =
      llvm::PointerType::get(StructTy_union_jvalue, 0);

  FuncTy_45_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_45 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_45_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_44 = llvm::PointerType::get(FuncTy_45, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_44);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_19);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_21);
  std::vector<llvm::Type *> FuncTy_48_args;
  FuncTy_48_args.push_back(PointerTy_1);
  FuncTy_48_args.push_back(PointerTy_8);
  FuncTy_48_args.push_back(PointerTy_3);
  FuncTy_48_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_48 = llvm::FunctionType::get(
      /*Result=*/PointerTy_13,
      /*Params=*/FuncTy_48_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_47 = llvm::PointerType::get(FuncTy_48, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_47);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_39);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_41);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_44);
  std::vector<llvm::Type *> FuncTy_50_args;
  FuncTy_50_args.push_back(PointerTy_1);
  FuncTy_50_args.push_back(PointerTy_8);
  FuncTy_50_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_50 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_50_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_49 = llvm::PointerType::get(FuncTy_50, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_49);
  std::vector<llvm::Type *> FuncTy_52_args;
  FuncTy_52_args.push_back(PointerTy_1);
  FuncTy_52_args.push_back(PointerTy_8);
  FuncTy_52_args.push_back(PointerTy_13);
  FuncTy_52_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_52 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_52_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_51 = llvm::PointerType::get(FuncTy_52, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_51);
  std::vector<llvm::Type *> FuncTy_54_args;
  FuncTy_54_args.push_back(PointerTy_1);
  FuncTy_54_args.push_back(PointerTy_8);
  FuncTy_54_args.push_back(PointerTy_13);
  FuncTy_54_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_54 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_54_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_53 = llvm::PointerType::get(FuncTy_54, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_53);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_49);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_51);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_53);
  std::vector<llvm::Type *> FuncTy_56_args;
  FuncTy_56_args.push_back(PointerTy_1);
  FuncTy_56_args.push_back(PointerTy_8);
  FuncTy_56_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_56 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 16),
      /*Params=*/FuncTy_56_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_55 = llvm::PointerType::get(FuncTy_56, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_55);
  std::vector<llvm::Type *> FuncTy_58_args;
  FuncTy_58_args.push_back(PointerTy_1);
  FuncTy_58_args.push_back(PointerTy_8);
  FuncTy_58_args.push_back(PointerTy_13);
  FuncTy_58_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_58 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 16),
      /*Params=*/FuncTy_58_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_57 = llvm::PointerType::get(FuncTy_58, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_57);
  std::vector<llvm::Type *> FuncTy_60_args;
  FuncTy_60_args.push_back(PointerTy_1);
  FuncTy_60_args.push_back(PointerTy_8);
  FuncTy_60_args.push_back(PointerTy_13);
  FuncTy_60_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_60 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 16),
      /*Params=*/FuncTy_60_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_59 = llvm::PointerType::get(FuncTy_60, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_59);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_55);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_57);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_59);
  std::vector<llvm::Type *> FuncTy_62_args;
  FuncTy_62_args.push_back(PointerTy_1);
  FuncTy_62_args.push_back(PointerTy_8);
  FuncTy_62_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_62 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_62_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_61 = llvm::PointerType::get(FuncTy_62, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_61);
  std::vector<llvm::Type *> FuncTy_64_args;
  FuncTy_64_args.push_back(PointerTy_1);
  FuncTy_64_args.push_back(PointerTy_8);
  FuncTy_64_args.push_back(PointerTy_13);
  FuncTy_64_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_64 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_64_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_63 = llvm::PointerType::get(FuncTy_64, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_63);
  std::vector<llvm::Type *> FuncTy_66_args;
  FuncTy_66_args.push_back(PointerTy_1);
  FuncTy_66_args.push_back(PointerTy_8);
  FuncTy_66_args.push_back(PointerTy_13);
  FuncTy_66_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_66 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_66_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_65 = llvm::PointerType::get(FuncTy_66, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_65);
  std::vector<llvm::Type *> FuncTy_68_args;
  FuncTy_68_args.push_back(PointerTy_1);
  FuncTy_68_args.push_back(PointerTy_8);
  FuncTy_68_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_68 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 64),
      /*Params=*/FuncTy_68_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_67 = llvm::PointerType::get(FuncTy_68, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_67);
  std::vector<llvm::Type *> FuncTy_70_args;
  FuncTy_70_args.push_back(PointerTy_1);
  FuncTy_70_args.push_back(PointerTy_8);
  FuncTy_70_args.push_back(PointerTy_13);
  FuncTy_70_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_70 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 64),
      /*Params=*/FuncTy_70_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_69 = llvm::PointerType::get(FuncTy_70, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_69);
  std::vector<llvm::Type *> FuncTy_72_args;
  FuncTy_72_args.push_back(PointerTy_1);
  FuncTy_72_args.push_back(PointerTy_8);
  FuncTy_72_args.push_back(PointerTy_13);
  FuncTy_72_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_72 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 64),
      /*Params=*/FuncTy_72_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_71 = llvm::PointerType::get(FuncTy_72, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_71);
  std::vector<llvm::Type *> FuncTy_74_args;
  FuncTy_74_args.push_back(PointerTy_1);
  FuncTy_74_args.push_back(PointerTy_8);
  FuncTy_74_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_74 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getFloatTy(mod->getContext()),
      /*Params=*/FuncTy_74_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_73 = llvm::PointerType::get(FuncTy_74, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_73);
  std::vector<llvm::Type *> FuncTy_76_args;
  FuncTy_76_args.push_back(PointerTy_1);
  FuncTy_76_args.push_back(PointerTy_8);
  FuncTy_76_args.push_back(PointerTy_13);
  FuncTy_76_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_76 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getFloatTy(mod->getContext()),
      /*Params=*/FuncTy_76_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_75 = llvm::PointerType::get(FuncTy_76, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_75);
  std::vector<llvm::Type *> FuncTy_78_args;
  FuncTy_78_args.push_back(PointerTy_1);
  FuncTy_78_args.push_back(PointerTy_8);
  FuncTy_78_args.push_back(PointerTy_13);
  FuncTy_78_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_78 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getFloatTy(mod->getContext()),
      /*Params=*/FuncTy_78_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_77 = llvm::PointerType::get(FuncTy_78, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_77);
  std::vector<llvm::Type *> FuncTy_80_args;
  FuncTy_80_args.push_back(PointerTy_1);
  FuncTy_80_args.push_back(PointerTy_8);
  FuncTy_80_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_80 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getDoubleTy(mod->getContext()),
      /*Params=*/FuncTy_80_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_79 = llvm::PointerType::get(FuncTy_80, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_79);
  std::vector<llvm::Type *> FuncTy_82_args;
  FuncTy_82_args.push_back(PointerTy_1);
  FuncTy_82_args.push_back(PointerTy_8);
  FuncTy_82_args.push_back(PointerTy_13);
  FuncTy_82_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_82 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getDoubleTy(mod->getContext()),
      /*Params=*/FuncTy_82_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_81 = llvm::PointerType::get(FuncTy_82, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_81);
  std::vector<llvm::Type *> FuncTy_84_args;
  FuncTy_84_args.push_back(PointerTy_1);
  FuncTy_84_args.push_back(PointerTy_8);
  FuncTy_84_args.push_back(PointerTy_13);
  FuncTy_84_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_84 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getDoubleTy(mod->getContext()),
      /*Params=*/FuncTy_84_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_83 = llvm::PointerType::get(FuncTy_84, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_83);
  std::vector<llvm::Type *> FuncTy_86_args;
  FuncTy_86_args.push_back(PointerTy_1);
  FuncTy_86_args.push_back(PointerTy_8);
  FuncTy_86_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_86 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_86_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_85 = llvm::PointerType::get(FuncTy_86, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_85);
  std::vector<llvm::Type *> FuncTy_88_args;
  FuncTy_88_args.push_back(PointerTy_1);
  FuncTy_88_args.push_back(PointerTy_8);
  FuncTy_88_args.push_back(PointerTy_13);
  FuncTy_88_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_88 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_88_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_87 = llvm::PointerType::get(FuncTy_88, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_87);
  std::vector<llvm::Type *> FuncTy_90_args;
  FuncTy_90_args.push_back(PointerTy_1);
  FuncTy_90_args.push_back(PointerTy_8);
  FuncTy_90_args.push_back(PointerTy_13);
  FuncTy_90_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_90 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_90_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_89 = llvm::PointerType::get(FuncTy_90, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_89);
  std::vector<llvm::Type *> FuncTy_92_args;
  FuncTy_92_args.push_back(PointerTy_1);
  FuncTy_92_args.push_back(PointerTy_8);
  FuncTy_92_args.push_back(PointerTy_8);
  FuncTy_92_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_92 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_92_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_91 = llvm::PointerType::get(FuncTy_92, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_91);
  std::vector<llvm::Type *> FuncTy_94_args;
  FuncTy_94_args.push_back(PointerTy_1);
  FuncTy_94_args.push_back(PointerTy_8);
  FuncTy_94_args.push_back(PointerTy_8);
  FuncTy_94_args.push_back(PointerTy_13);
  FuncTy_94_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_94 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_94_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_93 = llvm::PointerType::get(FuncTy_94, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_93);
  std::vector<llvm::Type *> FuncTy_96_args;
  FuncTy_96_args.push_back(PointerTy_1);
  FuncTy_96_args.push_back(PointerTy_8);
  FuncTy_96_args.push_back(PointerTy_8);
  FuncTy_96_args.push_back(PointerTy_13);
  FuncTy_96_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_96 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_96_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_95 = llvm::PointerType::get(FuncTy_96, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_95);
  std::vector<llvm::Type *> FuncTy_98_args;
  FuncTy_98_args.push_back(PointerTy_1);
  FuncTy_98_args.push_back(PointerTy_8);
  FuncTy_98_args.push_back(PointerTy_8);
  FuncTy_98_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_98 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_98_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_97 = llvm::PointerType::get(FuncTy_98, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_97);
  std::vector<llvm::Type *> FuncTy_100_args;
  FuncTy_100_args.push_back(PointerTy_1);
  FuncTy_100_args.push_back(PointerTy_8);
  FuncTy_100_args.push_back(PointerTy_8);
  FuncTy_100_args.push_back(PointerTy_13);
  FuncTy_100_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_100 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_100_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_99 = llvm::PointerType::get(FuncTy_100, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_99);
  std::vector<llvm::Type *> FuncTy_102_args;
  FuncTy_102_args.push_back(PointerTy_1);
  FuncTy_102_args.push_back(PointerTy_8);
  FuncTy_102_args.push_back(PointerTy_8);
  FuncTy_102_args.push_back(PointerTy_13);
  FuncTy_102_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_102 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_102_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_101 = llvm::PointerType::get(FuncTy_102, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_101);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_97);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_99);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_101);
  std::vector<llvm::Type *> FuncTy_104_args;
  FuncTy_104_args.push_back(PointerTy_1);
  FuncTy_104_args.push_back(PointerTy_8);
  FuncTy_104_args.push_back(PointerTy_8);
  FuncTy_104_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_104 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 16),
      /*Params=*/FuncTy_104_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_103 = llvm::PointerType::get(FuncTy_104, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_103);
  std::vector<llvm::Type *> FuncTy_106_args;
  FuncTy_106_args.push_back(PointerTy_1);
  FuncTy_106_args.push_back(PointerTy_8);
  FuncTy_106_args.push_back(PointerTy_8);
  FuncTy_106_args.push_back(PointerTy_13);
  FuncTy_106_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_106 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 16),
      /*Params=*/FuncTy_106_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_105 = llvm::PointerType::get(FuncTy_106, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_105);
  std::vector<llvm::Type *> FuncTy_108_args;
  FuncTy_108_args.push_back(PointerTy_1);
  FuncTy_108_args.push_back(PointerTy_8);
  FuncTy_108_args.push_back(PointerTy_8);
  FuncTy_108_args.push_back(PointerTy_13);
  FuncTy_108_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_108 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 16),
      /*Params=*/FuncTy_108_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_107 = llvm::PointerType::get(FuncTy_108, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_107);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_103);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_105);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_107);
  std::vector<llvm::Type *> FuncTy_110_args;
  FuncTy_110_args.push_back(PointerTy_1);
  FuncTy_110_args.push_back(PointerTy_8);
  FuncTy_110_args.push_back(PointerTy_8);
  FuncTy_110_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_110 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_110_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_109 = llvm::PointerType::get(FuncTy_110, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_109);
  std::vector<llvm::Type *> FuncTy_112_args;
  FuncTy_112_args.push_back(PointerTy_1);
  FuncTy_112_args.push_back(PointerTy_8);
  FuncTy_112_args.push_back(PointerTy_8);
  FuncTy_112_args.push_back(PointerTy_13);
  FuncTy_112_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_112 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_112_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_111 = llvm::PointerType::get(FuncTy_112, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_111);
  std::vector<llvm::Type *> FuncTy_114_args;
  FuncTy_114_args.push_back(PointerTy_1);
  FuncTy_114_args.push_back(PointerTy_8);
  FuncTy_114_args.push_back(PointerTy_8);
  FuncTy_114_args.push_back(PointerTy_13);
  FuncTy_114_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_114 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_114_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_113 = llvm::PointerType::get(FuncTy_114, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_113);
  std::vector<llvm::Type *> FuncTy_116_args;
  FuncTy_116_args.push_back(PointerTy_1);
  FuncTy_116_args.push_back(PointerTy_8);
  FuncTy_116_args.push_back(PointerTy_8);
  FuncTy_116_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_116 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 64),
      /*Params=*/FuncTy_116_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_115 = llvm::PointerType::get(FuncTy_116, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_115);
  std::vector<llvm::Type *> FuncTy_118_args;
  FuncTy_118_args.push_back(PointerTy_1);
  FuncTy_118_args.push_back(PointerTy_8);
  FuncTy_118_args.push_back(PointerTy_8);
  FuncTy_118_args.push_back(PointerTy_13);
  FuncTy_118_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_118 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 64),
      /*Params=*/FuncTy_118_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_117 = llvm::PointerType::get(FuncTy_118, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_117);
  std::vector<llvm::Type *> FuncTy_120_args;
  FuncTy_120_args.push_back(PointerTy_1);
  FuncTy_120_args.push_back(PointerTy_8);
  FuncTy_120_args.push_back(PointerTy_8);
  FuncTy_120_args.push_back(PointerTy_13);
  FuncTy_120_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_120 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 64),
      /*Params=*/FuncTy_120_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_119 = llvm::PointerType::get(FuncTy_120, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_119);
  std::vector<llvm::Type *> FuncTy_122_args;
  FuncTy_122_args.push_back(PointerTy_1);
  FuncTy_122_args.push_back(PointerTy_8);
  FuncTy_122_args.push_back(PointerTy_8);
  FuncTy_122_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_122 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getFloatTy(mod->getContext()),
      /*Params=*/FuncTy_122_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_121 = llvm::PointerType::get(FuncTy_122, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_121);
  std::vector<llvm::Type *> FuncTy_124_args;
  FuncTy_124_args.push_back(PointerTy_1);
  FuncTy_124_args.push_back(PointerTy_8);
  FuncTy_124_args.push_back(PointerTy_8);
  FuncTy_124_args.push_back(PointerTy_13);
  FuncTy_124_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_124 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getFloatTy(mod->getContext()),
      /*Params=*/FuncTy_124_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_123 = llvm::PointerType::get(FuncTy_124, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_123);
  std::vector<llvm::Type *> FuncTy_126_args;
  FuncTy_126_args.push_back(PointerTy_1);
  FuncTy_126_args.push_back(PointerTy_8);
  FuncTy_126_args.push_back(PointerTy_8);
  FuncTy_126_args.push_back(PointerTy_13);
  FuncTy_126_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_126 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getFloatTy(mod->getContext()),
      /*Params=*/FuncTy_126_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_125 = llvm::PointerType::get(FuncTy_126, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_125);
  std::vector<llvm::Type *> FuncTy_128_args;
  FuncTy_128_args.push_back(PointerTy_1);
  FuncTy_128_args.push_back(PointerTy_8);
  FuncTy_128_args.push_back(PointerTy_8);
  FuncTy_128_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_128 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getDoubleTy(mod->getContext()),
      /*Params=*/FuncTy_128_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_127 = llvm::PointerType::get(FuncTy_128, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_127);
  std::vector<llvm::Type *> FuncTy_130_args;
  FuncTy_130_args.push_back(PointerTy_1);
  FuncTy_130_args.push_back(PointerTy_8);
  FuncTy_130_args.push_back(PointerTy_8);
  FuncTy_130_args.push_back(PointerTy_13);
  FuncTy_130_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_130 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getDoubleTy(mod->getContext()),
      /*Params=*/FuncTy_130_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_129 = llvm::PointerType::get(FuncTy_130, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_129);
  std::vector<llvm::Type *> FuncTy_132_args;
  FuncTy_132_args.push_back(PointerTy_1);
  FuncTy_132_args.push_back(PointerTy_8);
  FuncTy_132_args.push_back(PointerTy_8);
  FuncTy_132_args.push_back(PointerTy_13);
  FuncTy_132_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_132 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getDoubleTy(mod->getContext()),
      /*Params=*/FuncTy_132_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_131 = llvm::PointerType::get(FuncTy_132, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_131);
  std::vector<llvm::Type *> FuncTy_134_args;
  FuncTy_134_args.push_back(PointerTy_1);
  FuncTy_134_args.push_back(PointerTy_8);
  FuncTy_134_args.push_back(PointerTy_8);
  FuncTy_134_args.push_back(PointerTy_13);
  llvm::FunctionType *FuncTy_134 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_134_args,
      /*isVarArg=*/true);

  llvm::PointerType *PointerTy_133 = llvm::PointerType::get(FuncTy_134, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_133);
  std::vector<llvm::Type *> FuncTy_136_args;
  FuncTy_136_args.push_back(PointerTy_1);
  FuncTy_136_args.push_back(PointerTy_8);
  FuncTy_136_args.push_back(PointerTy_8);
  FuncTy_136_args.push_back(PointerTy_13);
  FuncTy_136_args.push_back(PointerTy_43);
  llvm::FunctionType *FuncTy_136 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_136_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_135 = llvm::PointerType::get(FuncTy_136, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_135);
  std::vector<llvm::Type *> FuncTy_138_args;
  FuncTy_138_args.push_back(PointerTy_1);
  FuncTy_138_args.push_back(PointerTy_8);
  FuncTy_138_args.push_back(PointerTy_8);
  FuncTy_138_args.push_back(PointerTy_13);
  FuncTy_138_args.push_back(PointerTy_46);
  llvm::FunctionType *FuncTy_138 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_138_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_137 = llvm::PointerType::get(FuncTy_138, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_137);
  std::vector<llvm::Type *> FuncTy_140_args;
  FuncTy_140_args.push_back(PointerTy_1);
  FuncTy_140_args.push_back(PointerTy_8);
  FuncTy_140_args.push_back(PointerTy_3);
  FuncTy_140_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_140 = llvm::FunctionType::get(
      /*Result=*/PointerTy_16,
      /*Params=*/FuncTy_140_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_139 = llvm::PointerType::get(FuncTy_140, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_139);
  std::vector<llvm::Type *> FuncTy_142_args;
  FuncTy_142_args.push_back(PointerTy_1);
  FuncTy_142_args.push_back(PointerTy_8);
  FuncTy_142_args.push_back(PointerTy_16);
  llvm::FunctionType *FuncTy_142 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_142_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_141 = llvm::PointerType::get(FuncTy_142, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_141);
  std::vector<llvm::Type *> FuncTy_144_args;
  FuncTy_144_args.push_back(PointerTy_1);
  FuncTy_144_args.push_back(PointerTy_8);
  FuncTy_144_args.push_back(PointerTy_16);
  llvm::FunctionType *FuncTy_144 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_144_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_143 = llvm::PointerType::get(FuncTy_144, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_143);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_143);
  std::vector<llvm::Type *> FuncTy_146_args;
  FuncTy_146_args.push_back(PointerTy_1);
  FuncTy_146_args.push_back(PointerTy_8);
  FuncTy_146_args.push_back(PointerTy_16);
  llvm::FunctionType *FuncTy_146 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 16),
      /*Params=*/FuncTy_146_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_145 = llvm::PointerType::get(FuncTy_146, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_145);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_145);
  std::vector<llvm::Type *> FuncTy_148_args;
  FuncTy_148_args.push_back(PointerTy_1);
  FuncTy_148_args.push_back(PointerTy_8);
  FuncTy_148_args.push_back(PointerTy_16);
  llvm::FunctionType *FuncTy_148 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_148_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_147 = llvm::PointerType::get(FuncTy_148, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_147);
  std::vector<llvm::Type *> FuncTy_150_args;
  FuncTy_150_args.push_back(PointerTy_1);
  FuncTy_150_args.push_back(PointerTy_8);
  FuncTy_150_args.push_back(PointerTy_16);
  llvm::FunctionType *FuncTy_150 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 64),
      /*Params=*/FuncTy_150_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_149 = llvm::PointerType::get(FuncTy_150, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_149);
  std::vector<llvm::Type *> FuncTy_152_args;
  FuncTy_152_args.push_back(PointerTy_1);
  FuncTy_152_args.push_back(PointerTy_8);
  FuncTy_152_args.push_back(PointerTy_16);
  llvm::FunctionType *FuncTy_152 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getFloatTy(mod->getContext()),
      /*Params=*/FuncTy_152_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_151 = llvm::PointerType::get(FuncTy_152, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_151);
  std::vector<llvm::Type *> FuncTy_154_args;
  FuncTy_154_args.push_back(PointerTy_1);
  FuncTy_154_args.push_back(PointerTy_8);
  FuncTy_154_args.push_back(PointerTy_16);
  llvm::FunctionType *FuncTy_154 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getDoubleTy(mod->getContext()),
      /*Params=*/FuncTy_154_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_153 = llvm::PointerType::get(FuncTy_154, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_153);
  std::vector<llvm::Type *> FuncTy_156_args;
  FuncTy_156_args.push_back(PointerTy_1);
  FuncTy_156_args.push_back(PointerTy_8);
  FuncTy_156_args.push_back(PointerTy_16);
  FuncTy_156_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_156 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_156_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_155 = llvm::PointerType::get(FuncTy_156, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_155);
  std::vector<llvm::Type *> FuncTy_158_args;
  FuncTy_158_args.push_back(PointerTy_1);
  FuncTy_158_args.push_back(PointerTy_8);
  FuncTy_158_args.push_back(PointerTy_16);
  FuncTy_158_args.push_back(llvm::IntegerType::get(mod->getContext(), 8));
  llvm::FunctionType *FuncTy_158 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_158_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_157 = llvm::PointerType::get(FuncTy_158, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_157);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_157);
  std::vector<llvm::Type *> FuncTy_160_args;
  FuncTy_160_args.push_back(PointerTy_1);
  FuncTy_160_args.push_back(PointerTy_8);
  FuncTy_160_args.push_back(PointerTy_16);
  FuncTy_160_args.push_back(llvm::IntegerType::get(mod->getContext(), 16));
  llvm::FunctionType *FuncTy_160 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_160_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_159 = llvm::PointerType::get(FuncTy_160, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_159);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_159);
  std::vector<llvm::Type *> FuncTy_162_args;
  FuncTy_162_args.push_back(PointerTy_1);
  FuncTy_162_args.push_back(PointerTy_8);
  FuncTy_162_args.push_back(PointerTy_16);
  FuncTy_162_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_162 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_162_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_161 = llvm::PointerType::get(FuncTy_162, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_161);
  std::vector<llvm::Type *> FuncTy_164_args;
  FuncTy_164_args.push_back(PointerTy_1);
  FuncTy_164_args.push_back(PointerTy_8);
  FuncTy_164_args.push_back(PointerTy_16);
  FuncTy_164_args.push_back(llvm::IntegerType::get(mod->getContext(), 64));
  llvm::FunctionType *FuncTy_164 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_164_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_163 = llvm::PointerType::get(FuncTy_164, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_163);
  std::vector<llvm::Type *> FuncTy_166_args;
  FuncTy_166_args.push_back(PointerTy_1);
  FuncTy_166_args.push_back(PointerTy_8);
  FuncTy_166_args.push_back(PointerTy_16);
  FuncTy_166_args.push_back(llvm::Type::getFloatTy(mod->getContext()));
  llvm::FunctionType *FuncTy_166 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_166_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_165 = llvm::PointerType::get(FuncTy_166, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_165);
  std::vector<llvm::Type *> FuncTy_168_args;
  FuncTy_168_args.push_back(PointerTy_1);
  FuncTy_168_args.push_back(PointerTy_8);
  FuncTy_168_args.push_back(PointerTy_16);
  FuncTy_168_args.push_back(llvm::Type::getDoubleTy(mod->getContext()));
  llvm::FunctionType *FuncTy_168 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_168_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_167 = llvm::PointerType::get(FuncTy_168, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_167);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_47);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_39);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_41);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_44);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_49);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_51);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_53);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_49);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_51);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_53);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_55);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_57);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_59);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_55);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_57);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_59);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_61);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_63);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_65);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_67);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_69);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_71);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_73);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_75);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_77);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_79);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_81);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_83);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_85);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_87);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_89);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_139);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_141);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_143);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_143);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_145);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_145);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_147);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_149);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_151);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_153);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_155);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_157);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_157);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_159);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_159);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_161);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_163);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_165);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_167);
  std::vector<llvm::Type *> FuncTy_170_args;
  FuncTy_170_args.push_back(PointerTy_1);
  llvm::PointerType *PointerTy_171 =
      llvm::PointerType::get(llvm::IntegerType::get(mod->getContext(), 16), 0);

  FuncTy_170_args.push_back(PointerTy_171);
  FuncTy_170_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_170 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_170_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_169 = llvm::PointerType::get(FuncTy_170, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_169);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_25);
  std::vector<llvm::Type *> FuncTy_173_args;
  FuncTy_173_args.push_back(PointerTy_1);
  FuncTy_173_args.push_back(PointerTy_8);
  FuncTy_173_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_173 = llvm::FunctionType::get(
      /*Result=*/PointerTy_171,
      /*Params=*/FuncTy_173_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_172 = llvm::PointerType::get(FuncTy_173, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_172);
  std::vector<llvm::Type *> FuncTy_175_args;
  FuncTy_175_args.push_back(PointerTy_1);
  FuncTy_175_args.push_back(PointerTy_8);
  FuncTy_175_args.push_back(PointerTy_171);
  llvm::FunctionType *FuncTy_175 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_175_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_174 = llvm::PointerType::get(FuncTy_175, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_174);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_9);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_25);
  std::vector<llvm::Type *> FuncTy_177_args;
  FuncTy_177_args.push_back(PointerTy_1);
  FuncTy_177_args.push_back(PointerTy_8);
  FuncTy_177_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_177 = llvm::FunctionType::get(
      /*Result=*/PointerTy_3,
      /*Params=*/FuncTy_177_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_176 = llvm::PointerType::get(FuncTy_177, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_176);
  std::vector<llvm::Type *> FuncTy_179_args;
  FuncTy_179_args.push_back(PointerTy_1);
  FuncTy_179_args.push_back(PointerTy_8);
  FuncTy_179_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_179 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_179_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_178 = llvm::PointerType::get(FuncTy_179, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_178);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_25);
  std::vector<llvm::Type *> FuncTy_181_args;
  FuncTy_181_args.push_back(PointerTy_1);
  FuncTy_181_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_181_args.push_back(PointerTy_8);
  FuncTy_181_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_181 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_181_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_180 = llvm::PointerType::get(FuncTy_181, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_180);
  std::vector<llvm::Type *> FuncTy_183_args;
  FuncTy_183_args.push_back(PointerTy_1);
  FuncTy_183_args.push_back(PointerTy_8);
  FuncTy_183_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_183 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_183_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_182 = llvm::PointerType::get(FuncTy_183, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_182);
  std::vector<llvm::Type *> FuncTy_185_args;
  FuncTy_185_args.push_back(PointerTy_1);
  FuncTy_185_args.push_back(PointerTy_8);
  FuncTy_185_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_185_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_185 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_185_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_184 = llvm::PointerType::get(FuncTy_185, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_184);
  std::vector<llvm::Type *> FuncTy_187_args;
  FuncTy_187_args.push_back(PointerTy_1);
  FuncTy_187_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_187 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_187_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_186 = llvm::PointerType::get(FuncTy_187, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_186);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_186);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_186);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_186);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_186);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_186);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_186);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_186);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_176);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_176);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_172);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_172);
  std::vector<llvm::Type *> FuncTy_189_args;
  FuncTy_189_args.push_back(PointerTy_1);
  FuncTy_189_args.push_back(PointerTy_8);
  FuncTy_189_args.push_back(PointerTy_3);
  llvm::PointerType *PointerTy_190 =
      llvm::PointerType::get(llvm::IntegerType::get(mod->getContext(), 32), 0);

  llvm::FunctionType *FuncTy_189 = llvm::FunctionType::get(
      /*Result=*/PointerTy_190,
      /*Params=*/FuncTy_189_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_188 = llvm::PointerType::get(FuncTy_189, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_188);
  std::vector<llvm::Type *> FuncTy_192_args;
  FuncTy_192_args.push_back(PointerTy_1);
  FuncTy_192_args.push_back(PointerTy_8);
  FuncTy_192_args.push_back(PointerTy_3);
  llvm::PointerType *PointerTy_193 =
      llvm::PointerType::get(llvm::IntegerType::get(mod->getContext(), 64), 0);

  llvm::FunctionType *FuncTy_192 = llvm::FunctionType::get(
      /*Result=*/PointerTy_193,
      /*Params=*/FuncTy_192_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_191 = llvm::PointerType::get(FuncTy_192, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_191);
  std::vector<llvm::Type *> FuncTy_195_args;
  FuncTy_195_args.push_back(PointerTy_1);
  FuncTy_195_args.push_back(PointerTy_8);
  FuncTy_195_args.push_back(PointerTy_3);
  llvm::PointerType *PointerTy_196 =
      llvm::PointerType::get(llvm::Type::getFloatTy(mod->getContext()), 0);

  llvm::FunctionType *FuncTy_195 = llvm::FunctionType::get(
      /*Result=*/PointerTy_196,
      /*Params=*/FuncTy_195_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_194 = llvm::PointerType::get(FuncTy_195, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_194);
  std::vector<llvm::Type *> FuncTy_198_args;
  FuncTy_198_args.push_back(PointerTy_1);
  FuncTy_198_args.push_back(PointerTy_8);
  FuncTy_198_args.push_back(PointerTy_3);
  llvm::PointerType *PointerTy_199 =
      llvm::PointerType::get(llvm::Type::getDoubleTy(mod->getContext()), 0);

  llvm::FunctionType *FuncTy_198 = llvm::FunctionType::get(
      /*Result=*/PointerTy_199,
      /*Params=*/FuncTy_198_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_197 = llvm::PointerType::get(FuncTy_198, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_197);
  std::vector<llvm::Type *> FuncTy_201_args;
  FuncTy_201_args.push_back(PointerTy_1);
  FuncTy_201_args.push_back(PointerTy_8);
  FuncTy_201_args.push_back(PointerTy_3);
  FuncTy_201_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_201 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_201_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_200 = llvm::PointerType::get(FuncTy_201, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_200);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_200);
  std::vector<llvm::Type *> FuncTy_203_args;
  FuncTy_203_args.push_back(PointerTy_1);
  FuncTy_203_args.push_back(PointerTy_8);
  FuncTy_203_args.push_back(PointerTy_171);
  FuncTy_203_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_203 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_203_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_202 = llvm::PointerType::get(FuncTy_203, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_202);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_202);
  std::vector<llvm::Type *> FuncTy_205_args;
  FuncTy_205_args.push_back(PointerTy_1);
  FuncTy_205_args.push_back(PointerTy_8);
  FuncTy_205_args.push_back(PointerTy_190);
  FuncTy_205_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_205 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_205_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_204 = llvm::PointerType::get(FuncTy_205, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_204);
  std::vector<llvm::Type *> FuncTy_207_args;
  FuncTy_207_args.push_back(PointerTy_1);
  FuncTy_207_args.push_back(PointerTy_8);
  FuncTy_207_args.push_back(PointerTy_193);
  FuncTy_207_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_207 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_207_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_206 = llvm::PointerType::get(FuncTy_207, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_206);
  std::vector<llvm::Type *> FuncTy_209_args;
  FuncTy_209_args.push_back(PointerTy_1);
  FuncTy_209_args.push_back(PointerTy_8);
  FuncTy_209_args.push_back(PointerTy_196);
  FuncTy_209_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_209 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_209_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_208 = llvm::PointerType::get(FuncTy_209, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_208);
  std::vector<llvm::Type *> FuncTy_211_args;
  FuncTy_211_args.push_back(PointerTy_1);
  FuncTy_211_args.push_back(PointerTy_8);
  FuncTy_211_args.push_back(PointerTy_199);
  FuncTy_211_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_211 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_211_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_210 = llvm::PointerType::get(FuncTy_211, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_210);
  std::vector<llvm::Type *> FuncTy_213_args;
  FuncTy_213_args.push_back(PointerTy_1);
  FuncTy_213_args.push_back(PointerTy_8);
  FuncTy_213_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_213_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_213_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_213 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_213_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_212 = llvm::PointerType::get(FuncTy_213, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_212);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_212);
  std::vector<llvm::Type *> FuncTy_215_args;
  FuncTy_215_args.push_back(PointerTy_1);
  FuncTy_215_args.push_back(PointerTy_8);
  FuncTy_215_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_215_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_215_args.push_back(PointerTy_171);
  llvm::FunctionType *FuncTy_215 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_215_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_214 = llvm::PointerType::get(FuncTy_215, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_214);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_214);
  std::vector<llvm::Type *> FuncTy_217_args;
  FuncTy_217_args.push_back(PointerTy_1);
  FuncTy_217_args.push_back(PointerTy_8);
  FuncTy_217_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_217_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_217_args.push_back(PointerTy_190);
  llvm::FunctionType *FuncTy_217 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_217_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_216 = llvm::PointerType::get(FuncTy_217, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_216);
  std::vector<llvm::Type *> FuncTy_219_args;
  FuncTy_219_args.push_back(PointerTy_1);
  FuncTy_219_args.push_back(PointerTy_8);
  FuncTy_219_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_219_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_219_args.push_back(PointerTy_193);
  llvm::FunctionType *FuncTy_219 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_219_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_218 = llvm::PointerType::get(FuncTy_219, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_218);
  std::vector<llvm::Type *> FuncTy_221_args;
  FuncTy_221_args.push_back(PointerTy_1);
  FuncTy_221_args.push_back(PointerTy_8);
  FuncTy_221_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_221_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_221_args.push_back(PointerTy_196);
  llvm::FunctionType *FuncTy_221 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_221_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_220 = llvm::PointerType::get(FuncTy_221, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_220);
  std::vector<llvm::Type *> FuncTy_223_args;
  FuncTy_223_args.push_back(PointerTy_1);
  FuncTy_223_args.push_back(PointerTy_8);
  FuncTy_223_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_223_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  FuncTy_223_args.push_back(PointerTy_199);
  llvm::FunctionType *FuncTy_223 = llvm::FunctionType::get(
      /*Result=*/llvm::Type::getVoidTy(mod->getContext()),
      /*Params=*/FuncTy_223_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_222 = llvm::PointerType::get(FuncTy_223, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_222);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_212);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_212);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_214);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_214);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_216);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_218);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_220);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_222);
  std::vector<llvm::Type *> FuncTy_225_args;
  FuncTy_225_args.push_back(PointerTy_1);
  FuncTy_225_args.push_back(PointerTy_8);
  llvm::StructType *StructTy_struct_JNINativeMethod =
      mod->getTypeByName("struct.JNINativeMethod");
  if (!StructTy_struct_JNINativeMethod) {
    StructTy_struct_JNINativeMethod =
        llvm::StructType::create(mod->getContext(), "struct.JNINativeMethod");
  }
  std::vector<llvm::Type *> StructTy_struct_JNINativeMethod_fields;
  StructTy_struct_JNINativeMethod_fields.push_back(PointerTy_3);
  StructTy_struct_JNINativeMethod_fields.push_back(PointerTy_3);
  StructTy_struct_JNINativeMethod_fields.push_back(PointerTy_3);
  if (StructTy_struct_JNINativeMethod->isOpaque()) {
    StructTy_struct_JNINativeMethod->setBody(
        StructTy_struct_JNINativeMethod_fields, /*isPacked=*/false);
  }

  llvm::PointerType *PointerTy_226 =
      llvm::PointerType::get(StructTy_struct_JNINativeMethod, 0);

  FuncTy_225_args.push_back(PointerTy_226);
  FuncTy_225_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_225 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_225_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_224 = llvm::PointerType::get(FuncTy_225, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_224);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_25);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_25);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_25);
  std::vector<llvm::Type *> FuncTy_228_args;
  FuncTy_228_args.push_back(PointerTy_1);
  llvm::StructType *StructTy_struct_JNIInvokeInterface_ =
      mod->getTypeByName("struct.JNIInvokeInterface_");
  if (!StructTy_struct_JNIInvokeInterface_) {
    StructTy_struct_JNIInvokeInterface_ = llvm::StructType::create(
        mod->getContext(), "struct.JNIInvokeInterface_");
  }
  std::vector<llvm::Type *> StructTy_struct_JNIInvokeInterface__fields;
  StructTy_struct_JNIInvokeInterface__fields.push_back(PointerTy_3);
  StructTy_struct_JNIInvokeInterface__fields.push_back(PointerTy_3);
  StructTy_struct_JNIInvokeInterface__fields.push_back(PointerTy_3);
  std::vector<llvm::Type *> FuncTy_233_args;
  llvm::PointerType *PointerTy_231 =
      llvm::PointerType::get(StructTy_struct_JNIInvokeInterface_, 0);

  llvm::PointerType *PointerTy_230 = llvm::PointerType::get(PointerTy_231, 0);

  FuncTy_233_args.push_back(PointerTy_230);
  llvm::FunctionType *FuncTy_233 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_233_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_232 = llvm::PointerType::get(FuncTy_233, 0);

  StructTy_struct_JNIInvokeInterface__fields.push_back(PointerTy_232);
  std::vector<llvm::Type *> FuncTy_235_args;
  FuncTy_235_args.push_back(PointerTy_230);
  llvm::PointerType *PointerTy_236 = llvm::PointerType::get(PointerTy_3, 0);

  FuncTy_235_args.push_back(PointerTy_236);
  FuncTy_235_args.push_back(PointerTy_3);
  llvm::FunctionType *FuncTy_235 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_235_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_234 = llvm::PointerType::get(FuncTy_235, 0);

  StructTy_struct_JNIInvokeInterface__fields.push_back(PointerTy_234);
  StructTy_struct_JNIInvokeInterface__fields.push_back(PointerTy_232);
  std::vector<llvm::Type *> FuncTy_238_args;
  FuncTy_238_args.push_back(PointerTy_230);
  FuncTy_238_args.push_back(PointerTy_236);
  FuncTy_238_args.push_back(llvm::IntegerType::get(mod->getContext(), 32));
  llvm::FunctionType *FuncTy_238 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_238_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_237 = llvm::PointerType::get(FuncTy_238, 0);

  StructTy_struct_JNIInvokeInterface__fields.push_back(PointerTy_237);
  StructTy_struct_JNIInvokeInterface__fields.push_back(PointerTy_234);
  if (StructTy_struct_JNIInvokeInterface_->isOpaque()) {
    StructTy_struct_JNIInvokeInterface_->setBody(
        StructTy_struct_JNIInvokeInterface__fields, /*isPacked=*/false);
  }

  llvm::PointerType *PointerTy_229 = llvm::PointerType::get(PointerTy_230, 0);

  FuncTy_228_args.push_back(PointerTy_229);
  llvm::FunctionType *FuncTy_228 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_228_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_227 = llvm::PointerType::get(FuncTy_228, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_227);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_214);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_212);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_176);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_200);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_172);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_174);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_19);
  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_37);
  std::vector<llvm::Type *> FuncTy_240_args;
  FuncTy_240_args.push_back(PointerTy_1);
  llvm::FunctionType *FuncTy_240 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 8),
      /*Params=*/FuncTy_240_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_239 = llvm::PointerType::get(FuncTy_240, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_239);
  std::vector<llvm::Type *> FuncTy_242_args;
  FuncTy_242_args.push_back(PointerTy_1);
  FuncTy_242_args.push_back(PointerTy_3);
  FuncTy_242_args.push_back(llvm::IntegerType::get(mod->getContext(), 64));
  llvm::FunctionType *FuncTy_242 = llvm::FunctionType::get(
      /*Result=*/PointerTy_8,
      /*Params=*/FuncTy_242_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_241 = llvm::PointerType::get(FuncTy_242, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_241);
  std::vector<llvm::Type *> FuncTy_244_args;
  FuncTy_244_args.push_back(PointerTy_1);
  FuncTy_244_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_244 = llvm::FunctionType::get(
      /*Result=*/PointerTy_3,
      /*Params=*/FuncTy_244_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_243 = llvm::PointerType::get(FuncTy_244, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_243);
  std::vector<llvm::Type *> FuncTy_246_args;
  FuncTy_246_args.push_back(PointerTy_1);
  FuncTy_246_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_246 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 64),
      /*Params=*/FuncTy_246_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_245 = llvm::PointerType::get(FuncTy_246, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_245);
  std::vector<llvm::Type *> FuncTy_0_args;
  FuncTy_0_args.push_back(PointerTy_1);
  FuncTy_0_args.push_back(PointerTy_8);
  llvm::FunctionType *FuncTy_0 = llvm::FunctionType::get(
      /*Result=*/llvm::IntegerType::get(mod->getContext(), 32),
      /*Params=*/FuncTy_0_args,
      /*isVarArg=*/false);

  llvm::PointerType *PointerTy_247 = llvm::PointerType::get(FuncTy_0, 0);

  StructTy_struct_JNINativeInterface__fields.push_back(PointerTy_247);
  if (StructTy_struct_JNINativeInterface_->isOpaque()) {
    StructTy_struct_JNINativeInterface_->setBody(
        StructTy_struct_JNINativeInterface__fields, /*isPacked=*/false);
  }
}
