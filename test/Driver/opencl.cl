// RUN: %clang -S -### -cl-std=CL %s 2>&1 | FileCheck --check-prefix=CHECK-CL %s
// RUN: %clang -S -### -cl-std=CL1.1 %s 2>&1 | FileCheck --check-prefix=CHECK-CL11 %s
// RUN: %clang -S -### -cl-std=CL1.2 %s 2>&1 | FileCheck --check-prefix=CHECK-CL12 %s
// RUN: %clang -S -### -cl-std=CL2.0 %s 2>&1 | FileCheck --check-prefix=CHECK-CL20 %s
// RUN: %clang -S -### -cl-opt-disable %s 2>&1 | FileCheck --check-prefix=CHECK-OPT-DISABLE %s
// RUN: %clang -S -### -cl-strict-aliasing %s 2>&1 | FileCheck --check-prefix=CHECK-STRICT-ALIASING %s
// RUN: %clang -S -### -cl-single-precision-constant %s 2>&1 | FileCheck --check-prefix=CHECK-SINGLE-PRECISION-CONST %s
// RUN: %clang -S -### -cl-finite-math-only %s 2>&1 | FileCheck --check-prefix=CHECK-FINITE-MATH-ONLY %s
// RUN: %clang -S -### -cl-kernel-arg-info %s 2>&1 | FileCheck --check-prefix=CHECK-KERNEL-ARG-INFO %s
// RUN: %clang -S -### -cl-unsafe-math-optimizations %s 2>&1 | FileCheck --check-prefix=CHECK-UNSAFE-MATH-OPT %s
// RUN: %clang -S -### -cl-fast-relaxed-math %s 2>&1 | FileCheck --check-prefix=CHECK-FAST-RELAXED-MATH %s
// RUN: %clang -S -### -cl-mad-enable %s 2>&1 | FileCheck --check-prefix=CHECK-MAD-ENABLE %s
// RUN: %clang -S -### -cl-no-signed-zeros %s 2>&1 | FileCheck --check-prefix=CHECK-NO-SIGNED-ZEROS %s
// RUN: %clang -S -### -cl-denorms-are-zero %s 2>&1 | FileCheck --check-prefix=CHECK-DENORMS-ARE-ZERO %s
// RUN: not %clang -cl-std=c99 -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-C99 %s
// RUN: not %clang -cl-std=invalid -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID %s

// CHECK-CL: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-std=CL"
// CHECK-CL11: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-std=CL1.1"
// CHECK-CL12: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-std=CL1.2"
// CHECK-CL20: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-std=CL2.0"
// CHECK-OPT-DISABLE: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-opt-disable"
// CHECK-STRICT-ALIASING: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-strict-aliasing"
// CHECK-SINGLE-PRECISION-CONST: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-single-precision-constant"
// CHECK-FINITE-MATH-ONLY: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-finite-math-only"
// CHECK-KERNEL-ARG-INFO: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-kernel-arg-info"
// CHECK-UNSAFE-MATH-OPT: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-unsafe-math-optimizations"
// CHECK-FAST-RELAXED-MATH: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-fast-relaxed-math"
// CHECK-MAD-ENABLE: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-mad-enable"
// CHECK-NO-SIGNED-ZEROS: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-no-signed-zeros"
// CHECK-DENORMS-ARE-ZERO: {{.*}}clang{{.*}} "-cc1" {{.*}} "-cl-denorms-are-zero"
// CHECK-C99: error: invalid value 'c99' in '-cl-std=c99'
// CHECK-INVALID: error: invalid value 'invalid' in '-cl-std=invalid'

kernel void func(void);
