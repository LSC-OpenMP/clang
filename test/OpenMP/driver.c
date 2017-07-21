// Test that by default -fnoopenmp-use-tls is passed to frontend.
//
// RUN: %clang %s -### -o %t.o 2>&1 -fopenmp=libomp | FileCheck --check-prefix=CHECK-DEFAULT %s
// CHECK-DEFAULT: -cc1
// CHECK-DEFAULT-NOT: -fnoopenmp-use-tls
//
// RUN: %clang %s -### -o %t.o 2>&1 -fopenmp=libomp -fnoopenmp-use-tls | FileCheck --check-prefix=CHECK-NO-TLS %s
// CHECK-NO-TLS: -cc1
// CHECK-NO-TLS-SAME: -fnoopenmp-use-tls
//
// RUN: %clang %s -c -E -dM -fopenmp=libomp | FileCheck --check-prefix=CHECK-DEFAULT-VERSION45 %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=1 | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=0 | FileCheck --check-prefix=CHECK-DEFAULT-VERSION45 %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=100 | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=31 | FileCheck --check-prefix=CHECK-DEFAULT-VERSION %s
// CHECK-DEFAULT-VERSION45: #define _OPENMP 201511
// CHECK-DEFAULT-VERSION: #define _OPENMP 201107

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=40 | FileCheck --check-prefix=CHECK-40-VERSION %s
// CHECK-40-VERSION: #define _OPENMP 201307

// RUN: %clang %s -c -E -dM -fopenmp=libomp -fopenmp-version=45 | FileCheck --check-prefix=CHECK-45-VERSION %s
// CHECK-45-VERSION: #define _OPENMP 201511

// RUN: %clang %s -c -E -dM -fopenmp-version=1 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=31 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=40 | FileCheck --check-prefix=CHECK-VERSION %s
// RUN: %clang %s -c -E -dM -fopenmp-version=45 | FileCheck --check-prefix=CHECK-VERSION %s
// CHECK-VERSION-NOT: #define _OPENMP

// RUN: %clang %s -### -fopenmp=libomp -fopenmp-targets=nvptx64-unknown-unknown -g -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-NO-DEBUG %s
// RUN: %clang %s -### -fopenmp=libomp -fopenmp-targets=nvptx64-unknown-unknown -g -fopenmp-debug -fno-openmp-debug -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-NO-DEBUG %s
// RUN: %clang %s -### -fopenmp=libomp -fopenmp-targets=nvptx64-unknown-unknown -g -fopenmp-debug -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-DEBUG %s
// RUN: %clang %s -### -fopenmp=libomp -fopenmp-targets=nvptx64-unknown-unknown -g -fno-openmp-debug -fopenmp-debug -o %t.o 2>&1 | FileCheck --check-prefix=CHECK-DEBUG %s
// CHECK-NO-DEBUG: "-triple" "nvptx64-unknown-unknown"
// CHECK-NO-DEBUG-NOT: "-debug-info-kind=limited"
// CHECK-NO-DEBUG: "-triple"
// CHECK-DEBUG: "-triple" "nvptx64-unknown-unknown"
// CHECK-DEBUG-SAME: "-debug-info-kind=limited"
