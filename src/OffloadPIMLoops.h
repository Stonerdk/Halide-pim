#ifndef HALIDE_OFFLOAD_PIM_LOOPS_H
#define HALIDE_OFFLOAD_PIM_LOOPS_H

/** \file
 * Defines a lowering pass to pull loops marked with
 * GPU device APIs to a separate module, and call them through the
 * appropriate host runtime module.
 */

#include "Expr.h"

namespace Halide {

struct Target;

namespace Internal {

/** Pull loops marked with GPU device APIs to a separate
 * module, and call them through the appropriate host runtime module. */
Stmt inject_pim_offload(const Stmt &s, const Target &host_target, Stmt& execute_stmt);

}  // namespace Internal
}  // namespace Halide

#endif
