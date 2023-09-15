#ifndef HALIDE_PIM_LAYOUT_TRANSFORM_H
#define HALIDE_PIM_LAYOUT_TRANSFORM_H

/** \file
 * Defines the lowering pass that flattens multi-dimensional storage
 * into single-dimensional array access
 */

#include <map>
#include <string>
#include <vector>

#include "Argument.h"
#include "Function.h"

#include "Expr.h"

namespace Halide {

struct Target;

namespace Internal {

class Function;

/** Take a statement with multi-dimensional Realize, Provide, and Call
 * nodes, and turn it into a statement with single-dimensional
 * Allocate, Store, and Load nodes respectively. */
Stmt pim_layout_transform(Stmt s);

Stmt pim_layout_transform_split(Stmt s, std::map<std::string, Stmt>& splitted_stmts);

}  // namespace Internal
}  // namespace Halide

#endif
