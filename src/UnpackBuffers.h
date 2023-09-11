#ifndef HALIDE_UNPACK_BUFFERS_H
#define HALIDE_UNPACK_BUFFERS_H

/** \file
 * Defines the lowering pass that unpacks buffer arguments onto the symbol table
 */

#include "Expr.h"
#include "Argument.h"
#include "Function.h"
#include <map>
#include <string>

using namespace std;
namespace Halide {
namespace Internal {

/** Creates let stmts for the various buffer components
 * (e.g. foo.extent.0) in any referenced concrete buffers or buffer
 * parameters. After this pass, the only undefined symbols should
 * scalar parameters and the buffers themselves (e.g. foo.buffer). */
Stmt unpack_buffers(Stmt s);

Stmt unpack_buffers_upmem_lt(Stmt s, map<string, Stmt> &splitted_stmts, const vector<Argument> &args, const Function& output);

}  // namespace Internal
}  // namespace Halide

#endif
