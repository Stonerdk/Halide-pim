#ifndef HALIDE_CODEGEN_PIM_DEV_H
#define HALIDE_CODEGEN_PIM_DEV_H

/** \file
 * Defines the code-generator interface for producing GPU device code
 */
#include <string>
#include <vector>

#include "CodeGen_C.h"
#include "DeviceArgument.h"
#include "Expr.h"


namespace Halide {
namespace Internal {

struct CodeGen_PIM_Dev {
    virtual ~CodeGen_PIM_Dev();

    virtual void add_kernel(Stmt stmt, const std::string &name, const std::vector<DeviceArgument> &args) = 0;

    virtual void init_module() = 0;

    virtual void compile_to_src() = 0;

    static bool is_pim_var(const std::string &name);
    static bool is_pim_bank_var(const std::string &name);
    static bool is_pim_thread_var(const std::string &name);

    virtual bool kernel_run_takes_types() const {
        return false;
    }

    virtual std::string api_unique_name() = 0;
};



}
}
#endif
