#ifndef HALIDE_CODEGEN_UPMEMDPU_H
#define HALIDE_CODEGEN_UPMEMDPU_H

#include <string>
#include <vector>

#include "CodeGen_C.h"
#include "CodeGen_PIM_Dev.h"
#include "DeviceArgument.h"
#include "Expr.h"

namespace Halide {
namespace Internal {

class CodeGen_UPMEM_DPU_C: public CodeGen_C {
public:
    CodeGen_UPMEM_DPU_C(std::ostream &s, Target t)
        : CodeGen_C(s, t) {
    }
    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args);

protected:
    using CodeGen_C::visit;
    std::string print_type(Type type, AppendSpaceIfNeeded append_space = DoNotAppendSpace) override;
    std::string print_reinterpret(Type type, const Expr &e) override;
    std::string print_extern_call(const Call *op) override;
    std::string print_array_access(const std::string &name,
                                    const Type &type,
                                    const std::string &id_index);
    void add_vector_typedefs(const std::set<Type> &vector_types) override;

    std::string get_memory_space(const std::string &);

    std::string shared_name;

    void visit(const For *) override;
    void visit(const Ramp *op) override;
    void visit(const Broadcast *op) override;
    void visit(const Call *op) override;
    void visit(const Load *op) override;
    void visit(const Store *op) override;
    void visit(const Cast *op) override;
    void visit(const Select *op) override;
    void visit(const EQ *) override;
    void visit(const NE *) override;
    void visit(const LT *) override;
    void visit(const LE *) override;
    void visit(const GT *) override;
    void visit(const GE *) override;
    void visit(const Allocate *op) override;
    void visit(const Free *op) override;
    void visit(const AssertStmt *op) override;
    void visit(const Shuffle *op) override;
    void visit(const Min *op) override;
    void visit(const Max *op) override;
    void visit(const Atomic *op) override;
};


class CodeGen_UPMEM_DPU_Dev : public CodeGen_PIM_Dev {
public:
    CodeGen_UPMEM_DPU_Dev(const Target &target);

    void add_kernel(Stmt stmt, const std::string &name, const std::vector<DeviceArgument> &args) override;

    void init_module() override;

    void compile_to_src() override;

    bool kernel_run_takes_types() const override;

    std::string api_unique_name() override;
protected:
    std::ostringstream src_stream;
    CodeGen_UPMEM_DPU_C cgen;
};
}  // namespace Internal
}  // namespace Halide

#endif
