#ifndef HALIDE_CODEGEN_UPMEMDPU_H
#define HALIDE_CODEGEN_UPMEMDPU_H

#include <string>
#include <vector>

#include "CodeGen_C.h"
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
    std::string print_type(Type type, AppendSpaceIfNeeded append_space = DoNotAppendSpace);
    std::string print_reinterpret(Type type, const Expr &e);
    std::string print_extern_call(const Call *op);
    std::string print_array_access(const std::string &name,
                                    const Type &type,
                                    const std::string &id_index);
    void add_vector_typedefs(const std::set<Type> &vector_types);

    std::string get_memory_space(const std::string &);

    std::string shared_name;

    void visit(const For *);
    void visit(const Ramp *op);
    void visit(const Broadcast *op);
    void visit(const Call *op);
    void visit(const Load *op);
    void visit(const Store *op);
    void visit(const Cast *op);
    void visit(const Select *op);
    void visit(const EQ *);
    void visit(const NE *);
    void visit(const LT *);
    void visit(const LE *);
    void visit(const GT *);
    void visit(const GE *);
    void visit(const Allocate *op);
    void visit(const Free *op);
    void visit(const AssertStmt *op);
    void visit(const Shuffle *op);
    void visit(const Min *op);
    void visit(const Max *op);
    void visit(const Atomic *op);
};
}  // namespace Internal
}  // namespace Halide

#endif
