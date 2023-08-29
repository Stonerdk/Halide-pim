#ifndef HALIDE_CODEGEN_UPMEM_C_H
#define HALIDE_CODEGEN_UPMEM_C_H

#include "CodeGen_C.h"

namespace Halide {
namespace Internal {

using std::vector;
using std::string;
using std::pair;

class CodeGen_UPMEM_C : public CodeGen_C {

public:
    CodeGen_UPMEM_C(std::string fname,
                    std::ostream &s_host_c,
                    std::ostream &s_host_h,
                    std::ostream &s_kernel,
                    Target t);
    ~CodeGen_UPMEM_C();

    void compile(const Module &module);
protected:
    using CodeGen_C::visit;
    using CodeGen_C::compile;
    using CodeGen_C::stream;

    std::ostream& stream_host_h;
    std::ostream& stream_kernel;
    std::string fname = "";

    void compile(const LoweredFunc &f, const MetadataNameMap &metadata_name_map);

    bool define_global = true;
    string intercepted_unique_name;
    vector<pair<string, string>> globals;

    void emit_global_variables();

    void visit(const LetStmt *op) override;
    void visit(const AssertStmt *op) override;
    void visit(const IfThenElse *op) override;

    void intercept_unique_name(std::string& name) {
        intercepted_unique_name = name;
    }
    void release_unique_name() {
        intercepted_unique_name = "";
    }
    string print_assignment(Type t, const std::string &rhs) override;
};

}
}

#endif
