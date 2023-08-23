#include <algorithm>
#include <array>
#include <sstream>
#include <utility>

#include "CodeGen_UPMEM_DPU.h"

namespace Halide {
namespace Internal {

using std::ostringstream;
using std::sort;
using std::string;
using std::vector;


CodeGen_UPMEM_DPU_Dev::CodeGen_UPMEM_DPU_Dev(const Target &target): cgen(src_stream, target) {}

void CodeGen_UPMEM_DPU_Dev::add_kernel(Stmt stmt, const std::string &name, const std::vector<DeviceArgument> &args) {
    debug(2) << "CodeGen_UPMEM_DPU_Dev::compile" << name << "\n";

    cgen.add_kernel(stmt, name, args);
}

void CodeGen_UPMEM_DPU_Dev::init_module() {
    // wipe the internal kernel source
    src_stream.str("");
    src_stream.clear();
}

void CodeGen_UPMEM_DPU_Dev::compile_to_src() {
    // TODO: compile the src_stream into binary
    string str = src_stream.str();
    debug(1) << "UPMEM kernel:\n"
             << str << "\n";
}


bool CodeGen_UPMEM_DPU_Dev::kernel_run_takes_types() const {
    return true;
}

std::string CodeGen_UPMEM_DPU_Dev::api_unique_name() {
    return "upmem";
}


void CodeGen_UPMEM_DPU_C::add_kernel(Stmt stmt,
                const std::string &name,
                const std::vector<DeviceArgument> &args) {
    
    print(stmt);

    return;
}

std::string CodeGen_UPMEM_DPU_C::print_type(Type type, AppendSpaceIfNeeded append_space) {
    return "";
}

std::string CodeGen_UPMEM_DPU_C::print_reinterpret(Type type, const Expr &e) {
    return "";
}

std::string CodeGen_UPMEM_DPU_C::print_extern_call(const Call *op) {
    return "";
}

std::string CodeGen_UPMEM_DPU_C::print_array_access(const std::string &name,
                                const Type &type,
                                const std::string &id_index) {
    return "";
}

void CodeGen_UPMEM_DPU_C::add_vector_typedefs(const std::set<Type> &vector_types) {
    return;
}

std::string CodeGen_UPMEM_DPU_C::get_memory_space(const std::string &) {
    return "";
}

void CodeGen_UPMEM_DPU_C::visit(const For *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Ramp *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Broadcast *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Call *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Load *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Store *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Cast *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Select *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const EQ *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const NE *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const LT *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const LE *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const GT *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const GE *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Allocate *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Free *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const AssertStmt *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Shuffle *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Min *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Max *op) {
    CodeGen_C::visit(op);
}

void CodeGen_UPMEM_DPU_C::visit(const Atomic *op) {
    CodeGen_C::visit(op);
}

void upmem_compile_to_binary(string path) {

}


}
}