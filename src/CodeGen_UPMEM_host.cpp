#include "CodeGen_UPMEM_host.h"
#include "CodeGen_Internal.h"
#include "CodeGen_C.h"
#include "FindIntrinsics.h"
#include "IRMutator.h"
#include "Scope.h"
#include "Substitute.h"
#include "string"

using std::string;
using std::replace;

namespace Halide {
namespace Internal {

/*
class CopyOffloader: public IRMutator {
public:
    using IRMutator::visit;
    Stmt visit(const For *op) override {
        if (op->for_type == ForType::PIMBank && ends_with(op->name, ".copy_to")) {
            return op;
        } else if (op->for_type == ForType::PIMBank && ends_with(op->name, ".copy_from")) {
            return op;
        } else {
            return IRMutator::visit(op);
        }
    }
};
*/
extern "C" unsigned char halide_internal_initmod_inlined_c[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntime_h[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntimeCuda_h[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntimeHexagonHost_h[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntimeMetal_h[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntimeOpenCL_h[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntimeOpenGLCompute_h[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntimeQurt_h[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntimeD3D12Compute_h[];
extern "C" unsigned char halide_internal_runtime_header_HalideRuntimeWebGPU_h[];
extern "C" unsigned char halide_c_template_CodeGen_C_prologue[];
extern "C" unsigned char halide_c_template_CodeGen_C_vectors[];

const char *const kDefineMustUseResult = R"INLINE_CODE(#ifndef HALIDE_MUST_USE_RESULT
#ifdef __has_attribute
#if __has_attribute(nodiscard)
#define HALIDE_MUST_USE_RESULT [[nodiscard]]
#elif __has_attribute(warn_unused_result)
#define HALIDE_MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define HALIDE_MUST_USE_RESULT
#endif
#else
#define HALIDE_MUST_USE_RESULT
#endif
#endif
)INLINE_CODE";

class TypeInfoGatherer : public IRGraphVisitor {
private:
    using IRGraphVisitor::include;
    using IRGraphVisitor::visit;

    void include_type(const Type &t) {
        if (t.is_vector()) {
            if (t.is_bool()) {
                // bool vectors are always emitted as uint8 in the C++ backend
                // TODO: on some architectures, we could do better by choosing
                // a bitwidth that matches the other vectors in use; EliminateBoolVectors
                // could be used for this with a bit of work.
                vector_types_used.insert(UInt(8).with_lanes(t.lanes()));
            } else if (!t.is_handle()) {
                // Vector-handle types can be seen when processing (e.g.)
                // require() statements that are vectorized, but they
                // will all be scalarized away prior to use, so don't emit
                // them.
                vector_types_used.insert(t);
                if (t.is_int()) {
                    // If we are including an int-vector type, also include
                    // the same-width uint-vector type; there are various operations
                    // that can use uint vectors for intermediate results (e.g. lerp(),
                    // but also Mod, which can generate a call to abs() for int types,
                    // which always produces uint results for int inputs in Halide);
                    // it's easier to just err on the side of extra vectors we don't
                    // use since they are just type declarations.
                    vector_types_used.insert(t.with_code(halide_type_uint));
                }
            }
        }
    }

    void include_lerp_types(const Type &t) {
        if (t.is_vector() && t.is_int_or_uint() && (t.bits() >= 8 && t.bits() <= 32)) {
            include_type(t.widen());
        }
    }

protected:
    void include(const Expr &e) override {
        include_type(e.type());
        IRGraphVisitor::include(e);
    }

    // GCC's __builtin_shuffle takes an integer vector of
    // the size of its input vector. Make sure this type exists.
    void visit(const Shuffle *op) override {
        vector_types_used.insert(Int(32, op->vectors[0].type().lanes()));
        IRGraphVisitor::visit(op);
    }

    void visit(const For *op) override {
        for_types_used.insert(op->for_type);
        IRGraphVisitor::visit(op);
    }

    void visit(const Ramp *op) override {
        include_type(op->type.with_lanes(op->lanes));
        IRGraphVisitor::visit(op);
    }

    void visit(const Broadcast *op) override {
        include_type(op->type.with_lanes(op->lanes));
        IRGraphVisitor::visit(op);
    }

    void visit(const Cast *op) override {
        include_type(op->type);
        IRGraphVisitor::visit(op);
    }

    void visit(const Call *op) override {
        include_type(op->type);
        if (op->is_intrinsic(Call::lerp)) {
            // lower_lerp() can synthesize wider vector types.
            for (const auto &a : op->args) {
                include_lerp_types(a.type());
            }
        } else if (op->is_intrinsic()) {
            Expr lowered = lower_intrinsic(op);
            if (lowered.defined()) {
                lowered.accept(this);
                return;
            }
        }

        IRGraphVisitor::visit(op);
    }

public:
    std::set<ForType> for_types_used;
    std::set<Type> vector_types_used;
};

class ExternCallPrototypes : public IRGraphVisitor {
    struct NamespaceOrCall {
        const Call *call;  // nullptr if this is a subnamespace
        std::map<string, NamespaceOrCall> names;
        NamespaceOrCall(const Call *call = nullptr)
            : call(call) {
        }
    };
    std::map<string, NamespaceOrCall> c_plus_plus_externs;
    std::map<string, const Call *> c_externs;
    std::set<std::string> processed;
    std::set<std::string> internal_linkage;
    std::set<std::string> destructors;

    using IRGraphVisitor::visit;

    void visit(const Call *op) override {
        IRGraphVisitor::visit(op);

        if (!processed.count(op->name)) {
            if (op->call_type == Call::Extern || op->call_type == Call::PureExtern) {
                c_externs.insert({op->name, op});
            } else if (op->call_type == Call::ExternCPlusPlus) {
                std::vector<std::string> namespaces;
                std::string name = extract_namespaces(op->name, namespaces);
                std::map<string, NamespaceOrCall> *namespace_map = &c_plus_plus_externs;
                for (const auto &ns : namespaces) {
                    auto insertion = namespace_map->insert({ns, NamespaceOrCall()});
                    namespace_map = &insertion.first->second.names;
                }
                namespace_map->insert({name, NamespaceOrCall(op)});
            }
            processed.insert(op->name);
        }

        if (op->is_intrinsic(Call::register_destructor)) {
            internal_assert(op->args.size() == 2);
            const StringImm *fn = op->args[0].as<StringImm>();
            internal_assert(fn);
            destructors.insert(fn->value);
        }
    }

    void visit(const Allocate *op) override {
        IRGraphVisitor::visit(op);
        if (!op->free_function.empty()) {
            destructors.insert(op->free_function);
        }
    }

    void emit_function_decl(std::ostream &stream, const Call *op, const std::string &name) const {
        // op->name (rather than the name arg) since we need the fully-qualified C++ name
        if (internal_linkage.count(op->name)) {
            stream << "static ";
        }
        stream << type_to_c_type(op->type, /* append_space */ true) << name << "(";
        if (function_takes_user_context(name)) {
            stream << "void *";
            if (!op->args.empty()) {
                stream << ", ";
            }
        }
        for (size_t i = 0; i < op->args.size(); i++) {
            if (i > 0) {
                stream << ", ";
            }
            if (op->args[i].as<StringImm>()) {
                stream << "const char *";
            } else {
                stream << type_to_c_type(op->args[i].type(), true);
            }
        }
        stream << ");\n";
    }

    void emit_namespace_or_call(std::ostream &stream, const NamespaceOrCall &ns_or_call, const std::string &name) const {
        if (ns_or_call.call == nullptr) {
            stream << "namespace " << name << " {\n";
            for (const auto &ns_or_call_inner : ns_or_call.names) {
                emit_namespace_or_call(stream, ns_or_call_inner.second, ns_or_call_inner.first);
            }
            stream << "} // namespace " << name << "\n";
        } else {
            emit_function_decl(stream, ns_or_call.call, name);
        }
    }

public:
    ExternCallPrototypes() {
        // Make sure we don't catch calls that are already in the global declarations
        const char *strs[] = {(const char *)halide_c_template_CodeGen_C_prologue,
                              (const char *)halide_internal_runtime_header_HalideRuntime_h,
                              (const char *)halide_internal_initmod_inlined_c};
        for (const char *str : strs) {
            size_t j = 0;
            for (size_t i = 0; str[i]; i++) {
                char c = str[i];
                if (c == '(' && i > j + 1) {
                    // Could be the end of a function_name.
                    string name(str + j + 1, i - j - 1);
                    processed.insert(name);
                }

                if (('A' <= c && c <= 'Z') ||
                    ('a' <= c && c <= 'z') ||
                    c == '_' ||
                    ('0' <= c && c <= '9')) {
                    // Could be part of a function name.
                } else {
                    j = i;
                }
            }
        }
    }

    void set_internal_linkage(const std::string &name) {
        internal_linkage.insert(name);
    }

    bool has_c_declarations() const {
        return !c_externs.empty() || !destructors.empty();
    }

    bool has_c_plus_plus_declarations() const {
        return !c_plus_plus_externs.empty();
    }

    void emit_c_declarations(std::ostream &stream) const {
        for (const auto &call : c_externs) {
            emit_function_decl(stream, call.second, call.first);
        }
        for (const auto &d : destructors) {
            stream << "void " << d << "(void *, void *);\n";
        }
        stream << "\n";
    }

    void emit_c_plus_plus_declarations(std::ostream &stream) const {
        for (const auto &ns_or_call : c_plus_plus_externs) {
            emit_namespace_or_call(stream, ns_or_call.second, ns_or_call.first);
        }
        stream << "\n";
    }
};

CodeGen_UPMEM_C::CodeGen_UPMEM_C(std::string fname,
                    std::ostream &s_host_c,
                    std::ostream &s_host_h,
                    std::ostream &s_kernel,
                    Target t): 
                CodeGen_C(s_host_c, t, OutputKind::CImplementation), stream_host_h(s_host_h), stream_kernel(s_kernel) {

    size_t found_slash = fname.find_last_of("/\\");
    this->fname = found_slash != string::npos ? fname.substr(found_slash + 1) : fname;

    stream_host_h << "#ifndef HALIDE_" << c_print_name(this->fname) << "\n"
            << "#define HALIDE_" << c_print_name(this->fname) << "\n"
            << "#include <stdint.h>\n\n"
            << "struct halide_buffer_t;\n"
            << "struct halide_filter_metadata_t;\n"
            << "\n";
    forward_declared.insert(type_of<halide_buffer_t *>().handle_type);
    forward_declared.insert(type_of<halide_filter_metadata_t *>().handle_type);

    stream
        << halide_c_template_CodeGen_C_prologue << "\n"
        << halide_internal_runtime_header_HalideRuntime_h << "\n"
        << halide_internal_initmod_inlined_c << "\n";
    stream << "\n";

    stream << kDefineMustUseResult << "\n";
    stream << "#ifndef HALIDE_FUNCTION_ATTRS\n";
    stream << "#define HALIDE_FUNCTION_ATTRS\n";
    stream << "#endif\n";
}

CodeGen_UPMEM_C::~CodeGen_UPMEM_C() {
    stream_host_h << "#endif\n";
}


void CodeGen_UPMEM_C::compile(const Module &input) {
    stream << "\n#include \"" << fname + "_host.h" << "\"\n";

    Module new_module = split_module(input);
  
    add_platform_prologue();
    TypeInfoGatherer type_info;
    for (const auto &f : new_module.functions()) {
        if (f.body.defined()) 
            f.body.accept(&type_info);
    }
    uses_gpu_for_loops = (type_info.for_types_used.count(ForType::GPUBlock) ||
                          type_info.for_types_used.count(ForType::GPUThread) ||
                          type_info.for_types_used.count(ForType::GPULane));
    if (output_kind != CPlusPlusFunctionInfoHeader) {
        stream << "\n";
        for (const auto &f : new_module.functions()) {
            for (const auto &arg : f.args) {
                forward_declare_type_if_needed(arg.type);
            }
        }
        stream << "\n";
    }
    
    if (!is_header_or_extern_decl()) {
        add_vector_typedefs(type_info.vector_types_used);
        ExternCallPrototypes e;
        for (const auto &f : new_module.functions()) {
            f.body.accept(&e);
            if (f.linkage == LinkageType::Internal) {
                e.set_internal_linkage(f.name);
            }
        }
        if (e.has_c_declarations()) {
            set_name_mangling_mode(NameMangling::C);
            e.emit_c_declarations(stream);
        }
    }

    for (const auto &b : new_module.buffers()) {
        compile(b);
    }
    
    const auto metadata_name_map = new_module.get_metadata_name_map();
    for (const auto &f : new_module.functions()) {
        compile(f, metadata_name_map);
    }

    emit_global_variables();
}

// Internal module transformer
Module CodeGen_UPMEM_C::split_module(const Module& m) {
    // this is initialize part
    class SplitProducerConsumer : public IRMutator {
    public:
        using IRMutator::visit;
        Stmt pc;
        Stmt visit(const ProducerConsumer *op) override {
            pc = op->body;
            return Evaluate::make(0);
        }
    };

    Module new_module { m.name(), m.target() };
    for (const auto &f : m.functions()) {
        SplitProducerConsumer spc;
        auto init_statement = spc.mutate(f.body); // 그래 이거 안될듯

        if (spc.pc.defined()) {
            LoweredFunc init_f(f.name, f.args, init_statement, f.linkage);
            LoweredFunc lowered_f(f.name + "_produce", std::vector<LoweredArgument>(), spc.pc, f.linkage);
            new_module.append(init_f);
            new_module.append(lowered_f);
        } else {
            new_module.append(f);
        }
    }
    return new_module;
}

void CodeGen_UPMEM_C::compile(const LoweredFunc &f, const MetadataNameMap &metadata_name_map) {
    const std::vector<LoweredArgument> &args = f.args;

    have_user_context = false;
    for (const auto &arg : args) {
        // TODO: check that its type is void *?
        have_user_context |= (arg.name == "__user_context");
    }

    NameMangling name_mangling = f.name_mangling;
    if (name_mangling == NameMangling::Default) {
        name_mangling = (target.has_feature(Target::CPlusPlusMangling) || output_kind == CPlusPlusFunctionInfoHeader ? NameMangling::CPlusPlus : NameMangling::C);
    }

    set_name_mangling_mode(name_mangling);

    std::vector<std::string> namespaces;
    std::string simple_name = c_print_name(extract_namespaces(f.name, namespaces), false);
    if (!is_c_plus_plus_interface()) {
        user_assert(namespaces.empty()) << "Namespace qualifiers not allowed on function name if not compiling with Target::CPlusPlusNameMangling.\n";
    }

    if (!namespaces.empty()) {
        for (const auto &ns : namespaces) {
            stream << "namespace " << ns << " {\n";
            stream_host_h << "namesapce " << ns << "{\n";
        }
        stream << "\n";
        stream_host_h << "\n";
    }

    if (output_kind != CPlusPlusFunctionInfoHeader) {
        const auto emit_arg_decls = [&](std::ostream& oss, const Type &ucon_type = Type()) {
            const char *comma = "";
            for (const auto &arg : args) {
                oss << comma;
                if (arg.is_buffer()) {
                    oss << "struct halide_buffer_t *_"
                           << print_name(arg.name)
                           << "_buffer";
                } else {
                    // If this arg is the user_context value, *and* ucon_type is valid,
                    // use ucon_type instead of arg.type.
                    const Type &t = (arg.name == "__user_context" && ucon_type.bits() != 0) ? ucon_type : arg.type;
                    oss << print_type(t, AppendSpace) << print_name(arg.name);
                }
                comma = ", ";
            }
        };

        // Emit the function prototype
        if (f.linkage == LinkageType::Internal) {
            // If the function isn't public, mark it static.
            stream << "static ";
            stream_host_h << "static ";

        }
        stream << "HALIDE_FUNCTION_ATTRS\n";
        stream << "int " << simple_name << "(";
        stream_host_h << "HALIDE_FUNCTION_ATTRS\n";
        stream_host_h << "int " << simple_name << "(";
        emit_arg_decls(stream);
        emit_arg_decls(stream_host_h);

        stream_host_h << ");\n\n";

        stream << ") ";
        open_scope();

        if (uses_gpu_for_loops) {
            stream << get_indent() << "halide_error("
                    << (have_user_context ? "const_cast<void *>(__user_context)" : "nullptr")
                    << ", \"C++ Backend does not support gpu_blocks() or gpu_threads() yet, "
                    << "this function will always fail at runtime\");\n";
            stream << get_indent() << "return halide_error_code_device_malloc_failed;\n";
        } else {
            stream << get_indent() << "void * const _ucon = "
                    << (have_user_context ? "const_cast<void *>(__user_context)" : "nullptr")
                    << ";\n";
            stream << get_indent() << "halide_maybe_unused(_ucon);\n";

            for (const auto &arg : args) {
                string argName = print_name(arg.name);
                if (arg.is_buffer()) {
                    argName = print_name(arg.name) + "_buffer";
                    globals.push_back({ "struct halide_buffer_t *", argName });
                } else {
                    globals.push_back({ print_type(arg.type), argName });
                }
                stream << get_indent() << argName << " = _" << argName << ";\n";
            }

            Stmt body_to_print = preprocess_function_body(f.body);
            print(body_to_print);

            // Return success.
            stream << get_indent() << "return 0;\n";
            cache.clear();
        }

        // Ensure we use open/close_scope, so that the cache doesn't try to linger
        // across function boundaries for internal closures.
        close_scope("");



        if (f.linkage == LinkageType::ExternalPlusArgv || f.linkage == LinkageType::ExternalPlusMetadata) {
            // Emit the argv version
            emit_argv_wrapper(simple_name, args);
        }

        if (f.linkage == LinkageType::ExternalPlusMetadata) {
            // Emit the metadata.
            emit_metadata_getter(simple_name, args, metadata_name_map);
        }
    } else {
        if (f.linkage != LinkageType::Internal) {
            emit_constexpr_function_info(simple_name, args, metadata_name_map);
        }
    }

    if (!namespaces.empty()) {
        stream << "\n";
        stream_host_h << "\n";
        for (size_t i = namespaces.size(); i > 0; i--) {
            stream << "}  // namespace " << namespaces[i - 1] << "\n";
            stream_host_h << "}  // namespace " << namespaces[i - 1] << "\n";
        }
        stream << "\n";
        stream_host_h << "\n";
    }
}

void CodeGen_UPMEM_C::emit_global_variables() {
    for (auto g : globals) {
        stream_host_h << g.first << " " << g.second << ";\n";
    }
}

void CodeGen_UPMEM_C::visit(const LetStmt *op) {
    if (define_global && op->value.as<Call>()) {
        string temp_value = op->name;
        replace(temp_value.begin(), temp_value.end(), '.', '_');
        temp_value = "_" + temp_value;
        intercept_unique_name(temp_value);
        string id_value = print_expr(op->value);
        release_unique_name();

        const Call* c = op->value.as<Call>();
        globals.push_back({ print_type(c->type), temp_value });
        auto new_var = Variable::make(c->type, temp_value);
        Stmt body = substitute(op->name, new_var, op->body);
        body.accept(this);
    } else {
        CodeGen_C::visit(op);
    }
}

void CodeGen_UPMEM_C::visit(const AssertStmt *op) {
    bool past_define_global = define_global;
    define_global = false;
    CodeGen_C::visit(op);
    define_global = past_define_global;
}

void CodeGen_UPMEM_C::visit(const IfThenElse *op) {
    define_global = false;
    CodeGen_C::visit(op);
}

string CodeGen_UPMEM_C::print_assignment(Type t, const std::string &rhs) {
    if (define_global) {
        id = intercepted_unique_name.empty() ? unique_name('G') : intercepted_unique_name;
        stream << id << " = " << rhs << ";\n";
        cache[rhs] = id;
        return id;
    } else {
        return CodeGen_C::print_assignment(t, rhs);
    }
}

}
}