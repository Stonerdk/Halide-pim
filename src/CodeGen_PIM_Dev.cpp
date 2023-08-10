#include "CodeGen_PIM_Dev.h"

namespace Halide {
namespace Internal {

CodeGen_PIM_Dev::~CodeGen_PIM_Dev() = default;

bool CodeGen_PIM_Dev::is_pim_var(const std::string &name) {
    return is_pim_bank_var(name) || is_pim_thread_var(name);
}

bool CodeGen_PIM_Dev::is_pim_bank_var(const std::string &name) {
    return (ends_with(name, ".__bank_id_x") ||
            ends_with(name, ".__bank_id_y") ||
            ends_with(name, ".__bank_id_z") ||
            ends_with(name, ".__bank_id_w"));
}

bool CodeGen_PIM_Dev::is_pim_thread_var(const std::string &name) {
    return (ends_with(name, ".__thread_id_x"));

}
}
}