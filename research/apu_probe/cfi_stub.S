/*
 * CFI stubs for non-LTO modules on CFI-enabled kernels.
 *
 * __cfi_check: page-aligned, just returns (allows all checks).
 * __cfi_jt_init_module / __cfi_jt_cleanup_module: data pointers
 *   that cfi_init() dereferences to replace mod->init / mod->exit.
 *   We point them at our real init/exit functions.
 */

.text
.globl __cfi_check
.type __cfi_check, @function
__cfi_check:
    hint #34                  /* bti c */
    ret
.size __cfi_check, . - __cfi_check

.weak __cfi_check_fail
.globl __cfi_check_fail
.type __cfi_check_fail, @function
__cfi_check_fail:
    hint #34                  /* bti c */
    ret
.size __cfi_check_fail, . - __cfi_check_fail

/*
 * cfi_init() does: init = find_kallsyms_symbol_value(mod, "__cfi_jt_init_module");
 *                  if (init) mod->init = *init;
 * So __cfi_jt_init_module must be a pointer TO init_module.
 * Same for cleanup.
 *
 * Use .ref.data section to avoid section mismatch warnings
 * (.ref.data is whitelisted for cross-section references).
 */
.section .ref.data, "aw", @progbits
.globl __cfi_jt_init_module
.type __cfi_jt_init_module, @object
.size __cfi_jt_init_module, 8
__cfi_jt_init_module:
    .quad init_module

.globl __cfi_jt_cleanup_module
.type __cfi_jt_cleanup_module, @object
.size __cfi_jt_cleanup_module, 8
__cfi_jt_cleanup_module:
    .quad cleanup_module
