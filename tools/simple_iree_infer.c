// simple_iree_infer.c
// Minimal IREE inference runner with in-place bf16 input update and printing.

#include <stdio.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/api.h"
#include "iree/io/stdio_stream.h"
#include "iree/modules/hal/types.h"
#include "iree/tooling/context_util.h"
#include "iree/tooling/function_io.h"
#include "iree/tooling/function_util.h"
#include "iree/tooling/run_module.h"
#include "iree/vm/api.h"

// ------------------------- Flags -------------------------

IREE_FLAG(string, function, "main",
          "Name of an exported function to run.");

IREE_FLAG_LIST(
    string, input,
    "Input(s) to the function. Example:\n"
    "  --input=\"64xbf16=@models/bytes_1_to_64_bf16.bin\"");

IREE_FLAG(int32_t, iterations, 5,
          "Number of inference iterations.");

IREE_FLAG(int32_t, output_max_element_count, 32,
          "Maximum number of tensor elements to print.");

// ------------------------- Helpers -------------------------

// ------------------------- Main -------------------------

int main(int argc, char** argv) {
  IREE_TRACE_APP_ENTER();

  iree_flags_set_usage(
      "simple-iree-infer",
      "Minimal multi-iteration IREE inference runner.");

  iree_flags_parse_checked(IREE_FLAGS_PARSE_MODE_DEFAULT, &argc, &argv);

  iree_allocator_t host_allocator = iree_allocator_system();

  // Create instance
  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_tooling_create_instance(host_allocator, &instance));

  // Load modules from --module flags
  iree_tooling_module_list_t module_list;
  iree_tooling_module_list_initialize(&module_list);
  IREE_CHECK_OK(
      iree_tooling_load_modules_from_flags(instance, host_allocator,
                                           &module_list));

  iree_vm_module_t* main_module =
      iree_tooling_module_list_back(&module_list);

  // Create context + device
  iree_vm_context_t* context = NULL;
  iree_hal_device_t* device = NULL;
  iree_hal_allocator_t* device_allocator = NULL;
  IREE_CHECK_OK(
      iree_tooling_create_context_from_flags(
          instance, module_list.count, module_list.values,
          iree_string_view_empty(), host_allocator,
          &context, &device, &device_allocator));

  iree_tooling_module_list_reset(&module_list);

  // Lookup function
  iree_vm_function_t function;
  IREE_CHECK_OK(
      iree_vm_module_lookup_function_by_name(
          main_module, IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_make_cstring_view(FLAG_function), &function));

  // Parse inputs ONCE
  iree_vm_function_signature_t signature =
      iree_vm_function_signature(&function);
  iree_string_view_t arg_cconv, res_cconv;
  IREE_CHECK_OK(
      iree_vm_function_call_get_cconv_fragments(
          &signature, &arg_cconv, &res_cconv));

  iree_vm_list_t* inputs = NULL;
  IREE_CHECK_OK(
      iree_tooling_parse_variants(
          arg_cconv, FLAG_input_list(),
          device, device_allocator,
          host_allocator, &inputs));

  // Append async fence if needed
  iree_hal_fence_t* finish_fence = NULL;
  IREE_CHECK_OK(
      iree_tooling_append_async_fences(
          inputs, function, device,
          NULL, &finish_fence));

  // Outputs list
  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(
      iree_vm_list_create(
          iree_vm_make_undefined_type_def(),
          8, host_allocator, &outputs));

  // Wrap stdout
  iree_io_stream_t* stdout_stream = NULL;
  IREE_CHECK_OK(
      iree_io_stdio_stream_wrap(
          IREE_IO_STREAM_MODE_WRITABLE,
          stdout, false, host_allocator,
          &stdout_stream));

  fprintf(stdout, "EXEC @%.*s\n",
          (int)iree_vm_function_name(&function).size,
          iree_vm_function_name(&function).data);

  // ------------------ Inference Loop ------------------

  // Print inputs once
  fprintf(stdout, "INPUTS:\n");
  IREE_CHECK_OK(
      iree_tooling_print_variants(
          IREE_SV("input"),
          inputs,
          (iree_host_size_t)FLAG_output_max_element_count,
          stdout_stream,
          host_allocator));

  for (int iter = 0; iter < FLAG_iterations; ++iter) {
    iree_vm_list_clear(outputs);

    fprintf(stdout, "\n=== ITERATION %d ===\n", iter);

    // Invoke
    IREE_CHECK_OK(
        iree_vm_invoke(
            context, function,
            IREE_VM_INVOCATION_FLAG_NONE,
            NULL, inputs, outputs,
            host_allocator));

    // Wait if async
    if (finish_fence) {
      IREE_CHECK_OK(
          iree_hal_fence_wait(
              finish_fence, iree_infinite_timeout(),
              IREE_HAL_WAIT_FLAG_DEFAULT));
    }

    // Transfer outputs to host
    if (device) {
      iree_hal_buffer_params_t params = {
          .usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_MAPPING,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
                  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      };
      IREE_CHECK_OK(
          iree_tooling_transfer_variants(
              outputs, device, device_allocator,
              params, NULL, NULL));
    }

    // Print outputs
    fprintf(stdout, "OUTPUTS:\n");
    IREE_CHECK_OK(
        iree_tooling_print_variants(
            IREE_SV("result"),
            outputs,
            (iree_host_size_t)FLAG_output_max_element_count,
            stdout_stream,
            host_allocator));
  }

  // ------------------ Teardown ------------------

  iree_io_stream_release(stdout_stream);
  iree_vm_list_release(outputs);
  iree_hal_fence_release(finish_fence);
  iree_vm_list_release(inputs);
  iree_hal_allocator_release(device_allocator);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);

  IREE_TRACE_APP_EXIT(EXIT_SUCCESS);
  return EXIT_SUCCESS;
}
